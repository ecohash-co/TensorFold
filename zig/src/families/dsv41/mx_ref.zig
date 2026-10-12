//! Stage MX3's host references for tf-dsv41-mx-bench (no GPU): the original checkpoint's bytes (MXFP4 routed experts,
//! the bf16 router), float64 references of sampled pairs exactly as bench_mx_zigsplit.py's reference() computes them,
//! its router (sqrt(softplus) scores, top-k of score + bias, normalized and scaled), its rel_report, and the flat dump
//! files tools/dsv41-zig/mx/dump_to_bin.py writes from its --dump files.
const std = @import("std");
const mx = @import("mx_prompt.zig");

/// e2m1 codes (low nibble first in the checkpoint's bytes).
pub const e2m1 = [16]f64{ 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0 };

pub fn bf16ToF32(h: u16) f32 {
    return @bitCast(@as(u32, h) << 16);
}

/// Round to nearest even (__float2bfloat16_rn on non-NaN values).
pub fn f32ToBf16(f: f32) u16 {
    const u: u32 = @bitCast(f);
    if ((u & 0x7fffffff) > 0x7f800000) return @intCast((u >> 16) | 0x0040);
    return @intCast((u +% (0x7fff + ((u >> 16) & 1))) >> 16);
}

/// One row of an MXFP4 matrix: `k` values from words (k / 2 bytes, low nibble = even input) and e8m0 bytes (k / 32).
pub fn dequantRow(words: []const u8, e8m0: []const u8, out: []f64) void {
    for (out, 0..) |*o, i| {
        const b = words[i / 2];
        const code = if (i % 2 == 0) b & 0xf else b >> 4;
        o.* = e2m1[code] * std.math.ldexp(@as(f64, 1.0), @as(i32, e8m0[i / 32]) - 127);
    }
}

/// The SwiGLU the reference applies (bench_mx_zigsplit.reference): min(g, L) silu'd times clip(u, -L, L), in float64.
pub fn swigluRef(g: f64, u: f64, limit: f64) f64 {
    const gv = @min(g, limit);
    const uv = std.math.clamp(u, -limit, limit);
    return gv / (1.0 + @exp(-gv)) * uv;
}

/// rel_report: max |got - ref| over entries finite in both, over max |ref| (floored), and the non-finite counts.
pub const Rel = struct {
    max_abs: f64 = 0,
    rel: f64 = 0,
    nonfinite_got: usize = 0,
    nonfinite_ref: usize = 0,
    ref_absmax: f64 = 0,

    /// rel_report of one pair, accumulated as the Python reference accumulates its samples (max of each, sums of counts).
    pub fn add(acc: *Rel, got: []const f64, ref: []const f64) void {
        const r = report(got, ref);
        acc.max_abs = @max(acc.max_abs, r.max_abs);
        acc.rel = if (std.math.isFinite(r.rel)) @max(acc.rel, r.rel) else std.math.inf(f64);
        acc.nonfinite_got += r.nonfinite_got;
        acc.nonfinite_ref += r.nonfinite_ref;
        acc.ref_absmax = @max(acc.ref_absmax, r.ref_absmax);
    }
};

pub fn report(got: []const f64, ref: []const f64) Rel {
    var r: Rel = .{};
    var any = false;
    for (got, ref) |g, f| {
        const fg = std.math.isFinite(g);
        const ff = std.math.isFinite(f);
        r.nonfinite_got += @intFromBool(!fg);
        r.nonfinite_ref += @intFromBool(!ff);
        if (!(fg and ff)) continue;
        any = true;
        r.max_abs = @max(r.max_abs, @abs(g - f));
        r.ref_absmax = @max(r.ref_absmax, @abs(f));
    }
    if (!any) {
        r.max_abs = std.math.inf(f64);
        r.rel = std.math.inf(f64);
        return r;
    }
    r.rel = r.max_abs / @max(r.ref_absmax, 1e-12);
    return r;
}

// -- the original checkpoint ------------------------------------------------------------------------------------------

const Shard = struct {
    fd: std.c.fd_t,
    base: u64,
    h: mx.Header,
};

/// The original DeepSeek-V4.1-Flash checkpoint (castor /var/tmp/models/DeepSeek-V4.1-Flash): model.safetensors.index.json
/// and the shards' headers, tensors read by pread (rows of 2-D tensors).
pub const Ckpt = struct {
    arena: std.heap.ArenaAllocator,
    dir: []const u8,
    map: std.json.ObjectMap,
    shards: std.StringHashMapUnmanaged(Shard) = .empty,
    read_bytes: usize = 0,

    pub fn open(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !Ckpt {
        var arena = std.heap.ArenaAllocator.init(gpa);
        errdefer arena.deinit();
        const a = arena.allocator();
        const path = try std.fs.path.join(a, &.{ dir, "model.safetensors.index.json" });
        const text = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(64 << 20));
        const v = try std.json.parseFromSliceLeaky(std.json.Value, a, text, .{});
        if (v != .object) return error.BadIndex;
        const wm = v.object.get("weight_map") orelse return error.BadIndex;
        if (wm != .object) return error.BadIndex;
        return .{ .arena = arena, .dir = try a.dupe(u8, dir), .map = wm.object };
    }

    pub fn close(ck: *Ckpt) void {
        var it = ck.shards.valueIterator();
        while (it.next()) |s| _ = std.c.close(s.fd);
        ck.arena.deinit();
        ck.* = undefined;
    }

    fn shard(ck: *Ckpt, name: []const u8) !*Shard {
        const fv = ck.map.get(name) orelse {
            std.log.err("checkpoint has no {s}", .{name});
            return error.NoSuchTensor;
        };
        if (fv != .string) return error.BadIndex;
        if (ck.shards.getPtr(fv.string)) |s| return s;
        const a = ck.arena.allocator();
        const path = try std.fmt.allocPrintSentinel(a, "{s}/{s}", .{ ck.dir, fv.string }, 0);
        const fd = std.c.open(path, .{ .ACCMODE = .RDONLY, .CLOEXEC = true });
        if (fd < 0) return error.FileNotFound;
        errdefer _ = std.c.close(fd);
        var lb: [8]u8 = undefined;
        try preadAll(fd, &lb, 0);
        const hl = std.mem.readInt(u64, &lb, .little);
        if (hl > 1 << 28) return error.BadSafetensors;
        const hj = try a.alloc(u8, @intCast(hl));
        try preadAll(fd, hj, 8);
        const end = std.c.lseek(fd, 0, std.c.SEEK.END);
        if (end < 0) return error.ReadFailed;
        const h = try mx.parseHeader(a, hj, @as(usize, @intCast(end)) - 8 - @as(usize, @intCast(hl)));
        try ck.shards.put(a, fv.string, .{ .fd = fd, .base = 8 + hl, .h = h });
        return ck.shards.getPtr(fv.string).?;
    }

    /// A tensor's header entry (dtype, shape).
    pub fn entry(ck: *Ckpt, name: []const u8) !mx.Entry {
        const s = try ck.shard(name);
        return s.h.names.get(name) orelse error.NoSuchTensor;
    }

    /// Rows [r0, r1) of a tensor (the whole of each row, row-major bytes) into `out` (its exact size).
    pub fn rows(ck: *Ckpt, name: []const u8, r0: usize, r1: usize, out: []u8) !void {
        const s = try ck.shard(name);
        const e = s.h.names.get(name) orelse return error.NoSuchTensor;
        if (e.rank == 0 or r1 > e.shape[0] or r0 > r1) return error.BadRows;
        const row = (e.end - e.begin) / e.shape[0];
        if (out.len != (r1 - r0) * row) return error.BadRows;
        try preadAll(s.fd, out, s.base + e.begin + r0 * row);
        ck.read_bytes += out.len;
    }

    /// The whole tensor (allocated in `a`).
    pub fn all(ck: *Ckpt, a: std.mem.Allocator, name: []const u8) ![]u8 {
        const e = try ck.entry(name);
        const out = try a.alloc(u8, e.end - e.begin);
        try ck.rows(name, 0, e.shape[0], out);
        return out;
    }
};

fn preadAll(fd: std.c.fd_t, out: []u8, off: u64) !void {
    var done: usize = 0;
    while (done < out.len) {
        const n = std.c.pread(fd, out.ptr + done, out.len - done, @intCast(off + done));
        if (n < 0) {
            if (std.c.errno(n) == .INTR) continue;
            return error.ReadFailed;
        }
        if (n == 0) return error.EndOfFile;
        done += @intCast(n);
    }
}

/// The dequantized rows [r0, r1) x columns [c0, c1) of an MXFP4 matrix `pre`.weight / `pre`.scale (k inputs a row),
/// float64 [r1 - r0, c1 - c0] in `a` (c0, c1 multiples of 32).
pub fn dequantBlock(a: std.mem.Allocator, ck: *Ckpt, pre: []const u8, r0: usize, r1: usize, c0: usize, c1: usize) ![]f64 {
    var nb: [256]u8 = undefined;
    const wn = try std.fmt.bufPrint(&nb, "{s}.weight", .{pre});
    const we = try ck.entry(wn);
    const k = we.shape[1] * 2;
    if (c0 % 32 != 0 or c1 % 32 != 0 or c1 > k) return error.BadColumns;
    const n = r1 - r0;
    const words = try a.alloc(u8, n * k / 2);
    defer a.free(words);
    try ck.rows(wn, r0, r1, words);
    var sb: [256]u8 = undefined;
    const sn = try std.fmt.bufPrint(&sb, "{s}.scale", .{pre});
    const scales = try a.alloc(u8, n * k / 32);
    defer a.free(scales);
    try ck.rows(sn, r0, r1, scales);
    const w = c1 - c0;
    const out = try a.alloc(f64, n * w);
    for (0..n) |r| dequantRow(words[r * k / 2 + c0 / 2 ..][0 .. w / 2], scales[r * k / 32 + c0 / 32 ..][0 .. w / 32], out[r * w ..][0..w]);
    return out;
}

fn dotF64(w: []const f64, x: []const f64) f64 {
    var s: f64 = 0;
    for (w, x) |a, b| s += a * b;
    return s;
}

/// One routed pair's float64 reference from the checkpoint bytes (bench_mx_zigsplit.reference): act over the main
/// columns (and the rest columns when `rest`), y = W2[dcols, half] xd.
pub const PairRef = struct { act: []f64, act_rest: []f64, y: []f64 };

pub fn pairRef(a: std.mem.Allocator, ck: *Ckpt, layer: u32, x: mx.Expect, e: usize, xrow: []const f32, xd: []const f32, rest: bool, limit: f64) !PairRef {
    const xs = try a.alloc(f64, xrow.len);
    defer a.free(xs);
    for (xs, xrow) |*o, v| o.* = v;
    var pb: [3][128]u8 = undefined;
    const pre_g = try std.fmt.bufPrint(&pb[0], "layers.{d}.ffn.experts.{d}.w1", .{ layer, e });
    const pre_u = try std.fmt.bufPrint(&pb[1], "layers.{d}.ffn.experts.{d}.w3", .{ layer, e });
    const pre_d = try std.fmt.bufPrint(&pb[2], "layers.{d}.ffn.experts.{d}.w2", .{ layer, e });
    const out: PairRef = .{ .act = try a.alloc(f64, x.mainW()), .act_rest = try a.alloc(f64, if (rest) x.restW() else 0), .y = try a.alloc(f64, x.downN()) };
    for ([_][2]usize{ x.main, x.rest }, [_][]f64{ out.act, out.act_rest }, 0..) |cols, dst, j| {
        if (j == 1 and !rest) continue;
        const wg = try dequantBlock(a, ck, pre_g, cols[0], cols[1], 0, x.hidden);
        defer a.free(wg);
        const wu = try dequantBlock(a, ck, pre_u, cols[0], cols[1], 0, x.hidden);
        defer a.free(wu);
        for (dst, 0..) |*o, c| o.* = swigluRef(dotF64(wg[c * x.hidden ..][0..x.hidden], xs), dotF64(wu[c * x.hidden ..][0..x.hidden], xs), limit);
    }
    const xdd = try a.alloc(f64, xd.len);
    defer a.free(xdd);
    for (xdd, xd) |*o, v| o.* = v;
    const w2 = try dequantBlock(a, ck, pre_d, x.dcols[0], x.dcols[1], x.half[0], x.half[1]);
    defer a.free(w2);
    const hw = x.halfW();
    for (out.y, 0..) |*o, c| o.* = dotF64(w2[c * hw ..][0..hw], xdd);
    return out;
}

/// The shared expert's float64 reference from the slice's exact bf16 cuts: act over its sw columns here (gate rows wg,
/// up rows: bf16 [sw, D] each), y = W_d [dn, half] xd.
pub fn sharedRef(wg: []const u16, up: []const u16, wd: []const u16, d: usize, half: usize, xrow: []const f32, xd: []const f32, act: []f64, y: []f64, limit: f64) void {
    for (act, 0..) |*o, c| {
        var g: f64 = 0;
        var u: f64 = 0;
        for (0..d) |i| {
            g += @as(f64, bf16ToF32(wg[c * d + i])) * xrow[i];
            u += @as(f64, bf16ToF32(up[c * d + i])) * xrow[i];
        }
        o.* = swigluRef(g, u, limit);
    }
    for (y, 0..) |*o, c| {
        var s: f64 = 0;
        for (0..half) |i| s += @as(f64, bf16ToF32(wd[c * half + i])) * xd[i];
        o.* = s;
    }
}

// -- the router (bench_mx_zigsplit.router) -----------------------------------------------------------------------------

/// The layer's gate: weight bf16 [experts, D] as fp32, bias fp32 [experts].
pub const Gate = struct { w: []f32, b: []f32 };

pub fn gate(a: std.mem.Allocator, ck: *Ckpt, layer: u32, experts: usize, d: usize) !Gate {
    var nb: [128]u8 = undefined;
    const wn = try std.fmt.bufPrint(&nb, "layers.{d}.ffn.gate.weight", .{layer});
    const we = try ck.entry(wn);
    if (!we.is("BF16", &.{ experts, d })) return error.UnexpectedGate;
    const raw = try ck.all(a, wn);
    defer a.free(raw);
    const w = try a.alloc(f32, experts * d);
    const h = std.mem.bytesAsSlice(u16, raw);
    for (w, 0..) |*o, i| o.* = bf16ToF32(std.mem.littleToNative(u16, h[i]));
    var bb: [128]u8 = undefined;
    const bn = try std.fmt.bufPrint(&bb, "layers.{d}.ffn.gate.bias", .{layer});
    const be = try ck.entry(bn);
    if (!be.is("F32", &.{experts})) return error.UnexpectedGate;
    const braw = try ck.all(a, bn);
    defer a.free(braw);
    const b = try a.alloc(f32, experts);
    @memcpy(std.mem.sliceAsBytes(b), braw);
    return .{ .w = w, .b = b };
}

/// torch.nn.functional.softplus (beta 1, threshold 20).
fn softplus(v: f32) f32 {
    return if (v > 20) v else std.math.log1p(@exp(v));
}

/// Rows [r0, r1) of x fp32 [*, d]: logits = x W^T (fp32), score = sqrt(softplus(logits)), picks = the top_k of
/// score + bias in descending order (slots 0 .. top_k - 1), wts = score[picks] / their sum * scaling; the shared expert
/// (id experts) in the last slot with weight 1.
pub fn route(x: []const f32, g: Gate, d: usize, experts: usize, top_k: usize, scaling: f32, r0: usize, r1: usize, picks: []i32, wts: []f32) void {
    const slots = top_k + 1;
    var score: [1024]f32 = undefined;
    const V = @Vector(16, f32);
    for (r0..r1) |r| {
        const xr = x[r * d ..][0..d];
        for (0..experts) |e| {
            const wr = g.w[e * d ..][0..d];
            var acc: V = @splat(0);
            var c: usize = 0;
            while (c + 16 <= d) : (c += 16) acc += @as(V, xr[c..][0..16].*) * @as(V, wr[c..][0..16].*);
            var l = @reduce(.Add, acc);
            while (c < d) : (c += 1) l += xr[c] * wr[c];
            score[e] = @sqrt(softplus(l));
        }
        var chosen: [32]usize = undefined;
        var tot: f32 = 0;
        for (0..top_k) |q| {
            var best: usize = 0;
            var bv: f32 = -std.math.inf(f32);
            for (0..experts) |e| {
                var taken = false;
                for (chosen[0..q]) |cc| taken = taken or cc == e;
                const v = score[e] + g.b[e];
                if (!taken and v > bv) {
                    bv = v;
                    best = e;
                }
            }
            chosen[q] = best;
            tot += score[best];
        }
        for (0..top_k) |q| {
            picks[r * slots + q] = @intCast(chosen[q]);
            wts[r * slots + q] = score[chosen[q]] / tot * scaling;
        }
        picks[r * slots + top_k] = @intCast(experts);
        wts[r * slots + top_k] = 1.0;
    }
}

// -- the Python bench's dumps ------------------------------------------------------------------------------------------

pub const DumpItem = struct { pair: u32, e: u32, x: []f32, act: []f32, xd: []f32, y: []f32 };

pub const Dump = struct {
    layer: u32,
    node: u32,
    rows: u32,
    items: []DumpItem,
    check: []const u8, // the recorded check dict (JSON)
};

/// A dump_to_bin.py file (in `a`): dims must be (d, main_w, half, dn).
pub fn parseDump(a: std.mem.Allocator, bytes: []const u8, dims: [4]usize) !Dump {
    if (bytes.len < 40 or !std.mem.eql(u8, bytes[0..8], "TFMXDMP1")) return error.NotAnMxDump;
    var h: [8]u32 = undefined;
    for (&h, 0..) |*v, i| v.* = std.mem.readInt(u32, bytes[8 + 4 * i ..][0..4], .little);
    for (0..4) |i| if (h[4 + i] != dims[i]) return error.MxDumpDims;
    const per = 8 + 4 * (dims[0] + dims[1] + dims[2] + dims[3]);
    var at: usize = 40;
    const items = try a.alloc(DumpItem, h[3]);
    for (items) |*it| {
        if (at + per > bytes.len) return error.MxDumpShort;
        it.pair = std.mem.readInt(u32, bytes[at..][0..4], .little);
        it.e = std.mem.readInt(u32, bytes[at + 4 ..][0..4], .little);
        var o = at + 8;
        const fields = [_]*[]f32{ &it.x, &it.act, &it.xd, &it.y };
        for (fields, 0..) |fp, i| {
            const v = try a.alloc(f32, dims[i]);
            @memcpy(std.mem.sliceAsBytes(v), bytes[o..][0 .. 4 * dims[i]]);
            fp.* = v;
            o += 4 * dims[i];
        }
        at += per;
    }
    if (at + 4 > bytes.len) return error.MxDumpShort;
    const n = std.mem.readInt(u32, bytes[at..][0..4], .little);
    if (at + 4 + n > bytes.len) return error.MxDumpShort;
    return .{ .layer = h[0], .node = h[1], .rows = h[2], .items = items, .check = bytes[at + 4 ..][0..n] };
}

/// A recorded check's number (act_rel, y_rel, ...) from the dump's JSON.
pub fn checkNumber(a: std.mem.Allocator, check: []const u8, key: []const u8) ?f64 {
    const v = std.json.parseFromSliceLeaky(std.json.Value, a, check, .{}) catch return null;
    if (v != .object) return null;
    const x = v.object.get(key) orelse return null;
    return switch (x) {
        .float => |f| f,
        .integer => |i| @floatFromInt(i),
        else => null,
    };
}

// -- host tests (no GPU) ----------------------------------------------------------------------------------------------

const testing = std.testing;

test "e2m1 rows, bf16 rounding, the reference SwiGLU and rel_report" {
    // byte 0x2f = code 0xf (-6) at the even input, 0x2 (1.0) at the odd one; e8m0 128 = x2
    var out: [32]f64 = undefined;
    var words: [16]u8 = @splat(0);
    words[0] = 0xf7; // even: 7 (6.0), odd: 0xf (-6.0)
    words[1] = 0x21; // 0.5, 1.0
    dequantRow(&words, &.{128}, &out);
    try testing.expectEqual(@as(f64, 12.0), out[0]);
    try testing.expectEqual(@as(f64, -12.0), out[1]);
    try testing.expectEqual(@as(f64, 1.0), out[2]);
    try testing.expectEqual(@as(f64, 2.0), out[3]);
    try testing.expectEqual(@as(f64, 0.0), out[31]);
    try testing.expectEqual(@as(u16, 0x3f80), f32ToBf16(1.0));
    try testing.expectEqual(@as(u16, 0x3f80), f32ToBf16(1.00390625)); // a tie rounds to even
    try testing.expectEqual(@as(u16, 0x3f82), f32ToBf16(1.01171875));
    try testing.expectEqual(@as(f64, 0.0), swigluRef(0.0, 5.0, 10.0));
    try testing.expectApproxEqRel(@as(f64, 10.0 / (1.0 + @exp(-10.0)) * 10.0), swigluRef(50.0, 70.0, 10.0), 1e-15);
    try testing.expectApproxEqRel(@as(f64, 10.0 / (1.0 + @exp(-10.0)) * -10.0), swigluRef(50.0, -70.0, 10.0), 1e-15);
    const r = report(&.{ 1.0, 2.5, std.math.nan(f64) }, &.{ 1.0, 2.0, 4.0 });
    try testing.expectEqual(@as(f64, 0.5), r.max_abs);
    try testing.expectEqual(@as(f64, 0.25), r.rel);
    try testing.expectEqual(@as(usize, 1), r.nonfinite_got);
    var acc: Rel = .{};
    acc.add(&.{1.0}, &.{2.0});
    acc.add(&.{std.math.inf(f64)}, &.{std.math.inf(f64)});
    try testing.expect(std.math.isInf(acc.rel));
}

test "the router: top-k of score + bias, weights from the score alone, the shared expert last" {
    const d = 16;
    const e = 5;
    var w: [e * d]f32 = @splat(0);
    for (0..e) |i| w[i * d] = @floatFromInt(i); // logit e = e x[0]
    const b = [_]f32{ 0, 0, 0, 0, -10 }; // expert 4 has the best score but a bias that drops it
    var x: [d]f32 = @splat(0);
    x[0] = 1;
    var picks: [3]i32 = undefined;
    var wts: [3]f32 = undefined;
    route(&x, .{ .w = &w, .b = @constCast(&b) }, d, e, 2, 1.5, 0, 1, &picks, &wts);
    try testing.expectEqual([3]i32{ 3, 2, 5 }, picks);
    const s3 = @sqrt(std.math.log1p(@exp(@as(f32, 3))));
    const s2 = @sqrt(std.math.log1p(@exp(@as(f32, 2))));
    try testing.expectApproxEqRel(s3 / (s3 + s2) * 1.5, wts[0], 1e-6);
    try testing.expectEqual(@as(f32, 1.0), wts[2]);
}

test "a dump file round trip" {
    const a = testing.allocator;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const dims = [4]usize{ 4, 2, 3, 2 };
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(a);
    try buf.appendSlice(a, "TFMXDMP1");
    for ([_]u32{ 1, 2, 4096, 1, 4, 2, 3, 2, 77, 9 }) |v| try buf.appendSlice(a, std.mem.asBytes(&std.mem.nativeToLittle(u32, v)));
    for (0..11) |i| {
        const f: f32 = @floatFromInt(i);
        try buf.appendSlice(a, std.mem.asBytes(&f));
    }
    const js = "{\"act_rel\": 0.0087, \"pass\": true}";
    const n: u32 = js.len;
    try buf.appendSlice(a, std.mem.asBytes(&n));
    try buf.appendSlice(a, js);
    const dmp = try parseDump(arena.allocator(), buf.items, dims);
    try testing.expectEqual(@as(u32, 4096), dmp.rows);
    try testing.expectEqual(@as(u32, 9), dmp.items[0].e);
    try testing.expectEqual(@as(f32, 4), dmp.items[0].act[0]);
    try testing.expectEqual(@as(f32, 10), dmp.items[0].y[1]);
    try testing.expectEqual(@as(?f64, 0.0087), checkNumber(arena.allocator(), dmp.check, "act_rel"));
    try testing.expectError(error.MxDumpDims, parseDump(arena.allocator(), buf.items, .{ 4, 2, 3, 3 }));
}

// TF_MX_TEST_DUMP=<a dump_to_bin.py file> TF_MX_TEST_CKPT=<the original checkpoint> (both on castor; skipped when
// unset): this file's float64 reference of the dump's pairs reproduces the Python bench's recorded act_rel / y_rel
// (CPU only: a few MB of the checkpoint read).
test "the reference reproduces the Python bench's recorded check (castor files, opt-in)" {
    const dp = std.c.getenv("TF_MX_TEST_DUMP") orelse return error.SkipZigTest;
    const cp = std.c.getenv("TF_MX_TEST_CKPT") orelse return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = testing.io;
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, std.mem.span(dp), a, .limited(1 << 26));
    const dmp = try parseDump(a, bytes, .{ 5120, 512, 1152, 2560 });
    const cfg = @import("config.zig");
    var why: cfg.Why = .{};
    const c = try cfg.parse(a, cfg.test_config, &why);
    const x = try mx.Expect.of(c, .{ .rank = dmp.node % 2, .world = 2, .pair = dmp.node / 2, .parity = true }, dmp.layer);
    var ck = try Ckpt.open(testing.allocator, io, std.mem.span(cp));
    defer ck.close();
    var A: Rel = .{};
    var Y: Rel = .{};
    for (dmp.items) |it| {
        const r = try pairRef(a, &ck, dmp.layer, x, it.e, it.x, it.xd, false, c.swiglu_limit);
        const ga = try a.alloc(f64, it.act.len);
        for (ga, it.act) |*o, v| o.* = v;
        const gy = try a.alloc(f64, it.y.len);
        for (gy, it.y) |*o, v| o.* = v;
        A.add(ga, r.act);
        Y.add(gy, r.y);
    }
    const ra = checkNumber(a, dmp.check, "act_rel").?;
    const ry = checkNumber(a, dmp.check, "y_rel").?;
    std.debug.print("dump L{d} R{d}: act_rel {e} (recorded {e}), y_rel {e} (recorded {e})\n", .{ dmp.layer, dmp.rows, A.rel, ra, Y.rel, ry });
    try testing.expectApproxEqRel(ra, A.rel, 1e-6);
    try testing.expectApproxEqRel(ry, Y.rel, 1e-4);
}
