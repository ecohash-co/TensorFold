//! TF_DS_PROMPT_ROUTER (stage ZR1): an SP prompt chunk's MoE router on this node's own rows. Every node used to route
//! every row of the chunk (the fp32 casts, cuBLAS x.float() @ gate_w.float().t(), the route kernel) and use a quarter of
//! it; under SP rows each node routes the rows it mixed and the picks and weights (56 B a row) are all-gathered.
//!
//!   own   the gate's logits of the own rows by router_own.cu (straight from x bf16 and gate_w fp16, widened in the
//!         program exactly as today's casts), the route kernel on those rows, then picks and weights of every owner
//!         gathered in place (one NCCL all-gather of [rows, 2 slots] words)
//!   hide  (with own, needs TF_DS_PROMPT_OVL's comm stream) the FFN x gather on the comm stream while the own rows
//!         route; their picks' gather waits for it (comm.Order), so the router runs under X4's transfer
//!
//! With TF_DS_PROMPT_OVL=fxag, an own chunk's x gather is fxag's two exchanges where fxag fits (4,096-row passes), back
//! to back on one stream (the comm stream with hide), instead of fxag's split around today's router runs: own has no
//! such runs, its router needs only this node's x rows.
//!
//! Exact by construction, and checked at start. Today's logits come from cublasSgemm_v2 on runs of 2,048 rows
//! (prompt.xwTRows). Its kernel (cutlass_80_simt_sgemm_128x256_8x4_tn) splits K by M (stage ZX1: at M = 1,024, 93 % of
//! the logits differ), so cuBLAS on the own rows (M = 1,024 or 512) is not exact. But at a fixed M each element is a
//! fixed function of its own row and expert: split-K partitions [b_i, b_{i+1}) of K, each one FFMA chain over ascending
//! k from +0, added in partition order (serial split-K: D = p0, then D = p_i + D). router_own.cu runs exactly those
//! chains and adds for any rows with explicit _rn intrinsics. `calibrate` finds the partitions of the 2,048-row run
//! on this GPU and library by comparing every candidate with the library's own call, then checks own-row launches of
//! every variant against it; no match refuses to start (RouterOwnNotExact). The route kernel (_route, one program a
//! row reading its row's logits and the bias) and the casts are row-local, so the own rows' picks and weights are the
//! bytes today's whole launches write for them.
//!
//! The lever acts on an SP chunk only when every owner's rows lie inside one cuBLAS run of exactly 2,048 rows (`fits`:
//! a 2,048-row chunk, or a 4,096-row pass of two 2,048-row runs), a function of n and the pass's sub-runs alone, so the
//! same choice on every rank. Other chunks (rows2's other n, the head chunk past the cut, decode-sized chunks, image
//! spans) keep today's launches. Unset: today's launches everywhere.
const std = @import("std");
const cuda = @import("cuda");
const cublas = @import("cublas.zig");
const Owners = @import("prompt2d.zig").Owners;

pub const Levers = struct {
    own: bool = false,
    hide: bool = false,

    pub fn any(l: Levers) bool {
        return l.own;
    }

    /// The active levers' names, as the start line prints them.
    pub fn names(l: Levers, buf: []u8) []const u8 {
        var n: usize = 0;
        inline for (.{ "own", "hide" }) |f| if (@field(l, f)) {
            if (n > 0 and n < buf.len) {
                buf[n] = ',';
                n += 1;
            }
            const k = @min(f.len, buf.len - n);
            @memcpy(buf[n..][0..k], f[0..k]);
            n += k;
        };
        return buf[0..n];
    }
};

/// TF_DS_PROMPT_ROUTER's value: unset, "" or "off": none (today's launches); else a comma list of own, hide, or "all"
/// (own,hide). Any other word, or hide without own, is refused (the engine does not start).
pub fn parse(v: ?[]const u8) !Levers {
    const s = v orelse return .{};
    if (s.len == 0 or std.mem.eql(u8, s, "off")) return .{};
    var l: Levers = .{};
    var it = std.mem.splitScalar(u8, s, ',');
    while (it.next()) |w| {
        if (std.mem.eql(u8, w, "own")) l.own = true else if (std.mem.eql(u8, w, "hide")) l.hide = true else if (std.mem.eql(u8, w, "all")) {
            l.own = true;
            l.hide = true;
        } else return error.BadPromptRouter;
    }
    if (!l.own) return error.BadPromptRouter;
    return l;
}

/// The cuBLAS run whose arithmetic the own launches reproduce (prompt.xwTRows' runs on the served passes).
pub const run_rows: usize = 2048;

/// Whether an SP chunk of n rows with sub-runs of `sub_rows` (0: one run of n) routes on owned rows: every run is
/// run_rows rows and every owner's rows lie inside one run.
pub fn fits(o: Owners, sub_rows: usize) bool {
    if (!o.valid()) return false;
    const run = if (sub_rows == 0) o.n else sub_rows;
    if (run != run_rows or o.n % run != 0) return false;
    for (0..4) |h| {
        const a = o.r0(h);
        const r = o.rows(h);
        if (r == 0 or a / run != (a + r - 1) / run) return false;
    }
    return true;
}

/// Split-K partitions of K: n in [1, max_parts], b[0] = 0 < b[1] < ... < b[n] = K (router_own.cu's Parts).
pub const max_parts = 16;
pub const Parts = extern struct {
    n: c_int,
    b: [max_parts + 1]c_int,

    pub fn eql(a: Parts, b: Parts) bool {
        if (a.n != b.n) return false;
        const n: usize = @intCast(a.n);
        for (0..n + 1) |i| if (a.b[i] != b.b[i]) return false;
        return true;
    }
};

/// CUTLASS's split-K rule: partitions of round_up(ceil(K / s), align) k, the last shorter; null when that leaves an
/// empty partition or more than max_parts.
pub fn partsOf(k: usize, s: usize, al: usize) ?Parts {
    if (s == 0 or al == 0 or k == 0) return null;
    var ks = (k + s - 1) / s;
    ks = (ks + al - 1) / al * al;
    var p: Parts = .{ .n = 0, .b = @splat(0) };
    var at: usize = 0;
    while (at < k) {
        if (p.n == max_parts) return null;
        at = @min(at + ks, k);
        p.n += 1;
        p.b[@intCast(p.n)] = @intCast(at);
    }
    return p;
}

/// The candidates calibrate tries, in order: every split count 1..max_parts with every alignment CUTLASS uses for
/// its k partition, the duplicates (the same boundaries) dropped.
pub fn candidates(k: usize, out: *[max_parts * 8]Parts) []const Parts {
    const aligns = [_]usize{ 1, 4, 8, 16, 32, 64, 128, 256 };
    var n: usize = 0;
    // twelve first: what GB10's cuBLAS 13.6 picks for the 2,048-row run of 384 experts (stage ZR1's probe: twelve
    // partitions of 448, the last 192); every candidate is still tried against the library's own bits
    for ([_]usize{ 12, 2, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16 }) |s| for (aligns) |al| {
        const p = partsOf(k, s, al) orelse continue;
        for (out[0..n]) |q| {
            if (q.eql(p)) break;
        } else {
            out[n] = p;
            n += 1;
        }
    };
    return out[0..n];
}

/// router_own.cu's tilings (rows x experts a block). They differ only in which threads compute an element.
pub const Variant = enum(u8) {
    v64x64,
    v32x64,
    v128x64,
    v64x128,

    pub fn bm(v: Variant) usize {
        return switch (v) {
            .v64x64, .v64x128 => 64,
            .v32x64 => 32,
            .v128x64 => 128,
        };
    }
    pub fn bn(v: Variant) usize {
        return switch (v) {
            .v64x64, .v32x64, .v128x64 => 64,
            .v64x128 => 128,
        };
    }
    pub fn threads(v: Variant) usize {
        _ = v;
        return 256;
    }
    pub fn symbol(v: Variant, f: Form) [:0]const u8 {
        return switch (f) {
            .f32 => switch (v) {
                .v64x64 => "tf_ds_router_own_64x64",
                .v32x64 => "tf_ds_router_own_32x64",
                .v128x64 => "tf_ds_router_own_128x64",
                .v64x128 => "tf_ds_router_own_64x128",
            },
            .h => switch (v) {
                .v64x64 => "tf_ds_router_own_h_64x64",
                .v32x64 => "tf_ds_router_own_h_32x64",
                .v128x64 => "tf_ds_router_own_h_128x64",
                .v64x128 => "tf_ds_router_own_h_64x128",
            },
        };
    }
};

/// The operands' form: fp32 x and w (what cuBLAS reads), or x bf16 and w fp16 as the engine holds them (widened in
/// the program exactly as .float() does: the same FFMA operands, without the two cast launches).
pub const Form = enum(u8) { f32, h };

/// The tiling for `rows` own rows: 64x64 at both own-row counts the lever sees (stage ZR1, GB10, 384 experts, mean of
/// 20 layers: 1,024 rows 0.40 ms against 0.58 / 0.54 / 0.63 for 32x64 / 128x64 / 64x128; 512 rows 0.27 against 0.32 /
/// 0.51 / 0.59). Every tiling is checked at start, so the choice never changes bits.
pub fn variantFor(rows: usize, ne: usize) Variant {
    _ = rows;
    _ = ne;
    return .v64x64;
}

pub const Kernels = struct {
    module: cuda.Module,
    f: [2][4]cuda.Function,

    pub fn load(d: *const cuda.Driver, image: []const u8) !Kernels {
        var m = try cuda.Module.load(d, image);
        errdefer m.unload();
        var k: Kernels = .{ .module = m, .f = undefined };
        for (std.meta.tags(Form)) |f| for (std.meta.tags(Variant)) |v| {
            k.f[@intFromEnum(f)][@intFromEnum(v)] = try m.function(v.symbol(f));
        };
        return k;
    }

    pub fn unload(k: *Kernels) void {
        k.module.unload();
        k.* = undefined;
    }
};

/// y [rows, ne] fp32 (row stride ne) = the logits of x [rows, k] (row stride ldx) against w [ne, k] (fp32 both, or x
/// bf16 and w fp16: `f`) with today's 2,048-row run's arithmetic (`p`).
pub fn own(k: *const Kernels, s: cuda.Stream, f: Form, v: Variant, x: u64, ldx: usize, w: u64, y: u64, rows: usize, kd: usize, ne: usize, p: Parts) !void {
    if (rows == 0) return;
    const al: u64 = if (f == .f32) 16 else 8;
    if (ne % v.bn() != 0 or kd % 4 != 0 or ldx % 4 != 0 or x % al != 0 or w % al != 0) return error.RouterOwnShape;
    if (p.n < 1 or p.n > max_parts or p.b[@intCast(p.n)] != kd) return error.RouterOwnShape;
    var a: cuda.Args = .{};
    a.add(x);
    a.add(@as(c_longlong, @intCast(ldx)));
    a.add(w);
    a.add(y);
    a.add(@as(c_int, @intCast(rows)));
    a.add(@as(c_int, @intCast(kd)));
    a.add(@as(c_int, @intCast(ne)));
    a.add(p);
    const grid_y = (rows + v.bm() - 1) / v.bm();
    try cuda.launch.launch(k.f[@intFromEnum(f)][@intFromEnum(v)], .{ .grid = .{ .x = @intCast(ne / v.bn()), .y = @intCast(grid_y) }, .block = .{ .x = @intCast(v.threads()) } }, s, &a);
}

/// The engine's TF_DS_PROMPT_ROUTER state (the engine points here when own is on).
pub const State = struct {
    lv: Levers = .{},
    k: ?*const Kernels = null,
    parts: Parts = .{ .n = 0, .b = @splat(0) },
};

/// bf16 round to nearest even of an fp32 (finite values), as a value.
fn bf16Round(v: f32) f32 {
    const u: u32 = @bitCast(v);
    const r = u +% (0x7fff + ((u >> 16) & 1));
    return @bitCast(r & 0xffff0000);
}

/// The result of calibrate: the partitions, and how many candidates were tried before them.
pub const Calibration = struct { parts: Parts, tried: usize };

/// Finds today's 2,048-row run's split-K partitions on this GPU and library: x of run_rows bf16-valued rows (two
/// scales), the library's call on them (Blas.xwT of x.float() and wf = gate_w.float(), as prompt.xwTRows issues it)
/// against router_own.cu in the engine's form (x bf16, w16 = gate_w fp16) with each candidate over the whole run; the
/// first candidate equal in every byte is then checked on own-row windows of the run (the four quarters, the halves,
/// odd windows) with every tiling and both forms. Synchronizes the stream; uses its own scratch.
pub fn calibrate(gpa: std.mem.Allocator, d: *const cuda.Driver, k: *const Kernels, blas: *const cublas.Blas, s: cuda.Stream, w16: u64, wf: u64, kd: usize, ne: usize) !Calibration {
    const m = run_rows;
    var xb = try cuda.DeviceBuffer.alloc(d, m * kd * 4);
    defer xb.free();
    var xh = try cuda.DeviceBuffer.alloc(d, m * kd * 2);
    defer xh.free();
    var refb = try cuda.DeviceBuffer.alloc(d, m * ne * 4);
    defer refb.free();
    var ob = try cuda.DeviceBuffer.alloc(d, m * ne * 4);
    defer ob.free();
    const hx = try gpa.alloc(f32, m * kd);
    defer gpa.free(hx);
    const hh = try gpa.alloc(u16, m * kd);
    defer gpa.free(hh);
    const ref = try gpa.alloc(f32, m * ne);
    defer gpa.free(ref);
    const got = try gpa.alloc(f32, m * ne);
    defer gpa.free(got);
    var cand: [max_parts * 8]Parts = undefined;
    const cs = candidates(kd, &cand);
    var chosen: ?Parts = null;
    var tried: usize = 0;
    for ([_]f32{ 1.0, 8.0 }, 0..) |scale, round| {
        var prng = std.Random.DefaultPrng.init(0x5a31_0000 + round);
        const rnd = prng.random();
        for (hx, hh) |*v, *h| {
            v.* = bf16Round(@floatCast(rnd.floatNorm(f64) * scale));
            h.* = @intCast(@as(u32, @bitCast(v.*)) >> 16);
        }
        try xb.upload(0, std.mem.sliceAsBytes(hx));
        try xh.upload(0, std.mem.sliceAsBytes(hh));
        // (NULL-stream copies from pageable memory: done before the non-blocking stream reads them)
        try d.check(d.api.cuCtxSynchronize(), "cuCtxSynchronize");
        try blas.xwT(xb.ptr, wf, refb.ptr, m, kd, ne);
        try s.synchronize();
        try refb.download(0, std.mem.sliceAsBytes(ref));
        if (chosen == null) {
            for (cs) |p| {
                tried += 1;
                try own(k, s, .h, .v64x64, xh.ptr, kd, w16, ob.ptr, m, kd, ne, p);
                try s.synchronize();
                try ob.download(0, std.mem.sliceAsBytes(got));
                if (std.mem.eql(u8, std.mem.sliceAsBytes(got), std.mem.sliceAsBytes(ref))) {
                    chosen = p;
                    break;
                }
            }
        }
        const p = chosen orelse return error.RouterOwnNotExact;
        // own-row windows: quarters (a 2,048-row chunk's owners), halves (a 4,096-row pass's owners in a run), odd
        const wins = [_][2]usize{ .{ 0, 512 }, .{ 512, 512 }, .{ 1024, 512 }, .{ 1536, 512 }, .{ 0, 1024 }, .{ 1024, 1024 }, .{ 16, 368 }, .{ 1104, 944 }, .{ 7, 1 }, .{ 0, m } };
        for (wins) |win| for (std.meta.tags(Form)) |f| for (std.meta.tags(Variant)) |v| {
            const x = if (f == .f32) xb.ptr + win[0] * kd * 4 else xh.ptr + win[0] * kd * 2;
            try own(k, s, f, v, x, kd, if (f == .f32) wf else w16, ob.ptr, win[1], kd, ne, p);
            try s.synchronize();
            try ob.download(0, std.mem.sliceAsBytes(got[0 .. win[1] * ne]));
            if (!std.mem.eql(u8, std.mem.sliceAsBytes(got[0 .. win[1] * ne]), std.mem.sliceAsBytes(ref[win[0] * ne ..][0 .. win[1] * ne]))) return error.RouterOwnNotExact;
        };
    }
    return .{ .parts = chosen.?, .tried = tried };
}

test "TF_DS_PROMPT_ROUTER: unset / off is today's; own, own,hide, all; bad words and hide alone fail" {
    try std.testing.expectEqual(Levers{}, try parse(null));
    try std.testing.expectEqual(Levers{}, try parse(""));
    try std.testing.expectEqual(Levers{}, try parse("off"));
    try std.testing.expectEqual(Levers{ .own = true }, try parse("own"));
    try std.testing.expectEqual(Levers{ .own = true, .hide = true }, try parse("own,hide"));
    try std.testing.expectEqual(Levers{ .own = true, .hide = true }, try parse("hide,own"));
    try std.testing.expectEqual(Levers{ .own = true, .hide = true }, try parse("all"));
    try std.testing.expectError(error.BadPromptRouter, parse("hide"));
    try std.testing.expectError(error.BadPromptRouter, parse("1"));
    try std.testing.expectError(error.BadPromptRouter, parse("own,"));
    try std.testing.expectError(error.BadPromptRouter, parse("mine"));
    try std.testing.expectError(error.BadPromptRouter, parse("own,fxag"));
    var buf: [32]u8 = undefined;
    try std.testing.expectEqualStrings("own", (try parse("own")).names(&buf));
    try std.testing.expectEqualStrings("own,hide", (try parse("all")).names(&buf));
}

test "TF_DS_PROMPT_ROUTER: fits only where every owner's rows lie inside one 2,048-row cuBLAS run" {
    // the served chunk and pass
    try std.testing.expect(fits(Owners.of(2048), 0));
    try std.testing.expect(fits(Owners.of(2048), 2048));
    try std.testing.expect(fits(Owners.of(4096), 2048));
    // one run of other M, or a pass whose last run is shorter, or a run of 4,096
    try std.testing.expect(!fits(Owners.of(4096), 0));
    try std.testing.expect(!fits(Owners.of(1024), 0));
    try std.testing.expect(!fits(Owners.of(3072), 2048));
    try std.testing.expect(!fits(Owners.of(1500), 0));
    // every n: fits => every owner inside one run of exactly 2,048 rows, and the runs cover n in whole runs
    var n: usize = 64;
    while (n <= 4096) : (n += 1) for ([_]usize{ 0, 2048 }) |sub| {
        const o = Owners.of(n);
        if (!fits(o, sub)) continue;
        try std.testing.expect(n == 2048 or n == 4096);
        const run = if (sub == 0) n else sub;
        try std.testing.expectEqual(run_rows, run);
        var cover: usize = 0;
        for (0..4) |h| {
            try std.testing.expect(o.rows(h) > 0);
            try std.testing.expectEqual(o.r0(h) / run, (o.r0(h) + o.rows(h) - 1) / run);
            try std.testing.expect(o.r0(h) % 16 == 0); // the own launch's pointers stay 16-byte aligned
            cover += o.rows(h);
        }
        try std.testing.expectEqual(n, cover);
    };
}

test "TF_DS_PROMPT_ROUTER: split-K partitions by CUTLASS's rule, the candidates distinct and complete" {
    const p2 = partsOf(5120, 2, 4).?;
    try std.testing.expectEqual(@as(c_int, 2), p2.n);
    try std.testing.expectEqual(@as(c_int, 2560), p2.b[1]);
    try std.testing.expectEqual(@as(c_int, 5120), p2.b[2]);
    const p3 = partsOf(5120, 3, 4).?;
    try std.testing.expectEqual(@as(c_int, 3), p3.n);
    try std.testing.expectEqual(@as(c_int, 1708), p3.b[1]);
    try std.testing.expectEqual(@as(c_int, 3416), p3.b[2]);
    try std.testing.expectEqual(@as(c_int, 5120), p3.b[3]);
    try std.testing.expectEqual(@as(c_int, 1), partsOf(5120, 1, 256).?.n);
    try std.testing.expect(partsOf(5120, 17, 1) == null);
    // the run GB10's cuBLAS 13.6 picks for 2,048 rows of 384 experts: twelve partitions of 448 (the last 192)
    const p12 = partsOf(5120, 12, 32).?;
    try std.testing.expectEqual(@as(c_int, 12), p12.n);
    for (1..12) |j| try std.testing.expectEqual(@as(c_int, @intCast(448 * j)), p12.b[j]);
    try std.testing.expectEqual(@as(c_int, 5120), p12.b[12]);
    try std.testing.expect(partsOf(5120, 12, 64).?.eql(p12));
    var buf: [max_parts * 8]Parts = undefined;
    const cs = candidates(5120, &buf);
    try std.testing.expect(cs.len > 8);
    // twelve first; every one ends at K, strictly increasing, distinct
    try std.testing.expect(cs[0].eql(partsOf(5120, 12, 1).?));
    for (cs, 0..) |p, i| {
        try std.testing.expectEqual(@as(c_int, 5120), p.b[@intCast(p.n)]);
        const pn: usize = @intCast(p.n);
        for (1..pn + 1) |j| try std.testing.expect(p.b[j] > p.b[j - 1]);
        for (cs[0..i]) |q| try std.testing.expect(!q.eql(p));
    }
}

test "TF_DS_PROMPT_ROUTER: Parts is router_own.cu's struct (an int and seventeen ints, by value)" {
    try std.testing.expectEqual(@as(usize, 72), @sizeOf(Parts));
    try std.testing.expectEqual(@as(usize, 4), @alignOf(Parts));
    try std.testing.expectEqual(@as(f32, 1.0), bf16Round(1.0));
    try std.testing.expectEqual(@as(u32, 0x3f80_0000), @as(u32, @bitCast(bf16Round(@bitCast(@as(u32, 0x3f80_7fff))))));
    try std.testing.expectEqual(@as(u32, 0x3f81_0000), @as(u32, @bitCast(bf16Round(@bitCast(@as(u32, 0x3f80_8001))))));
}
