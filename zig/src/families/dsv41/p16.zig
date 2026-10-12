//! TF_DS_PROMPT_P16 (tier 2: a deliberate numerics change, quality-gated; ZIG-PLATFORM.md "Adoption rules"): on an SP
//! prompt chunk the fp32 partials Two.toOwners exchanges with the rows' owners (wo_b's ch.pa and the MoE's ch.pm,
//! ZQ1's `sr g^1+2+3` class: 3 x 2,560 x 4 B = 30 KiB a chunk row a layer each node sends, both sublayers) go over the
//! wire as bf16: the sender rounds the rows it sends (round to nearest even, p16.cu) into a staging copy, NCCL moves
//! half the bytes, and the owner widens what it received back to fp32 (exact) in place of toOwnersCopy's byte copy,
//! so the hc mix and its captured Triton variants read the fp32 [2, rows, D] gather they read today (no new variant).
//! The owner's own part never leaves the node and stays fp32.
//!
//! Which chunks: every chunk that takes SP rows (prompt.spFor: TF_DS_PROMPT_SP=rows' encoder-only chunks, rows2's
//! chunks of >= 256 rows that start before the replay, and on rows2's chunk that reaches the replay only its layers
//! before the cut; ch.sp is null after it) and every path of their toOwners: the plain group, A2A=shift, OVL=wob
//! (pa's sends on the comm stream), OVL=dnc (pm row-major on the comm stream) and OVL=dn (pm half-major, a half at a
//! time), and the overlapped passes' second chunk (Overlap's Two). Untouched: chunks without SP rows (under 256 rows,
//! decoder chunks, image spans, the head past the cut: quarters as today), catRank / pack / Engram / x gathers, decode
//! rounds and drafts.
//!
//! Unset, "" or "0": off, the served path (no staging buffer, no kernel module, every launch and exchange as today).
//! "1": on (needs the four-node split and TF_DS_PROMPT_SP=rows or rows2). Anything else refuses to start.
const std = @import("std");
const cuda = @import("cuda");
const prompt2d = @import("prompt2d.zig");

pub const start_line = "TF_DS_PROMPT_P16=1: owner partials exchanged as bf16 (tier 2, numerics change)";

/// TF_DS_PROMPT_P16's value: null (unset), "" or "0": off; "1": on; else error.BadPromptP16.
pub fn parse(v: ?[]const u8) !bool {
    const s = v orelse return false;
    if (s.len == 0 or std.mem.eql(u8, s, "0")) return false;
    if (std.mem.eql(u8, s, "1")) return true;
    return error.BadPromptP16;
}

pub fn fromEnv() !bool {
    const v = std.c.getenv("TF_DS_PROMPT_P16") orelse return parse(null);
    return parse(std.mem.span(v));
}

/// The start check: P16 only acts on SP chunks' toOwners, so it needs the four-node split and SP rows.
pub fn check(on: bool, four: bool, sp_rows: bool) !void {
    if (!on) return;
    if (!four) return error.P16NeedsTheFourNodeSplit;
    if (!sp_rows) return error.P16NeedsSpRows;
}

/// p16.cu's tf_p16_rne on the host: fp32 bits -> bf16 bits, round to nearest even (__float2bfloat16_rn on every
/// non-NaN value); a NaN stays a quiet NaN of its sign.
pub fn rne(u: u32) u16 {
    if ((u & 0x7fffffff) > 0x7f800000) return @intCast((u >> 16) | 0x0040);
    return @intCast((u +% (0x7fff + ((u >> 16) & 1))) >> 16);
}

/// p16.cu's tf_p16_widen: bf16 bits -> fp32 bits (exact).
pub fn widen(h: u16) u32 {
    return @as(u32, h) << 16;
}

/// The bytes node `me` sends in one toOwners of a part of widths w (element size esize): every other owner's rows.
pub fn sendBytes(me: usize, o: prompt2d.Owners, w: [2]usize, esize: usize) usize {
    var n: usize = 0;
    for (0..4) |h| if (h != me) {
        n += prompt2d.ownerExchange(me, h, o, w, esize).send_bytes;
    };
    return n;
}

/// The bytes node `me` receives in one toOwners.
pub fn recvBytes(me: usize, o: prompt2d.Owners, w: [2]usize, esize: usize) usize {
    var n: usize = 0;
    for (0..4) |h| if (h != me) {
        n += prompt2d.ownerExchange(me, h, o, w, esize).recv_bytes;
    };
    return n;
}

/// The row ranges node `me` sends of a chunk's part (everything but its own rows): [first row, rows], up to two.
pub fn sentRuns(me: usize, o: prompt2d.Owners) [2][2]usize {
    const a = o.r0(me);
    const b = a + o.rows(me);
    return .{ .{ 0, a }, .{ b, o.n - b } };
}

/// One pack launch of Two.p16Pack: `count` values from byte src_off of the fp32 part to byte dst_off of the bf16 stage.
pub const Pack = struct { src_off: usize, dst_off: usize, count: usize };

/// Two.p16Pack's launches on node `me` for a part of `wp` columns: the rows it sends (sentRuns), at the same row index
/// in the stage (so ownerExchange's offsets at esize 2 address them). An empty run is a launch of no values.
pub fn packPlan(me: usize, o: prompt2d.Owners, wp: usize) [2]Pack {
    var out: [2]Pack = undefined;
    for (sentRuns(me, o), &out) |run, *x| x.* = .{ .src_off = run[0] * wp * 4, .dst_off = run[0] * wp * 2, .count = run[1] * wp };
    return out;
}

pub const Kernels = struct {
    module: cuda.Module,
    pack_f: cuda.Function,
    widen_f: cuda.Function,

    pub fn load(d: *const cuda.Driver, image: []const u8) !Kernels {
        var m = try cuda.Module.load(d, image);
        errdefer m.unload();
        return .{ .module = m, .pack_f = try m.function("tf_ds_p16_pack_kernel"), .widen_f = try m.function("tf_ds_p16_widen_rows_kernel") };
    }

    pub fn unload(k: *Kernels) void {
        k.module.unload();
        k.* = undefined;
    }

    /// dst bf16 [count] = rne(src fp32 [count]) on stream s.
    pub fn pack(k: *const Kernels, s: cuda.Stream, src: u64, dst: u64, count: usize) !void {
        if (count == 0) return;
        if (count % 4 != 0 or src % 16 != 0 or dst % 8 != 0) return error.P16Shape;
        const n4 = count / 4;
        var a: cuda.Args = .{};
        a.add(src);
        a.add(dst);
        a.add(@as(u64, n4));
        const grid: u32 = @intCast(@min((n4 + 255) / 256, 65535));
        try cuda.launch.launch(k.pack_f, .{ .grid = .{ .x = grid }, .block = .{ .x = 256 } }, s, &a);
    }

    /// `rows` rows of `cols` values: dst fp32 (rows dst_ld bytes apart) = widen(src bf16, rows src_ld bytes apart).
    pub fn widenRows(k: *const Kernels, s: cuda.Stream, src: u64, src_ld: usize, dst: u64, dst_ld: usize, cols: usize, rows: usize) !void {
        if (rows == 0 or cols == 0) return;
        if (cols % 4 != 0 or src % 8 != 0 or dst % 16 != 0 or src_ld % 8 != 0 or dst_ld % 16 != 0 or rows > 65535) return error.P16Shape;
        if (src_ld < cols * 2 or dst_ld < cols * 4) return error.Overlap;
        var a: cuda.Args = .{};
        a.add(src);
        a.add(@as(u64, src_ld / 8));
        a.add(dst);
        a.add(@as(u64, dst_ld / 16));
        a.add(@as(u64, cols / 4));
        try cuda.launch.launch(k.widen_f, .{ .grid = .{ .x = @intCast((cols / 4 + 255) / 256), .y = @intCast(rows) }, .block = .{ .x = 256 } }, s, &a);
    }
};

// ---- host tests ------------------------------------------------------------------------------------------------------

/// An independent reference: the bf16 nearest to x (ties to the even mantissa), from the two candidates' distances in
/// f64; overflow past the largest finite bf16 (by at least half its last step) gives infinity.
fn refRne(u: u32) u16 {
    const x: f32 = @bitCast(u);
    const lo: u16 = @intCast(u >> 16); // truncation: toward zero
    if ((u & 0xffff) == 0) return lo;
    const hi: u16 = lo + 1; // away from zero (0x7f7f + 1 = 0x7f80: infinity)
    const xl: f64 = @as(f32, @bitCast(widen(lo)));
    // infinity (hi past the largest finite bf16) is measured as the next step would be: 2^128 (the half-way point to
    // it is 2^128 - 2^119, where fp32 values round to infinity)
    const xh_raw: f32 = @bitCast(widen(hi));
    const xh: f64 = if (std.math.isInf(xh_raw)) std.math.copysign(std.math.ldexp(@as(f64, 1), 128), @as(f64, x)) else xh_raw;
    const d_lo = @abs(@as(f64, x) - xl);
    const d_hi = @abs(xh - @as(f64, x));
    if (d_lo < d_hi) return lo;
    if (d_hi < d_lo) return hi;
    return if (lo & 1 == 0) lo else hi;
}

test "TF_DS_PROMPT_P16: rne is round to nearest even (an independent reference: 2^16 tops x 24 low halves)" {
    const lows = [_]u16{ 0x0000, 0x0001, 0x0002, 0x3fff, 0x4000, 0x7ffe, 0x7fff, 0x8000, 0x8001, 0x8002, 0xbfff, 0xc000, 0xfffe, 0xffff, 0x1234, 0x5555, 0x7f00, 0x80ff, 0xaaaa, 0xdead, 0x0100, 0x00ff, 0xf000, 0x0fff };
    var checked: usize = 0;
    for (0..0x10000) |top| for (lows) |lo| {
        const u: u32 = (@as(u32, @intCast(top)) << 16) | lo;
        if ((u & 0x7fffffff) > 0x7f800000) continue; // NaN: its own test
        if ((u & 0x7fffffff) == 0x7f800000) continue; // infinity: its own test
        try std.testing.expectEqual(refRne(u), rne(u));
        checked += 1;
    };
    try std.testing.expect(checked > 1_500_000);
}

test "TF_DS_PROMPT_P16: rne ties go to even, both signs" {
    // 1 + 2^-8 exactly half way between bf16 1.0 (0x3f80) and 1.0078125 (0x3f81): to 0x3f80 (even)
    try std.testing.expectEqual(@as(u16, 0x3f80), rne(0x3f808000));
    // half way between 0x3f81 and 0x3f82: to 0x3f82 (even)
    try std.testing.expectEqual(@as(u16, 0x3f82), rne(0x3f818000));
    try std.testing.expectEqual(@as(u16, 0xbf80), rne(0xbf808000));
    try std.testing.expectEqual(@as(u16, 0xbf82), rne(0xbf818000));
    // just past half way rounds up, just below rounds down
    try std.testing.expectEqual(@as(u16, 0x3f81), rne(0x3f808001));
    try std.testing.expectEqual(@as(u16, 0x3f80), rne(0x3f807fff));
    // a carry into the exponent: the largest mantissa of a binade rounds to the next power of two
    try std.testing.expectEqual(@as(u16, 0x4000), rne(0x3fffffff));
}

test "TF_DS_PROMPT_P16: signed zero, infinities, overflow, NaN" {
    try std.testing.expectEqual(@as(u16, 0x0000), rne(0x00000000));
    try std.testing.expectEqual(@as(u16, 0x8000), rne(0x80000000));
    try std.testing.expectEqual(@as(u16, 0x7f80), rne(0x7f800000));
    try std.testing.expectEqual(@as(u16, 0xff80), rne(0xff800000));
    // fp32's largest finite value is past bf16's half step above its largest finite value: infinity, as the intrinsic
    try std.testing.expectEqual(@as(u16, 0x7f80), rne(0x7f7fffff));
    try std.testing.expectEqual(@as(u16, 0xff80), rne(0xff7fffff));
    // bf16's largest finite value plus just under half a step stays finite
    try std.testing.expectEqual(@as(u16, 0x7f7f), rne(0x7f7f7fff));
    // every NaN stays a NaN of its sign (the payload's top bits kept, quiet bit set): never an infinity
    for ([_]u32{ 0x7f800001, 0x7f80ffff, 0x7fc00000, 0x7fffffff, 0xff800001, 0xffc00000, 0xffffffff, 0x7f810000 }) |u| {
        const h = rne(u);
        try std.testing.expect((h & 0x7fff) > 0x7f80);
        try std.testing.expectEqual(u >> 31, @as(u32, h >> 15));
        const w: f32 = @bitCast(widen(h));
        try std.testing.expect(std.math.isNan(w));
    }
}

test "TF_DS_PROMPT_P16: subnormals round like normals (no flush to zero)" {
    // the smallest fp32 subnormal: under half bf16's smallest subnormal (2^-133) -> +0 / -0
    try std.testing.expectEqual(@as(u16, 0x0000), rne(0x00000001));
    try std.testing.expectEqual(@as(u16, 0x8000), rne(0x80000001));
    // bf16's smallest subnormal and half way above it (tie to even: back down), just past half (up)
    try std.testing.expectEqual(@as(u16, 0x0001), rne(0x00010000));
    try std.testing.expectEqual(@as(u16, 0x0000), rne(0x00008000));
    try std.testing.expectEqual(@as(u16, 0x0001), rne(0x00008001));
    try std.testing.expectEqual(@as(u16, 0x0002), rne(0x00018000));
    // the largest subnormal rounds up into the smallest normal
    try std.testing.expectEqual(@as(u16, 0x0080), rne(0x007fffff));
    try std.testing.expectEqual(@as(u16, 0x807f), rne(0x807f0000));
}

test "TF_DS_PROMPT_P16: widen is exact and rne(widen(h)) == h for every non-NaN bf16" {
    for (0..0x10000) |i| {
        const h: u16 = @intCast(i);
        if ((h & 0x7fff) > 0x7f80) continue;
        try std.testing.expectEqual(h, rne(widen(h)));
        // the widened value is the bf16 value: its low half is zero
        try std.testing.expectEqual(@as(u32, 0), widen(h) & 0xffff);
    }
    // the round trip's error is at most half a bf16 step: 2^-8 of the binade's power of two, so at most |x| / 256
    var prng = std.Random.DefaultPrng.init(0x9161);
    const r = prng.random();
    for (0..100_000) |_| {
        const u = r.int(u32) & 0xbf7fffff | 0x00800000; // finite, normal
        const x: f32 = @bitCast(u);
        const y: f32 = @bitCast(widen(rne(u)));
        try std.testing.expect(@abs(@as(f64, y) - x) <= @abs(@as(f64, x)) * (1.0 / 256.0));
    }
}

test "TF_DS_PROMPT_P16: parse, refusals and the start check" {
    try std.testing.expectEqual(false, try parse(null));
    try std.testing.expectEqual(false, try parse(""));
    try std.testing.expectEqual(false, try parse("0"));
    try std.testing.expectEqual(true, try parse("1"));
    for ([_][]const u8{ "2", "on", "true", "bf16", " 1", "1 ", "01", "yes", "-1" }) |v| try std.testing.expectError(error.BadPromptP16, parse(v));
    try check(false, false, false);
    try check(true, true, true);
    try std.testing.expectError(error.P16NeedsSpRows, check(true, true, false));
    try std.testing.expectError(error.P16NeedsTheFourNodeSplit, check(true, false, true));
    try std.testing.expectEqualStrings("TF_DS_PROMPT_P16=1: owner partials exchanged as bf16 (tier 2, numerics change)", start_line);
}

test "TF_DS_PROMPT_P16: the owner exchange's bytes halve (rows and rows2, every node, both halves of dn)" {
    const w = [2]usize{ 2560, 2560 }; // wo_b's / down's columns a pair (hidden 5,120)
    for ([_]usize{ 256, 320, 1024, 2048, 2112, 4096, 2047, 1999 }) |n| {
        const o = prompt2d.Owners.of(n);
        if (!o.valid()) continue;
        var sent32: usize = 0;
        var sent16: usize = 0;
        for (0..4) |me| {
            const s32 = sendBytes(me, o, w, 4);
            const s16 = sendBytes(me, o, w, 2);
            try std.testing.expectEqual(s32, 2 * s16);
            try std.testing.expectEqual(recvBytes(me, o, w, 4), 2 * recvBytes(me, o, w, 2));
            // the rows sent are every row but the node's own
            try std.testing.expectEqual((n - o.rows(me)) * w[me / 2] * 4, s32);
            // dn's halves: each half's exchange is toOwners' at half the width
            const wh = try prompt2d.halves(w);
            try std.testing.expectEqual(s16, 2 * sendBytes(me, o, wh, 2));
            // the pack covers exactly the sent rows (sentRuns)
            var rows: usize = 0;
            for (sentRuns(me, o)) |run| rows += run[1];
            try std.testing.expectEqual(n - o.rows(me), rows);
            sent32 += s32;
            sent16 += s16;
        }
        try std.testing.expectEqual(sent32, 2 * sent16);
        if (n % 64 == 0) {
            // four quarters (rows): each node sends 3 n / 4 rows of 2,560 values, 7,680 B a chunk row a sublayer fp32
            // (15,360 B a layer: ZQ1's 15.4 KB), 3,840 B as bf16 (7,680 B a layer)
            try std.testing.expectEqual(n * 7680, sent32 / 4);
            try std.testing.expectEqual(n * 3840, sent16 / 4);
        }
    }
}

/// The host model of one P16 toOwners for owner `me` (Two.toOwnersSend + toOwnersCopy with p16 on, or the dn halves'
/// pair): every node's fp32 part packed by packPlan (rne), the bf16 bytes moved as ownerExchange(.., 2) says, and the
/// owner's dst filled by p16Copy's rule (own rows copied from its fp32 part, the others widened). parts[g] holds node
/// g's part row-major [n, w[g / 2]] or, with dn, half-major [2, n, w[g / 2] / 2].
fn simulate(comptime n: usize, comptime dw: usize, o: prompt2d.Owners, w: [2]usize, me: usize, dn: bool, parts: *const [4][n * dw]u32, dst: []u32) !void {
    const rm = o.rows(me);
    const wh = [2]usize{ w[0] / 2, w[1] / 2 };
    const nh: usize = if (dn) 2 else 1;
    const ww: [2]usize = if (dn) wh else w;
    // each sender's stage (bf16), its pack launches per half
    var stage: [4][n * dw]u16 = @splat(@splat(0x5a5a));
    for (0..4) |g| for (0..nh) |h| {
        const src_half = h * n * ww[g / 2]; // values
        for (packPlan(g, o, ww[g / 2])) |pk| for (0..pk.count) |i| {
            stage[g][src_half + pk.dst_off / 2 + i] = rne(parts[g][src_half + pk.src_off / 4 + i]);
        };
    };
    // the owner's receive area (bf16, half h's area at dnPartsOff(.., 2, h))
    var recv: [2 * 4 * n * dw]u16 = @splat(0xa5a5);
    for (0..nh) |h| for (0..4) |g| {
        if (g == me) continue;
        const sx = prompt2d.ownerExchange(g, me, o, ww, 2);
        const rx = prompt2d.ownerExchange(me, g, o, ww, 2);
        try std.testing.expectEqual(sx.send_bytes, rx.recv_bytes);
        const base = if (dn) prompt2d.dnPartsOff(o, w, 2, h) / 2 else 0;
        const src = h * n * ww[g / 2] + sx.send_off / 2;
        @memcpy(recv[base + rx.recv_off / 2 ..][0 .. rx.recv_bytes / 2], stage[g][src..][0 .. sx.send_bytes / 2]);
    };
    // p16Copy: c4 places, c2 sources
    for (0..nh) |h| {
        const c4 = if (dn) prompt2d.ownerCopiesHalf(me, o, w, 4, h) else prompt2d.ownerCopies(me, o, w, 4);
        const c2 = if (dn) prompt2d.ownerCopiesHalf(me, o, w, 2, h) else prompt2d.ownerCopies(me, o, w, 2);
        const own = h * n * ww[me / 2];
        const rbase = if (dn) prompt2d.dnPartsOff(o, w, 2, h) / 2 else 0;
        for (c4, c2) |x, y| for (0..rm) |i| for (0..x.bytes / 4) |j| {
            dst[x.dst_off / 4 + i * x.dst_ld / 4 + j] = if (x.from == me)
                parts[me][own + x.src_off / 4 + i * x.src_ld / 4 + j]
            else
                widen(recv[rbase + y.src_off / 2 + i * y.src_ld / 2 + j]);
        };
    }
}

test "TF_DS_PROMPT_P16: the owner's [2, rows, D] is today's toOwners result with every other node's values rounded (row-major and dn halves)" {
    var prng = std.Random.DefaultPrng.init(0x16);
    const r = prng.random();
    inline for (.{ prompt2d.Owners{ .q = 4, .n = 16 }, prompt2d.Owners{ .q = 4, .n = 19 }, prompt2d.Owners{ .q = 3, .n = 12 } }) |o| {
        inline for (.{ [2]usize{ 8, 8 }, [2]usize{ 12, 8 } }) |w| {
            const n: usize = o.n;
            const dw: usize = w[0] + w[1];
            // partial[r][row][col], values with low bits set (so rounding shows) and a few specials
            var partial: [2][n][dw]u32 = undefined;
            for (0..2) |rr| for (0..n) |i| for (0..dw) |j| {
                partial[rr][i][j] = r.int(u32) & 0xbfffffff;
            };
            partial[0][0][0] = 0x80000000; // -0
            partial[1][n - 1][dw - 1] = 0x00008001; // a subnormal that rounds up
            partial[0][n - 1][1] = 0x3f808000; // a tie to even
            inline for (.{ false, true }) |dn| {
                var parts: [4][n * dw]u32 = @splat(@splat(0));
                const wh = [2]usize{ w[0] / 2, w[1] / 2 };
                for (0..4) |g| for (0..n) |i| for (0..w[g / 2]) |j| {
                    const v = partial[g % 2][i][(g / 2) * w[0] + j];
                    if (dn) {
                        const h = j / wh[g / 2];
                        parts[g][h * n * wh[g / 2] + i * wh[g / 2] + j % wh[g / 2]] = v;
                    } else parts[g][i * w[g / 2] + j] = v;
                };
                for (0..4) |me| {
                    const rm = o.rows(me);
                    var dst: [2 * n * dw]u32 = @splat(0xffffffff);
                    try simulate(n, dw, o, w, me, dn, &parts, dst[0 .. 2 * rm * dw]);
                    for (0..2) |rr| for (0..rm) |i| for (0..dw) |j| {
                        const from = rr + 2 * (if (j < w[0]) @as(usize, 0) else 1); // the node whose part holds it
                        const v = partial[rr][o.r0(me) + i][j];
                        const want = if (from == me) v else widen(rne(v));
                        try std.testing.expectEqual(want, dst[(rr * rm + i) * dw + j]);
                    };
                }
            }
        }
    }
}

test "TF_DS_PROMPT_P16: received parts fit the slots toOwners already reserves, and the bf16 copies read inside them" {
    const w = [2]usize{ 2560, 2560 };
    for ([_]usize{ 256, 2048, 4096, 2047 }) |n| {
        const o = prompt2d.Owners.of(n);
        for (0..4) |me| {
            const c2 = prompt2d.ownerCopies(me, o, w, 2);
            const c4 = prompt2d.ownerCopies(me, o, w, 4);
            for (0..4) |h| {
                if (h == me) continue;
                const x = prompt2d.ownerExchange(me, h, o, w, 2);
                // the widen reads exactly the bf16 bytes received from h, writes exactly the fp32 bytes the byte copy did
                try std.testing.expectEqual(x.recv_off, c2[h].src_off);
                try std.testing.expectEqual(c2[h].src_ld * o.rows(me), x.recv_bytes);
                try std.testing.expectEqual(c4[h].bytes, 2 * c2[h].bytes);
                try std.testing.expect(x.recv_off + x.recv_bytes <= prompt2d.ownerExchange(me, h, o, w, 4).recv_off + prompt2d.ownerExchange(me, h, o, w, 4).recv_bytes);
            }
            // dn's half areas at bf16 end inside the fp32 ones
            try std.testing.expect(prompt2d.dnPartsOff(o, w, 2, 2) * 2 == prompt2d.dnPartsOff(o, w, 4, 2));
        }
    }
}
