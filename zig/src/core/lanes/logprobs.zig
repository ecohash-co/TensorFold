//! Per-token log-probabilities (OpenAI's `logprobs` / `top_logprobs`): each reported token's log-softmax over the
//! whole vocabulary at the row that drew it, and the row's best `top` tokens. The distribution is the one the draw
//! reads before anything reshapes it: the raw fp32 logits at temperature 1 (OpenAI reports the same), never the
//! temperature-scaled, top_k / top_p / min_p filtered scores the sampler then takes its token from, and no penalties
//! (the served lane has none). A greedy row's token is its argmax, so its logprob is the row's largest.
//!
//! The arithmetic is a sharded log-sum-exp: each contiguous run of vocabulary columns (a rank's part of the head)
//! gives a `Partial` (its max, the sum of exp(logit - max): each exp in fp32 lanes, ~1 ulp, summed in f64; its best
//! `k` by value then id), and `merge`
//! takes the global max M and rescales each part's sum by exp(max_i - M), so the logsumexp lse = M + ln(sum) is the
//! full vocabulary's, and the top-k is the union's best k (torch.topk's set: of equal values the lower ids). A row is
//! cut into `row_shards` equal parts on the host, in that order, so the result is the same bit for bit wherever the
//! parts are reduced (a rank's own head columns could send their Partial to rank 0 instead of the whole row).
//! A NaN logit is never larger and adds nothing (the device argmax's rule); +inf is not expected from the head.
//!
//! Cost: nothing unless a stream asks (no kernel, no collective, no copy); a row that asks costs one pass over its
//! fp32 logits on rank 0's host (the backend's copy of the round's [R, vocab] logits, which sampled rows take anyway):
//! ~0.3 ms a 129,280-column row on one x86 core (ReleaseFast), so a six-row window adds ~2 ms to its round.
const std = @import("std");

/// The most top_logprobs a request may ask (OpenAI's limit).
pub const max_top = 20;

/// Equal column runs a row is reduced in (the four-node lane's vocabulary parts).
pub const row_shards = 4;

/// One run of a row's columns, reduced: its max, its sum of exp(logit - max) and its best `n` by value, then id.
pub const Partial = struct {
    max: f32 = -std.math.inf(f32),
    sum: f64 = 0,
    n: u8 = 0,
    ids: [max_top]u32 = undefined,
    vals: [max_top]f32 = undefined,

    /// Whether (va, ia) comes before (vb, ib): the larger value, then the lower id (sampling.zig's lexsort order).
    fn before(va: f32, ia: u32, vb: f32, ib: u32) bool {
        return va > vb or (va == vb and ia < ib);
    }

    /// Keep (v, id) among the best `k` (sorted, the worst last).
    fn offer(p: *Partial, k: usize, v: f32, id: u32) void {
        if (k == 0) return;
        if (p.n == k and !before(v, id, p.vals[k - 1], p.ids[k - 1])) return;
        var j: usize = if (p.n < k) p.n else k - 1;
        while (j > 0 and before(v, id, p.vals[j - 1], p.ids[j - 1])) : (j -= 1) {
            p.vals[j] = p.vals[j - 1];
            p.ids[j] = p.ids[j - 1];
        }
        p.vals[j] = v;
        p.ids[j] = id;
        if (p.n < k) p.n += 1;
    }
};

/// Columns a vector step of `partial` reads.
const lanes_v = 16;
const V32 = @Vector(lanes_v, f32);
const V64 = @Vector(lanes_v, f64);

/// exp(x) for x <= 0 (a logit less the row's max), in fp32 lanes: Cephes expf's reduction and polynomial, 2^n put in
/// the exponent bits (within 2 ulp; below -87.3 it is 0, which no sum of at least 1 notices). libm's expf is called
/// lane by lane and costs ~20x more over a 129,280-column row.
fn expNonPositive(x: V32) V32 {
    const I32 = @Vector(lanes_v, i32);
    const under = x < @as(V32, @splat(-87.3));
    const c = @max(x, @as(V32, @splat(-87.3)));
    const n = @round(c * @as(V32, @splat(1.44269504088896341)));
    const r = c - n * @as(V32, @splat(0.693359375)) - n * @as(V32, @splat(-2.12194440e-4));
    var q: V32 = @splat(1.9875691500e-4);
    q = q * r + @as(V32, @splat(1.3981999507e-3));
    q = q * r + @as(V32, @splat(8.3334519073e-3));
    q = q * r + @as(V32, @splat(4.1665795894e-2));
    q = q * r + @as(V32, @splat(1.6666665459e-1));
    q = q * r + @as(V32, @splat(5.0000001201e-1));
    const e = q * r * r + r + @as(V32, @splat(1.0));
    const ni: I32 = @intFromFloat(n);
    const two_n: V32 = @bitCast((ni + @as(I32, @splat(127))) << @as(@Vector(lanes_v, u5), @splat(23)));
    return @select(f32, under, @as(V32, @splat(0)), e * two_n);
}

/// The Partial of `cols`, the columns from vocabulary id `first`, keeping the best `k` (at most max_top). Each exp is
/// taken in fp32 on (logit - max) <= 0 (expNonPositive), summed in f64 lanes then reduced in a fixed order (the same
/// bits every run).
pub fn partial(cols: []const f32, first: u32, k: usize) Partial {
    std.debug.assert(k <= max_top);
    const ninf: V32 = @splat(-std.math.inf(f32));
    var p: Partial = .{};
    const body = cols.len - cols.len % lanes_v;
    var mx: V32 = ninf;
    var i: usize = 0;
    while (i < body) : (i += lanes_v) {
        const x: V32 = cols[i..][0..lanes_v].*;
        mx = @max(mx, @select(f32, x == x, x, ninf)); // a NaN is never larger
    }
    p.max = @reduce(.Max, mx);
    for (cols[body..]) |v| {
        if (v > p.max) p.max = v;
    }
    if (p.max == -std.math.inf(f32)) return p; // empty, or every column NaN or -inf: nothing to add
    const m: V32 = @splat(p.max);
    var acc: V64 = @splat(0);
    var thr: f32 = -std.math.inf(f32); // the kth best so far once k are held: a column below it is never kept
    i = 0;
    while (i < body) : (i += lanes_v) {
        const x: V32 = cols[i..][0..lanes_v].*;
        const live = x > ninf; // NaN and -inf add nothing
        acc += @as(V64, @floatCast(@select(f32, live, expNonPositive(x - m), @as(V32, @splat(0)))));
        if (k > 0 and @reduce(.Or, x >= @as(V32, @splat(thr)))) for (cols[i..][0..lanes_v], i..) |v, j| {
            if (v >= thr) {
                p.offer(k, v, first + @as(u32, @intCast(j)));
                if (p.n == k) thr = p.vals[k - 1];
            }
        };
    }
    var sum = @reduce(.Add, acc);
    for (cols[body..], body..) |v, j| {
        if (!(v > -std.math.inf(f32))) continue;
        sum += @as(f64, expNonPositive(@splat(v - p.max))[0]);
        p.offer(k, v, first + @as(u32, @intCast(j)));
    }
    p.sum = sum;
    return p;
}

/// The parts' union, in order: the global max, every sum rescaled to it (exp(max_i - M)), the best `k` of the union.
pub fn merge(parts: []const Partial, k: usize) Partial {
    var out: Partial = .{};
    for (parts) |p| {
        if (p.max > out.max) out.max = p.max;
    }
    if (out.max == -std.math.inf(f32)) return out;
    var sum: f64 = 0;
    for (parts) |p| {
        if (p.max == -std.math.inf(f32)) continue;
        sum += p.sum * @exp(@as(f64, p.max) - @as(f64, out.max));
        for (p.ids[0..p.n], p.vals[0..p.n]) |id, v| out.offer(k, v, id);
    }
    out.sum = sum;
    return out;
}

/// The vocabulary's logsumexp from a reduced row (-inf for a row with no finite logit).
pub fn lse(p: Partial) f64 {
    if (p.max == -std.math.inf(f32)) return -std.math.inf(f64);
    return @as(f64, p.max) + @log(p.sum);
}

/// A verified row's report: the logsumexp, the best `n` with their logprobs, the drawn token's, and the logprob of
/// the token the row's successor reads (the next window row's input, or the stream's next forced token): what a
/// forced or teacher-forced token commits in place of the draw.
pub const Row = struct {
    lse: f64,
    n: u8 = 0,
    ids: [max_top]u32 = undefined,
    lps: [max_top]f64 = undefined,
    sampled: u32,
    sampled_lp: f64,
    next: ?u32 = null,
    next_lp: f64 = 0,

    /// The report of `token` committed from this row: the drawn or the read token's logprob, else one of the best,
    /// else unscored (a token cut in by the thinking budget, not on any of those).
    pub fn forToken(r: *const Row, token: u32) Token {
        var t: Token = .{ .token = token, .logprob = std.math.nan(f64), .n = r.n };
        @memcpy(t.ids[0..r.n], r.ids[0..r.n]);
        @memcpy(t.lps[0..r.n], r.lps[0..r.n]);
        if (token == r.sampled) {
            t.logprob = r.sampled_lp;
        } else if (r.next != null and token == r.next.?) {
            t.logprob = r.next_lp;
        } else for (r.ids[0..r.n], r.lps[0..r.n]) |id, lp| {
            if (id == token) t.logprob = lp;
        }
        return t;
    }
};

/// A committed token's report, as a stream keeps it beside its emitted tokens.
pub const Token = struct {
    token: u32,
    /// NaN: unscored (no row reported this token: a backend without logprobs, or a token forced past its row).
    logprob: f64,
    n: u8 = 0,
    ids: [max_top]u32 = undefined,
    lps: [max_top]f64 = undefined,

    pub fn unscored(token: u32) Token {
        return .{ .token = token, .logprob = std.math.nan(f64) };
    }

    pub fn scored(t: *const Token) bool {
        return !std.math.isNan(t.logprob);
    }

    pub fn top(t: *const Token) []const u32 {
        return t.ids[0..t.n];
    }
};

/// The logprob of column `id` of `logits` against the row's logsumexp (fp32 logit, f64 difference).
fn logprobAt(logits: []const f32, id: u32, l: f64) f64 {
    if (id >= logits.len) return std.math.nan(f64);
    return @as(f64, logits[id]) - l;
}

/// Reduce one row of fp32 logits (the whole vocabulary, in id order) over `shards` equal column runs: its Row with
/// the best `top` (at most max_top), the drawn token `sampled`'s logprob and `next`'s (null: none).
pub fn rowOver(logits: []const f32, shards: usize, top: usize, sampled: u32, next: ?u32) Row {
    std.debug.assert(shards >= 1 and shards <= 64);
    const k = @min(top, max_top);
    var parts: [64]Partial = undefined;
    const width = (logits.len + shards - 1) / shards;
    var n: usize = 0;
    var at: usize = 0;
    while (at < logits.len) : (at += width) {
        const end = @min(logits.len, at + width);
        parts[n] = partial(logits[at..end], @intCast(at), k);
        n += 1;
    }
    const all = merge(parts[0..n], k);
    const l = lse(all);
    var r: Row = .{ .lse = l, .n = all.n, .sampled = sampled, .sampled_lp = logprobAt(logits, sampled, l) };
    for (0..all.n) |i| {
        r.ids[i] = all.ids[i];
        r.lps[i] = @as(f64, all.vals[i]) - l;
    }
    if (next) |t| {
        r.next = t;
        r.next_lp = logprobAt(logits, t, l);
    }
    return r;
}

/// rowOver with the lane's `row_shards`.
pub fn row(logits: []const f32, top: usize, sampled: u32, next: ?u32) Row {
    return rowOver(logits, row_shards, top, sampled, next);
}

/// A reference for the tests: one pass over the whole row in f64, the top-k by a full sort.
fn reference(a: std.mem.Allocator, logits: []const f32, k: usize) !struct { lse: f64, ids: []u32 } {
    var m: f64 = -std.math.inf(f64);
    for (logits) |v| m = @max(m, @as(f64, v));
    var s: f64 = 0;
    for (logits) |v| s += @exp(@as(f64, v) - m);
    const order = try a.alloc(u32, logits.len);
    for (order, 0..) |*o, i| o.* = @intCast(i);
    const Ctx = struct {
        l: []const f32,
        fn less(c: @This(), x: u32, y: u32) bool {
            return c.l[x] > c.l[y] or (c.l[x] == c.l[y] and x < y);
        }
    };
    std.mem.sort(u32, order, Ctx{ .l = logits }, Ctx.less);
    return .{ .lse = m + @log(s), .ids = order[0..k] };
}

test "a sharded row's logsumexp and top-k are the single pass's, whatever the shards" {
    const a = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x10b9);
    const rnd = prng.random();
    // DeepSeek-V4.1's vocabulary width, logits spread like a head's (a few large, most near zero), with ties
    const V = 129280;
    const logits = try a.alloc(f32, V);
    defer a.free(logits);
    for (logits) |*v| v.* = @floatCast(rnd.floatNorm(f64) * 3.0);
    logits[77] = 31.5;
    logits[129279] = 31.5; // a tie with 77 at the top: the lower id first
    logits[64640] = 29.0;
    logits[5] = 29.0;
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    const want = try reference(arena.allocator(), logits, max_top);
    for ([_]usize{ 1, 2, 3, 4, 7, 64 }) |shards| {
        const r = rowOver(logits, shards, max_top, 77, 5);
        try std.testing.expectApproxEqAbs(want.lse, r.lse, 1e-7); // fp32 exps, f64 sums
        try std.testing.expectEqual(@as(u8, max_top), r.n);
        try std.testing.expectEqualSlices(u32, want.ids, r.ids[0..r.n]);
        for (r.ids[0..r.n], r.lps[0..r.n]) |id, lp| try std.testing.expectEqual(@as(f64, logits[id]) - r.lse, lp);
        try std.testing.expectEqual(r.lps[0], r.sampled_lp); // the argmax drew: its logprob is the row's best
        try std.testing.expectEqual(@as(f64, 29.0) - r.lse, r.next_lp);
    }
    // the probabilities over the whole vocabulary sum to 1
    const r = row(logits, 0, 77, null);
    var total: f64 = 0;
    for (logits) |v| total += @exp(@as(f64, v) - r.lse);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), total, 1e-7);
    try std.testing.expectEqual(@as(u8, 0), r.n);
}

test "merging partials: the global max rescales each part, empty and NaN parts add nothing" {
    const cols = [_]f32{ 1.0, -2.0, std.math.nan(f32), 4.0, 4.0, -std.math.inf(f32), 0.5, 3.0 };
    const whole = partial(&cols, 0, 3);
    var parts: [3]Partial = undefined;
    parts[0] = partial(cols[0..3], 0, 3);
    parts[1] = partial(cols[3..3], 3, 3); // an empty run
    parts[2] = partial(cols[3..], 3, 3);
    const m = merge(&parts, 3);
    try std.testing.expectEqual(whole.max, m.max);
    try std.testing.expectApproxEqAbs(lse(whole), lse(m), 1e-9);
    try std.testing.expectEqualSlices(u32, &.{ 3, 4, 7 }, m.ids[0..m.n]);
    const want: f64 = @log(@exp(1.0) + @exp(-2.0) + 2 * @exp(4.0) + @exp(0.5) + @exp(3.0));
    try std.testing.expectApproxEqAbs(want, lse(m), 1e-7); // the exps are fp32's
    // a row of NaN alone: nothing to report
    const none = [_]f32{ std.math.nan(f32), std.math.nan(f32) };
    try std.testing.expectEqual(-std.math.inf(f64), lse(merge(&.{partial(&none, 0, 2)}, 2)));
}

test "a committed token's report: drawn, read next, among the best, or unscored" {
    const logits = [_]f32{ 0.0, 5.0, 1.0, 2.0, -1.0 };
    const r = rowOver(&logits, 2, 2, 1, 4);
    try std.testing.expectEqualSlices(u32, &.{ 1, 3 }, r.ids[0..r.n]);
    const drawn = r.forToken(1);
    try std.testing.expect(drawn.scored());
    try std.testing.expectEqual(r.sampled_lp, drawn.logprob);
    try std.testing.expectEqualSlices(u32, &.{ 1, 3 }, drawn.top());
    try std.testing.expectEqual(@as(f64, -1.0) - r.lse, r.forToken(4).logprob); // the forced successor
    try std.testing.expectEqual(r.lps[1], r.forToken(3).logprob); // one of the best
    try std.testing.expect(!r.forToken(2).scored()); // none of those: unscored
    try std.testing.expect(!Token.unscored(7).scored());
}

test "the fp32 lanes' exp within 2 ulp of exp on [-87.3, 0], 0 below" {
    var x: f64 = 0;
    var worst: f64 = 0;
    while (x > -87.3) : (x -= 0.0137) {
        var v: V32 = @splat(@floatCast(x));
        v[1] = @floatCast(x / 3);
        const got = expNonPositive(v);
        for ([_]f64{ @as(f32, @floatCast(x)), @as(f32, @floatCast(x / 3)) }, [_]f32{ got[0], got[1] }) |xi, g| {
            const want = @exp(xi);
            worst = @max(worst, @abs(@as(f64, g) - want) / want);
        }
    }
    try std.testing.expect(worst < 2.5e-7);
    try std.testing.expectEqual(@as(f32, 1.0), expNonPositive(@splat(0.0))[0]);
    try std.testing.expectEqual(@as(f32, 0.0), expNonPositive(@splat(-90.0))[0]);
}
