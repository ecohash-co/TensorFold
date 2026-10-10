//! TF_DS_PROMPT_PROF=1: a prompt chunk's GPU time by phase on this rank. A timing event on the chunk's stream at the
//! end of each phase's stretch of launches (prompt.pm); the interval from one mark to the next is the GPU time of the
//! phase named by the later mark, including any time the GPU waited for the host there (the `host` phase marks the
//! stretches where the host alone was working: the Engram rows' wait). One synchronize a chunk (on its last mark),
//! then one JSON line on stderr (the lane's log):
//!   {"prompt_prof": {"rank", "k", "start", "n", "replay", "sp", "cut_rows", "marks", "ms", "enc_us_row", "dec_ms",
//!    "phases_ms": {phase: ms, ...}}}
//! and after each prompt's last chunk (and at close) the rank's running sums ("prompt_prof_sum").
//! Unset: no Prof exists, prompt.pm is a null test, nothing is recorded and nothing waits.
const std = @import("std");
const cuda = @import("cuda");

/// Where the chunk's GPU time goes:
///   attn    the attention and indexer kernels (window / ring gathers, kv norm + RoPE, sparse attention, the indexer's
///           scores and top-k, the compressor's elementwise steps and FP4 stores)
///   moe     the routed experts' EXL3 kernels (gate / up, down, the parity rest and its assembly)
///   router  the MoE gate's fp32 logits (casts + cuBLAS SGEMM) and the routing
///   dense   the dense EXL3 prompt GEMMs (wq_a, wkv, wq_b, wo_a, wo_b, the compressor's and indexer's projections) and
///           the indexer's cuBLAS head weights
///   mix     the mHC mixes (hc_pre / hc_pre_pf / hc_post), RMSNorms, the DSpark taps, row copies
///   engram  Engram's upload, projection, sum (its arithmetic) and gate
///   comm    every NCCL exchange (its transfer and the wait for the peers to reach it)
///   host    stretches the GPU spent waiting for the host (the Engram rows)
///   other   embed, the replay cut's copies, the head, the drafter's absorb
pub const Phase = enum(u8) { attn, moe, router, dense, mix, engram, comm, host, other };
pub const phases = @typeInfo(Phase).@"enum".field_names;
pub const n_phases = phases.len;

/// A chunk's interval i (mark i - 1 to mark i, i >= 1) to phase[i]: each phase's sum into `out` (added), and the sum of
/// the intervals from mark `dec_from` on (the decoder layers' rows, null: none). Returns that decoder sum.
pub fn attribute(phase: []const Phase, ms: []const f64, dec_from: ?usize, out: *[n_phases]f64) f64 {
    var dec: f64 = 0;
    for (1..phase.len) |i| {
        out[@intFromEnum(phase[i])] += ms[i];
        if (dec_from) |d| if (i > d) {
            dec += ms[i];
        };
    }
    return dec;
}

pub const Prof = struct {
    pub const max_marks = 8192;
    gpa: std.mem.Allocator,
    events: []cuda.Event,
    phase: []Phase,
    ms: []f64,
    n: usize = 0,
    dropped: u64 = 0, // marks past max_marks (their time goes to the chunk's last interval)
    rank: usize,
    // the chunk being timed
    k: u64 = 0,
    start: usize = 0,
    rows: usize = 0,
    replay: usize = 0,
    sp: []const u8 = "off",
    cut_rows: usize = 0, // the rows past the replay cut (0: an encoder-only chunk)
    dec_from: ?usize = null, // the mark at the replay cut
    open: bool = false,
    // the rank's sums over its chunks
    sum: [n_phases]f64 = @splat(0),
    sum_ms: f64 = 0,
    sum_dec_ms: f64 = 0,
    sum_rows: u64 = 0, // the chunks' rows (each chunk's n before any cut)
    sum_enc_rows: u64 = 0, // their rows through the encoder layers
    chunks: u64 = 0,

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, rank: usize) !Prof {
        const ev = try gpa.alloc(cuda.Event, max_marks);
        errdefer gpa.free(ev);
        var made: usize = 0;
        errdefer for (ev[0..made]) |*x| x.deinit();
        for (ev) |*x| {
            x.* = try cuda.Event.init(d, true);
            made += 1;
        }
        const ph = try gpa.alloc(Phase, max_marks);
        errdefer gpa.free(ph);
        const ms = try gpa.alloc(f64, max_marks);
        return .{ .gpa = gpa, .events = ev, .phase = ph, .ms = ms, .rank = rank };
    }

    pub fn deinit(p: *Prof) void {
        for (p.events) |*x| x.deinit();
        p.gpa.free(p.events);
        p.gpa.free(p.phase);
        p.gpa.free(p.ms);
    }

    /// A chunk's first mark (its start), before its first launch.
    pub fn begin(p: *Prof, s: cuda.Stream, start: usize, rows: usize, replay: usize) !void {
        p.n = 0;
        p.start = start;
        p.rows = rows;
        p.replay = replay;
        p.sp = "off";
        p.cut_rows = 0;
        p.dec_from = null;
        p.open = true;
        try p.events[0].record(s);
        p.phase[0] = .other;
        p.n = 1;
    }

    /// The end of a stretch of `ph`'s launches.
    pub fn mark(p: *Prof, s: cuda.Stream, ph: Phase) !void {
        if (!p.open) return;
        if (p.n == max_marks) {
            // full: the last event moves here, its interval takes this one's time (counted as dropped)
            try p.events[p.n - 1].record(s);
            p.dropped += 1;
            return;
        }
        try p.events[p.n].record(s);
        p.phase[p.n] = ph;
        p.n += 1;
    }

    /// The replay cut: the marks after this one time the decoder layers' `rows` rows.
    pub fn cut(p: *Prof, rows: usize) void {
        if (!p.open) return;
        p.dec_from = p.n - 1;
        p.cut_rows = rows;
    }

    /// The chunk's end: its last mark waited for (the one synchronize), its intervals summed by phase, its line out.
    pub fn finish(p: *Prof) !void {
        if (!p.open) return;
        p.open = false;
        if (p.n < 2) return;
        try p.events[p.n - 1].synchronize();
        p.ms[0] = 0;
        for (1..p.n) |i| p.ms[i] = try cuda.Event.elapsedMs(p.events[i - 1], p.events[i]);
        var by: [n_phases]f64 = @splat(0);
        const dec = attribute(p.phase[0..p.n], p.ms[0..p.n], p.dec_from, &by);
        var total: f64 = 0;
        for (by) |x| total += x;
        const enc_rows: f64 = @floatFromInt(@max(p.rows, 1));
        var buf: [2048]u8 = undefined;
        var w: std.Io.Writer = .fixed(&buf);
        w.print("{{\"prompt_prof\": {{\"rank\": {d}, \"k\": {d}, \"start\": {d}, \"n\": {d}, \"replay\": {d}, \"sp\": \"{s}\", \"cut_rows\": {d}, \"marks\": {d}, \"ms\": {d:.3}, \"enc_us_row\": {d:.2}, \"dec_ms\": {d:.3}, \"phases_ms\": {{", .{ p.rank, p.k, p.start, p.rows, p.replay, p.sp, p.cut_rows, p.n, total, (total - dec) * 1000 / enc_rows, dec }) catch {};
        for (phases, 0..) |name, i| w.print("{s}\"{s}\": {d:.3}", .{ if (i == 0) "" else ", ", name, by[i] }) catch {};
        w.print("}}}}}}\n", .{}) catch {};
        std.debug.print("{s}", .{w.buffered()});
        for (by, 0..) |x, i| p.sum[i] += x;
        p.sum_ms += total;
        p.sum_dec_ms += dec;
        p.sum_rows += p.rows;
        p.sum_enc_rows += p.rows;
        p.chunks += 1;
        p.k += 1;
    }

    /// The rank's sums so far: each phase's ms and µs a prompt row (over every chunk's rows), and the encoder's.
    pub fn summary(p: *const Prof) void {
        if (p.chunks == 0) return;
        const rows: f64 = @floatFromInt(@max(p.sum_rows, 1));
        var buf: [2048]u8 = undefined;
        var w: std.Io.Writer = .fixed(&buf);
        w.print("{{\"prompt_prof_sum\": {{\"rank\": {d}, \"chunks\": {d}, \"rows\": {d}, \"ms\": {d:.1}, \"dec_ms\": {d:.1}, \"us_row\": {d:.2}, \"enc_us_row\": {d:.2}, \"dropped_marks\": {d}, \"us_row_by_phase\": {{", .{ p.rank, p.chunks, p.sum_rows, p.sum_ms, p.sum_dec_ms, p.sum_ms * 1000 / rows, (p.sum_ms - p.sum_dec_ms) * 1000 / rows, p.dropped }) catch {};
        for (phases, 0..) |name, i| w.print("{s}\"{s}\": {d:.2}", .{ if (i == 0) "" else ", ", name, p.sum[i] * 1000 / rows }) catch {};
        w.print("}}}}}}\n", .{}) catch {};
        std.debug.print("{s}", .{w.buffered()});
    }
};

/// TF_DS_PROMPT_PROF=1.
pub fn wanted() bool {
    const v = std.c.getenv("TF_DS_PROMPT_PROF") orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "1");
}

test "a chunk's intervals by the later mark's phase; the decoder's from the cut on" {
    const ph = [_]Phase{ .other, .mix, .comm, .attn, .comm, .mix, .moe, .other };
    const ms = [_]f64{ 0, 1, 2, 3, 4, 5, 6, 7 };
    var out: [n_phases]f64 = @splat(0);
    const dec = attribute(&ph, &ms, 4, &out);
    try std.testing.expectEqual(@as(f64, 5 + 6 + 7), dec);
    try std.testing.expectEqual(@as(f64, 1 + 5), out[@intFromEnum(Phase.mix)]);
    try std.testing.expectEqual(@as(f64, 2 + 4), out[@intFromEnum(Phase.comm)]);
    try std.testing.expectEqual(@as(f64, 3), out[@intFromEnum(Phase.attn)]);
    try std.testing.expectEqual(@as(f64, 6), out[@intFromEnum(Phase.moe)]);
    try std.testing.expectEqual(@as(f64, 7), out[@intFromEnum(Phase.other)]); // mark 0 (the start) times nothing
    var total: f64 = 0;
    for (out) |x| total += x;
    try std.testing.expectEqual(@as(f64, 28), total);
    var none: [n_phases]f64 = @splat(0);
    try std.testing.expectEqual(@as(f64, 0), attribute(&ph, &ms, null, &none));
}
