//! TF_DS_PROMPT_OVL (stage ZO1): some of an SP prompt chunk's exchanges on a comm stream beside the compute stream, so
//! their transfer runs under compute that does not need them yet. Exact by construction: no kernel's operands, shapes
//! or summation order change; only when a launch is issued, and on which of two streams, changes.
//!
//!   engram  an Engram layer L's projection (it depends only on the token ids) is issued at layer L - 1 after the
//!           attention mixes, its sum's step 1 (fp32 with the row partner) and the add on the comm stream under layer
//!           L - 1's attention, its step 2 (bf16 with the column partner) and the copies into kv under layer L - 1's
//!           experts; layer L's gate waits for them (needs TF_DS_PROMPT_ENGRAM=sp: Two.engramSumSp's plan)
//!   xag     the attention mixes' x all-gather on the comm stream; meanwhile the attention's row-local front (wq_a,
//!           wkv, q_norm, wq_b) runs on this node's own rows (they are already here), then on the rest once it lands
//!   wob     wo_b for the other owners' rows first, toOwners (which never reads this node's own rows of the partial)
//!           on the comm stream, wo_b for the own rows meanwhile; the copies after it wait for it
//!   prio    the comm stream at the context's greatest priority (its NCCL blocks dispatch first)
//!
//! One NCCL communicator; comm.Order keeps every NCCL operation of the chunk in host issue order on the GPU whichever
//! stream it went on, and the ranks issue the same operations in the same order (every choice here depends only on the
//! chunk's n, start, replay and spans and the env), so no ordering hazard (ZP2's two-communicator deadlock) exists.
const std = @import("std");
const cuda = @import("cuda");
const comm = @import("comm.zig");
const Owners = @import("prompt2d.zig").Owners;

/// The levers TF_DS_PROMPT_OVL asks for.
pub const Levers = struct {
    engram: bool = false,
    xag: bool = false,
    wob: bool = false,
    prio: bool = false,

    pub fn any(l: Levers) bool {
        return l.engram or l.xag or l.wob;
    }

    /// The names of the active levers, as the start line prints them.
    pub fn names(l: Levers, buf: []u8) []const u8 {
        var n: usize = 0;
        inline for (.{ "engram", "xag", "wob", "prio" }) |f| if (@field(l, f)) {
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

/// TF_DS_PROMPT_OVL's value: unset, "" or "off": none (today's launches); else a comma list of engram, xag, wob, prio,
/// or "all" (engram, xag, wob); "prio" alone (no lever) is refused, as is any other word.
pub fn parse(v: ?[]const u8) !Levers {
    const s = v orelse return .{};
    if (s.len == 0 or std.mem.eql(u8, s, "off")) return .{};
    var l: Levers = .{};
    var it = std.mem.splitScalar(u8, s, ',');
    while (it.next()) |w| {
        if (std.mem.eql(u8, w, "engram")) l.engram = true else if (std.mem.eql(u8, w, "xag")) l.xag = true else if (std.mem.eql(u8, w, "wob")) l.wob = true else if (std.mem.eql(u8, w, "prio")) l.prio = true else if (std.mem.eql(u8, w, "all")) {
            l.engram = true;
            l.xag = true;
            l.wob = true;
        } else return error.BadPromptOvl;
    }
    if (!l.any()) return error.BadPromptOvl;
    return l;
}

/// Rows [r0, r0 + rows) of a chunk.
pub const Range = struct { r0: usize, rows: usize };

/// The prompt GEMM (exl3_prefill.matmul) takes launches of more than this many rows; at or under it mm takes the
/// grouped decode path, a different kernel (different bits). A split launch must keep every part above it.
pub const gemm_min_rows = 128;

/// A chunk's rows for a row-split launch on node me, own rows first: [own, before them, after them] (before / after
/// may be empty). Together they cover [0, n) once. null when any non-empty part has gemm_min_rows rows or fewer, or
/// the owners are not valid: the caller then makes today's one launch.
pub fn ownFirst(o: Owners, me: usize) ?[3]Range {
    if (!o.valid() or me > 3) return null;
    const r0 = o.r0(me);
    const rm = o.rows(me);
    const parts = [3]Range{ .{ .r0 = r0, .rows = rm }, .{ .r0 = 0, .rows = r0 }, .{ .r0 = r0 + rm, .rows = o.n - r0 - rm } };
    for (parts) |p| if (p.rows != 0 and p.rows <= gemm_min_rows) return null;
    return parts;
}

/// The Engram layer whose projection and sum layer li issues (TF_DS_PROMPT_OVL=engram): li + 1 when it is an Engram
/// layer before the replay cut (the SP chunk's encoder layers; `cut` = the decoder's first layer); null else (an Engram
/// layer 0 has no layer before it and keeps today's in-place sum).
pub fn engramAhead(engram_layers: []const u16, li: usize, cut: usize) ?usize {
    const nx = li + 1;
    if (nx >= cut) return null;
    if (std.mem.indexOfScalar(u16, engram_layers, @intCast(nx)) == null) return null;
    return nx;
}

/// What a side segment's completion event is for (each kind has at most one segment in flight).
pub const Kind = enum(u8) { xag, engram, wob };

/// The comm stream, its handoff events and the launch order (model.Model.side; set on the engine for SP chunks only).
pub const Side = struct {
    lv: Levers,
    stream: cuda.Stream,
    go: cuda.Event, // compute -> comm: recorded on the compute stream right before a segment, waited for by the comm stream
    done: [3]cuda.Event, // comm -> compute, one per Kind: recorded after a segment, waited for where its result is read
    order: comm.Order,
    segments: u64 = 0, // side segments issued (diagnosis)

    pub fn init(d: *const cuda.Driver, lv: Levers) !Side {
        var s = if (lv.prio) try cuda.Stream.initGreatest(d, true) else try cuda.Stream.init(d, true);
        errdefer s.deinit();
        var go = try cuda.Event.init(d, false);
        errdefer go.deinit();
        var done: [3]cuda.Event = undefined;
        var made: usize = 0;
        errdefer for (done[0..made]) |*x| x.deinit();
        for (&done) |*x| {
            x.* = try cuda.Event.init(d, false);
            made += 1;
        }
        const ev = try cuda.Event.init(d, false);
        return .{ .lv = lv, .stream = s, .go = go, .done = done, .order = .{ .ev = ev } };
    }

    pub fn deinit(s: *Side) void {
        s.order.ev.deinit();
        for (&s.done) |*x| x.deinit();
        s.go.deinit();
        s.stream.deinit();
    }

    /// A segment's start: the comm stream waits for everything issued so far on `main` (its operands and the
    /// readers of what it overwrites).
    pub fn fork(s: *Side, main: cuda.Stream) !void {
        try s.go.record(main);
        try s.stream.wait(s.go);
        s.segments += 1;
    }

    /// A segment's end (issued right after its last launch).
    pub fn end(s: *Side, k: Kind) !void {
        try s.done[@intFromEnum(k)].record(s.stream);
    }

    /// `main` waits for kind k's last segment.
    pub fn join(s: *Side, main: cuda.Stream, k: Kind) !void {
        try main.wait(s.done[@intFromEnum(k)]);
    }

    /// A chunk's start: no operation issued yet in its order.
    pub fn reset(s: *Side) void {
        s.order.last = 0;
    }
};

test "TF_DS_PROMPT_OVL: unset / off is today's; lists, all, and bad words fail" {
    try std.testing.expectEqual(Levers{}, try parse(null));
    try std.testing.expectEqual(Levers{}, try parse(""));
    try std.testing.expectEqual(Levers{}, try parse("off"));
    try std.testing.expectEqual(Levers{ .engram = true }, try parse("engram"));
    try std.testing.expectEqual(Levers{ .xag = true, .wob = true }, try parse("xag,wob"));
    try std.testing.expectEqual(Levers{ .engram = true, .xag = true, .wob = true }, try parse("all"));
    try std.testing.expectEqual(Levers{ .engram = true, .xag = true, .wob = true, .prio = true }, try parse("all,prio"));
    try std.testing.expectError(error.BadPromptOvl, parse("prio"));
    try std.testing.expectError(error.BadPromptOvl, parse("1"));
    try std.testing.expectError(error.BadPromptOvl, parse("engram,"));
    try std.testing.expectError(error.BadPromptOvl, parse("xag,moe"));
    var buf: [64]u8 = undefined;
    try std.testing.expectEqualStrings("engram,xag,wob,prio", (try parse("all,prio")).names(&buf));
    try std.testing.expectEqualStrings("wob", (try parse("wob")).names(&buf));
}

test "ownFirst: own rows first, the rest before and after, every row once; null where a part would leave the prompt GEMM" {
    var n: usize = 256;
    while (n <= 4096) : (n += 1) {
        const o = Owners.of(n);
        for (0..4) |me| {
            const got = ownFirst(o, me);
            // every part above the GEMM threshold <=> the split is offered
            const want = o.rows(me) > gemm_min_rows and (o.r0(me) == 0 or o.r0(me) > gemm_min_rows) and
                (n - o.r0(me) - o.rows(me) == 0 or n - o.r0(me) - o.rows(me) > gemm_min_rows);
            try std.testing.expectEqual(want, got != null);
            const ps = got orelse continue;
            try std.testing.expectEqual(o.r0(me), ps[0].r0);
            try std.testing.expectEqual(o.rows(me), ps[0].rows);
            var hit: [4096]u8 = @splat(0);
            for (ps) |p| {
                if (p.rows != 0) try std.testing.expect(p.rows > gemm_min_rows);
                for (p.r0..p.r0 + p.rows) |r| hit[r] += 1;
                // the specializations of today's launches: non-empty parts start on 16-row tiles and hold whole tiles,
                // except the part holding the chunk's last row, which is in n's %16 class (as today's launch on n)
                if (p.rows != 0) try std.testing.expect(p.r0 % 16 == 0);
                if (p.rows != 0 and p.r0 + p.rows != n) try std.testing.expect(p.rows % 16 == 0);
                if (p.rows != 0 and p.r0 + p.rows == n) try std.testing.expectEqual(n % 16, p.rows % 16);
            }
            for (hit[0..n]) |h| try std.testing.expectEqual(@as(u8, 1), h);
        }
    }
    // the served passes (2,048 and 4,096 rows) always split
    for ([_]usize{ 2048, 4096 }) |nn| for (0..4) |me| try std.testing.expect(ownFirst(Owners.of(nn), me) != null);
    // small rows2 chunks keep the one launch
    try std.testing.expect(ownFirst(Owners.of(256), 0) == null);
    try std.testing.expect(ownFirst(Owners.of(512), 1) == null);
}

test "engramAhead: each Engram layer before the cut issued exactly once, from the layer before it" {
    const layers = [_]u16{ 1, 14 };
    const cut = 21;
    var seen: [64]u8 = @splat(0);
    for (0..cut) |li| if (engramAhead(&layers, li, cut)) |l| {
        try std.testing.expectEqual(li + 1, l);
        seen[l] += 1;
    };
    try std.testing.expectEqual(@as(u8, 1), seen[1]);
    try std.testing.expectEqual(@as(u8, 1), seen[14]);
    var total: usize = 0;
    for (seen) |x| total += x;
    try std.testing.expectEqual(@as(usize, 2), total);
    // an Engram layer 0, or one at / past the cut: no layer issues it ahead
    try std.testing.expect(engramAhead(&[_]u16{0}, 0, cut) == null);
    try std.testing.expect(engramAhead(&[_]u16{21}, 20, cut) == null);
}
