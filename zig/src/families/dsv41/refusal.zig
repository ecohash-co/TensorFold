//! A primitive a rank refuses (lanes.follow's `refused`, rank 0's request error) is safe to step past only when every
//! rank refused the same frame and every rank's device then drained: then each rank posted the same collectives (a
//! collective completes on one rank only when all ranks posted it, so a rank that stopped early would leave the others'
//! drains waiting) and the ranks are back in lockstep for the next frame. Rank-dependent work (the four-node split's
//! owned rows, TF_DS_PROMPT_ATTN=isplit's row blocks) can refuse on one rank only (stage ZS2: helen refused a fill
//! chunk with MissingTritonVariant while ranks 0-2 waited in the chunk's collectives forever, /v1/models still up).
//!
//! So each follower, after a refusal and its device drain, sends rank 0 a report frame back over its link (rank 0
//! sends frames down the link, nothing else reads it on rank 0 after the start). Rank 0's Guard thread reads them; rank
//! 0 notes its own refusals the same way after its drain. A refused frame every rank reports within the wait
//! (TF_DS_REFUSAL_WAIT_S, 30 s) is agreed: rank 0 returns the error for that request (HTTP 500 with the reason) and
//! serves the next. Anything else (a rank that does not refuse the frame, a drain that never ends, a follower whose
//! link closes, a report of another frame while one is open) leaves the ranks out of lockstep with work queued that can
//! never complete or pair correctly, which no rank can undo short of tearing NCCL down: the Guard prints why and ends
//! rank 0's process (exit 70), so the serve check / watchdog restarts the four ranks instead of a silent hang.
const std = @import("std");
const c = std.c;
const link = @import("link.zig");

/// The back channel's frame (a follower to rank 0): tag, the refused frame's step (u64 little-endian), the error name.
pub const report_tag: u8 = 0xF1;

/// Rank 0 ends its process with this code when the ranks lose lockstep.
pub const exit_code: u8 = 70;

/// How long rank 0 waits for every rank's report of a refused frame (TF_DS_REFUSAL_WAIT_S, seconds; default 30).
pub fn waitNs() u64 {
    const default: u64 = 30;
    const s = if (c.getenv("TF_DS_REFUSAL_WAIT_S")) |v| std.fmt.parseInt(u64, std.mem.span(v), 10) catch default else default;
    return @max(s, 1) * std.time.ns_per_s;
}

pub fn encodeReport(buf: []u8, step: u64, name: []const u8) []const u8 {
    const n = @min(name.len, buf.len - 9);
    buf[0] = report_tag;
    std.mem.writeInt(u64, buf[1..9], step, .little);
    @memcpy(buf[9..][0..n], name[0..n]);
    return buf[0 .. 9 + n];
}

pub const Report = struct { step: u64, name: []const u8 };

pub fn decodeReport(frame: []const u8) !Report {
    if (frame.len < 9 or frame[0] != report_tag) return error.BadReport;
    return .{ .step = std.mem.readInt(u64, frame[1..9], .little), .name = frame[9..] };
}

/// What rank 0 knows of the open refusal: its frame's step, the ranks that refused it (bit g), since when.
pub const Agree = struct {
    world: u32,
    step: u64 = 0,
    mask: u32 = 0, // 0: no refusal open
    since: u64 = 0,

    pub const Verdict = enum { none, pending, agreed, diverged };

    fn all(a: *const Agree) u32 {
        return (@as(u32, 1) << @intCast(a.world)) - 1;
    }

    /// Rank `rank` refused frame `step` at `now`: diverged when another frame's refusal is open.
    pub fn add(a: *Agree, step: u64, rank: u32, now: u64) Verdict {
        if (rank >= a.world) return .diverged;
        if (a.mask != 0 and step != a.step) return .diverged;
        if (a.mask == 0) {
            a.step = step;
            a.since = now;
        }
        a.mask |= @as(u32, 1) << @intCast(rank);
        return if (a.mask == a.all()) .agreed else .pending;
    }

    /// The open refusal at `now`: agreed once every rank reported it, diverged once `wait_ns` passed without that.
    pub fn verdict(a: *const Agree, now: u64, wait_ns: u64) Verdict {
        if (a.mask == 0) return .none;
        if (a.mask == a.all()) return .agreed;
        return if (now -| a.since > wait_ns) .diverged else .pending;
    }

    pub fn clear(a: *Agree) void {
        a.mask = 0;
    }
};

pub fn monotonicNs() u64 {
    var ts: c.timespec = undefined;
    _ = c.clock_gettime(c.CLOCK.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

fn sleepMs(ms: u64) void {
    var ts: c.timespec = .{ .sec = @intCast(ms / 1000), .nsec = @intCast((ms % 1000) * std.time.ns_per_ms) };
    _ = c.nanosleep(&ts, &ts);
}

/// The process's end when the ranks lost lockstep (the Guard's hook on rank 0).
pub fn exitLoudly(why: []const u8) void {
    std.debug.print("{{\"lockstep_lost\": \"{s}\", \"exit\": {d}}}\n", .{ why, exit_code });
    std.log.err("dsv41: the ranks lost lockstep ({s}); ending rank 0 (exit {d}) so the serve restarts all four ranks", .{ why, exit_code });
    c._exit(exit_code);
}

/// Rank 0's watch over its followers' reports and links (a thread), and the open refusal's agreement.
pub const Guard = struct {
    agree: Agree,
    wait_ns: u64,
    fds: [3]c.fd_t = .{ -1, -1, -1 },
    nf: usize = 0,
    lock: std.atomic.Value(bool) = .init(false),
    stopping: std.atomic.Value(bool) = .init(false),
    losing: std.atomic.Value(bool) = .init(false),
    lost: std.atomic.Value(bool) = .init(false),
    why: [256]u8 = undefined,
    why_len: usize = 0,
    hook: *const fn ([]const u8) void,
    thread: ?std.Thread = null,

    /// Watches `followers` (rank g's link at g - 1) for a world of followers.len + 1 ranks; `hook` runs once when the
    /// ranks lose lockstep (exitLoudly: it does not return).
    pub fn start(gpa: std.mem.Allocator, followers: []const ?link.Link, wait_ns: u64, hook: *const fn ([]const u8) void) !*Guard {
        const g = try gpa.create(Guard);
        errdefer gpa.destroy(g);
        g.* = .{ .agree = .{ .world = @intCast(followers.len + 1) }, .wait_ns = wait_ns, .hook = hook };
        for (followers) |l| {
            g.fds[g.nf] = (l orelse return error.NoFollowerLink).fd;
            g.nf += 1;
        }
        g.thread = try std.Thread.spawn(.{}, loop, .{g});
        return g;
    }

    /// Before rank 0 tells the followers it is done (their links then close): the thread ends, the Guard is freed.
    pub fn stop(g: *Guard, gpa: std.mem.Allocator) void {
        g.stopping.store(true, .release);
        if (g.thread) |t| t.join();
        gpa.destroy(g);
    }

    pub fn lostLockstep(g: *const Guard) bool {
        return g.lost.load(.acquire);
    }

    fn acquire(g: *Guard) void {
        while (g.lock.swap(true, .acquire)) std.atomic.spinLoopHint();
    }

    fn release(g: *Guard) void {
        g.lock.store(false, .release);
    }

    fn lose(g: *Guard, comptime fmt: []const u8, args: anytype) void {
        if (g.losing.swap(true, .acq_rel)) return;
        const w = std.fmt.bufPrint(&g.why, fmt, args) catch g.why[0..];
        g.why_len = w.len;
        g.lost.store(true, .release); // after `why` (lostLockstep readers read it)
        g.hook(g.why[0..g.why_len]);
    }

    /// Rank `rank` refused frame `step` (`name`: the error), its device drained.
    pub fn note(g: *Guard, step: u64, rank: u32, name: []const u8) void {
        g.acquire();
        const open = g.agree.mask != 0;
        const open_step = g.agree.step;
        const v = g.agree.add(step, rank, monotonicNs());
        g.release();
        std.log.warn("dsv41: rank {d} refused frame {d} ({s})", .{ rank, step, name });
        if (v == .diverged) {
            if (open) g.lose("rank {d} refused frame {d} ({s}) while frame {d}'s refusal was open", .{ rank, step, name, open_step }) else g.lose("rank {d} out of the world refused frame {d}", .{ rank, step });
        }
    }

    /// Rank 0's main thread after its own refusal of frame `step` (noted, its device drained): waits until every rank
    /// refused that frame (true: the request ends with the error and rank 0 serves on). False once the ranks lost
    /// lockstep (with exitLoudly the process ends first).
    pub fn awaitAgreed(g: *Guard, step: u64) bool {
        while (true) {
            if (g.lostLockstep()) return false;
            g.acquire();
            const v = g.agree.verdict(monotonicNs(), g.wait_ns);
            const same = g.agree.step == step;
            if (v == .agreed and same) g.agree.clear();
            g.release();
            switch (v) {
                .agreed => if (same) return true,
                .diverged => {
                    g.timedOut();
                    return false;
                },
                else => {},
            }
            sleepMs(1);
        }
    }

    fn timedOut(g: *Guard) void {
        g.acquire();
        const a = g.agree;
        g.release();
        g.lose("frame {d} refused by ranks mask 0x{x} of {d}, the others did not refuse it (or never drained) within {d} s", .{ a.step, a.mask, a.world, g.wait_ns / std.time.ns_per_s });
    }

    fn loop(g: *Guard) void {
        var pfd: [3]c.pollfd = undefined;
        for (0..g.nf) |i| pfd[i] = .{ .fd = g.fds[i], .events = c.POLL.IN, .revents = 0 };
        var buf: [512]u8 = undefined;
        while (!g.stopping.load(.acquire) and !g.lostLockstep()) {
            const r = c.poll(&pfd, @intCast(g.nf), 100);
            if (g.stopping.load(.acquire)) return;
            if (r > 0) for (pfd[0..g.nf], 0..) |*p, i| {
                if (p.revents == 0) continue;
                const rank: u32 = @intCast(i + 1);
                const l: link.Link = .{ .fd = g.fds[i] };
                const frame = l.recv(&buf) catch |err| {
                    if (g.stopping.load(.acquire)) return;
                    g.lose("rank {d}'s link to rank 0 closed ({s}): that follower ended", .{ rank, @errorName(err) });
                    return;
                };
                const rep = decodeReport(frame) catch {
                    std.log.err("dsv41: rank {d} sent rank 0 a frame that is no refusal report ({d} bytes); ignored", .{ rank, frame.len });
                    continue;
                };
                g.note(rep.step, rank, rep.name);
            };
            g.acquire();
            const v = g.agree.verdict(monotonicNs(), g.wait_ns);
            g.release();
            if (v == .diverged) {
                g.timedOut();
                return;
            }
        }
    }
};

// -- tests ----------------------------------------------------------------------------------------------------------

test "a refusal is agreed only when every rank reports the same frame in time" {
    var a: Agree = .{ .world = 4 };
    try std.testing.expectEqual(Agree.Verdict.none, a.verdict(0, 10));
    try std.testing.expectEqual(Agree.Verdict.pending, a.add(7, 3, 100));
    try std.testing.expectEqual(Agree.Verdict.pending, a.add(7, 0, 101));
    try std.testing.expectEqual(Agree.Verdict.pending, a.verdict(105, 10));
    try std.testing.expectEqual(Agree.Verdict.diverged, a.verdict(111, 10)); // ranks 1 and 2 never did
    try std.testing.expectEqual(Agree.Verdict.diverged, a.add(8, 1, 106)); // another frame while 7 is open
    try std.testing.expectEqual(Agree.Verdict.pending, a.add(7, 1, 106));
    try std.testing.expectEqual(Agree.Verdict.pending, a.add(7, 1, 106)); // a frame refused twice (snap_restore's two primitives)
    try std.testing.expectEqual(Agree.Verdict.agreed, a.add(7, 2, 107));
    try std.testing.expectEqual(Agree.Verdict.agreed, a.verdict(1000, 10));
    a.clear();
    try std.testing.expectEqual(Agree.Verdict.none, a.verdict(1000, 10));
    try std.testing.expectEqual(Agree.Verdict.diverged, a.add(9, 4, 0)); // no rank 4 in a world of 4
}

test "report frames read back" {
    var b: [64]u8 = undefined;
    const f = encodeReport(&b, 1 << 40, "MissingTritonVariant");
    const r = try decodeReport(f);
    try std.testing.expectEqual(@as(u64, 1 << 40), r.step);
    try std.testing.expectEqualStrings("MissingTritonVariant", r.name);
    try std.testing.expectError(error.BadReport, decodeReport("step"));
}

/// A loopback world for the Guard: rank 0's listening links and the followers' connected ends.
const Loop = struct {
    ranks0: [3]?link.Link = .{ null, null, null },
    ends: [3]?link.Link = .{ null, null, null },
    n: usize,

    fn open(n: usize) !Loop {
        var lp: Loop = .{ .n = n };
        const io = std.testing.io;
        const base: u16 = 30000 + @as(u16, @intCast(@mod(std.Io.Timestamp.now(io, .awake).nanoseconds, 2000))) * 4;
        const C = struct {
            fn run(port: u16, out: *?link.Link) void {
                out.* = link.Link.connect(std.testing.io, .{ 127, 0, 0, 1 }, port, 10) catch null;
            }
        };
        for (0..n) |i| {
            const t = try std.Thread.spawn(.{}, C.run, .{ base + @as(u16, @intCast(i)), &lp.ends[i] });
            lp.ranks0[i] = try link.Link.listen(.{ 127, 0, 0, 1 }, base + @as(u16, @intCast(i)));
            t.join();
            if (lp.ends[i] == null) return error.ConnectFailed;
        }
        return lp;
    }

    fn close(lp: *Loop) void {
        for (&lp.ranks0, &lp.ends) |*a, *b| {
            if (a.*) |*x| x.close();
            if (b.*) |*x| x.close();
        }
    }

    fn report(lp: *Loop, rank: usize, step: u64, name: []const u8) !void {
        var b: [64]u8 = undefined;
        try lp.ends[rank - 1].?.send(encodeReport(&b, step, name));
    }
};

var test_lost: std.atomic.Value(u32) = .init(0);
fn testHook(_: []const u8) void {
    _ = test_lost.fetchAdd(1, .monotonic);
}

test "every rank refuses the frame: rank 0's request fails and the ranks serve on" {
    const gpa = std.testing.allocator;
    var lp = try Loop.open(3);
    defer lp.close();
    test_lost.store(0, .monotonic);
    const g = try Guard.start(gpa, lp.ranks0[0..3], 5 * std.time.ns_per_s, testHook);
    defer g.stop(gpa);
    // rank 0 and its three followers refuse frame 41 (a symmetric refusal: TooManyRows, ChunkTooLong, ...)
    try lp.report(2, 41, "ChunkTooLong");
    g.note(41, 0, "ChunkTooLong");
    try lp.report(1, 41, "ChunkTooLong");
    try lp.report(3, 41, "ChunkTooLong");
    try std.testing.expect(g.awaitAgreed(41));
    // and the next refused frame is agreed the same way
    try lp.report(1, 57, "ChunkTooLong");
    try lp.report(2, 57, "ChunkTooLong");
    try lp.report(3, 57, "ChunkTooLong");
    g.note(57, 0, "ChunkTooLong");
    try std.testing.expect(g.awaitAgreed(57));
    try std.testing.expect(!g.lostLockstep());
    try std.testing.expectEqual(@as(u32, 0), test_lost.load(.monotonic));
}

test "stage ZS2: one follower refuses a frame rank 0 runs: the Guard ends the wait instead of hanging" {
    const gpa = std.testing.allocator;
    var lp = try Loop.open(3);
    defer lp.close();
    test_lost.store(0, .monotonic);
    const g = try Guard.start(gpa, lp.ranks0[0..3], 300 * std.time.ns_per_ms, testHook);
    defer g.stop(gpa);
    // helen (rank 3) refuses fill chunk frame 1234; ranks 0-2 sit in the chunk's collectives (rank 0's main thread
    // never returns, so only the Guard's thread can act)
    try lp.report(3, 1234, "MissingTritonVariant");
    const t0 = monotonicNs();
    while (!g.lostLockstep() and monotonicNs() - t0 < 5 * std.time.ns_per_s) sleepMs(5);
    try std.testing.expect(g.lostLockstep());
    try std.testing.expectEqual(@as(u32, 1), test_lost.load(.monotonic));
    try std.testing.expect(std.mem.indexOf(u8, g.why[0..g.why_len], "frame 1234 refused by ranks mask 0x8") != null);
}

test "rank 0 refuses a frame its followers ran: its wait ends without agreement" {
    const gpa = std.testing.allocator;
    var lp = try Loop.open(1);
    defer lp.close();
    test_lost.store(0, .monotonic);
    const g = try Guard.start(gpa, lp.ranks0[0..1], 200 * std.time.ns_per_ms, testHook);
    defer g.stop(gpa);
    g.note(9, 0, "MissingTritonVariant");
    try std.testing.expect(!g.awaitAgreed(9));
    try std.testing.expect(g.lostLockstep());
    try std.testing.expectEqual(@as(u32, 1), test_lost.load(.monotonic));
}

test "a follower whose link closes ends rank 0's serving loudly" {
    const gpa = std.testing.allocator;
    var lp = try Loop.open(2);
    defer lp.close();
    test_lost.store(0, .monotonic);
    const g = try Guard.start(gpa, lp.ranks0[0..2], 5 * std.time.ns_per_s, testHook);
    defer g.stop(gpa);
    lp.ends[1].?.close();
    lp.ends[1] = null;
    const t0 = monotonicNs();
    while (!g.lostLockstep() and monotonicNs() - t0 < 5 * std.time.ns_per_s) sleepMs(5);
    try std.testing.expect(g.lostLockstep());
    try std.testing.expect(std.mem.indexOf(u8, g.why[0..g.why_len], "rank 2's link") != null);
}

test "a stopped Guard ignores the followers' links closing (rank 0 said done)" {
    const gpa = std.testing.allocator;
    var lp = try Loop.open(1);
    defer lp.close();
    test_lost.store(0, .monotonic);
    const g = try Guard.start(gpa, lp.ranks0[0..1], 5 * std.time.ns_per_s, testHook);
    g.stop(gpa);
    lp.ends[0].?.close();
    lp.ends[0] = null;
    sleepMs(150);
    try std.testing.expectEqual(@as(u32, 0), test_lost.load(.monotonic));
}
