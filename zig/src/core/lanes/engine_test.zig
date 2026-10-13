//! The round loop on the fake target: drafted == one-token rounds, shared rounds == solo, greedy and sampled.
const std = @import("std");
const Config = @import("config.zig").Config;
const Engine = @import("engine.zig").Engine;
const sm = @import("stream.zig");
const fake = @import("fake.zig");
const SuffixLookup = @import("proposer.zig").SuffixLookup;
const Sampling = @import("sampling.zig").Sampling;

const gpa = std.testing.allocator;

const Case = struct {
    prompt: []const u32,
    max_new: u32 = 40,
    sampling: ?Sampling = null,
    drafts: bool = true,
    think_budget: u32 = 0,
};

fn model() !Config {
    return modelWith(true);
}

/// model() with one-token rounds a step ahead or not (a backend without `queue`, as DeepSeek-V4.1's lanes).
fn modelWith(gpu_tokens: bool) !Config {
    var costs: [16]@import("config.zig").Cost = undefined;
    for (&costs, 1..) |*c, w| c.* = .{ .width = @intCast(w), .ms = 5.0 + 0.8 * @as(f64, @floatFromInt(w)) };
    return Config.init(gpa, .{ .exact_width = 16, .gpu_tokens = gpu_tokens, .mtp = true, .speculate = true, .speculate_early = false, .drafts = 4, .window_costs = &costs, .mtp_step_ms = 0.5, .hidden_rows = true, .batch_rows = 32, .max_streams = 8, .draft_streams = true }, 16, 15);
}

/// Every case's emitted tokens, the cases admitted together and stepped until done.
fn run(cases: []const Case) ![][]u32 {
    var cfg = try model();
    defer cfg.deinit(gpa);
    var target: fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: fake.FixedClock = .{};
    var engine = Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer engine.deinit();
    const streams = try gpa.alloc(sm.Stream, cases.len);
    defer gpa.free(streams);
    const proposers = try gpa.alloc(SuffixLookup, cases.len);
    defer gpa.free(proposers);
    for (cases, streams, proposers) |c, *s, *p| {
        p.* = try SuffixLookup.init(gpa, .{ .min_match = 4 });
        s.* = try sm.Stream.init(gpa, .{ .id = "s", .prompt = c.prompt, .max_new = c.max_new, .eos = &.{96}, .sampling = c.sampling, .drafts = c.drafts, .proposer = p.proposer(), .think_budget = c.think_budget, .think_close = &.{ 90, 91, 92 }, .think_end = 91 });
    }
    defer for (streams, proposers) |*s, *p| {
        s.deinit(gpa);
        p.deinit();
    };
    for (streams) |*s| try engine.addStream(s);
    while (engine.activeCount() > 0) try engine.step();
    const out = try gpa.alloc([]u32, cases.len);
    for (out, streams) |*o, *s| o.* = try gpa.dupe(u32, s.emitted());
    return out;
}

fn free(runs: [][]u32) void {
    for (runs) |r| gpa.free(r);
    gpa.free(runs);
}

const GuardCase = struct {
    prompt: []const u32,
    max_new: u32 = 400,
    drafts: bool = true,
    think_budget: u32 = 0,
    loop_guard: bool = true,
    think_open: ?bool = null,
    cycle_after: usize = 70,
    pattern: []const u32 = &.{ 11, 12, 13 },
    answer_cycles: bool = false,
};

const GuardOut = struct {
    emitted: []u32,
    reason: sm.Reason,
    period: ?u32,
    finished: bool,
};

fn runGuard(cases: []const GuardCase, step_limit: usize) ![]GuardOut {
    var cfg = try model();
    defer cfg.deinit(gpa);
    const c = cases[0];
    var target: fake.Fake = .{ .gpa = gpa, .cycle_after = c.cycle_after, .probe_prompt = c.prompt.len, .pattern = c.pattern, .answer_cycles = c.answer_cycles };
    defer target.deinit();
    var clock: fake.FixedClock = .{};
    var engine = Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer engine.deinit();
    const streams = try gpa.alloc(sm.Stream, cases.len);
    defer gpa.free(streams);
    const proposers = try gpa.alloc(SuffixLookup, cases.len);
    defer gpa.free(proposers);
    for (cases, streams, proposers) |item, *s, *p| {
        p.* = try SuffixLookup.init(gpa, .{ .min_match = 4 });
        s.* = try sm.Stream.init(gpa, .{ .id = "guard", .prompt = item.prompt, .max_new = item.max_new, .eos = &.{96}, .drafts = item.drafts, .proposer = p.proposer(), .think_budget = item.think_budget, .think_close = &.{ 90, 91, 92 }, .think_end = 91, .think_open = item.think_open, .loop_guard = item.loop_guard });
    }
    defer for (streams, proposers) |*s, *p| {
        s.deinit(gpa);
        p.deinit();
    };
    for (streams) |*s| try engine.addStream(s);
    var steps: usize = 0;
    while (engine.activeCount() > 0 and steps < step_limit) : (steps += 1) try engine.step();
    const out = try gpa.alloc(GuardOut, cases.len);
    for (out, streams) |*result, *s| result.* = .{ .emitted = try gpa.dupe(u32, s.emitted()), .reason = s.reason, .period = s.loop_period, .finished = s.finished };
    return out;
}

fn freeGuard(runs: []GuardOut) void {
    for (runs) |run_result| gpa.free(run_result.emitted);
    gpa.free(runs);
}

const p1 = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5 };
const p2 = [_]u32{ 2, 7, 1, 8, 2, 8, 1, 8, 2, 8, 4, 5, 9 };

test "drafted rounds commit the one-token decode, greedy and sampled" {
    for ([_]?Sampling{ null, .{ .seed = 5, .temperature = 0.7, .top_k = 0, .top_p = 0.95 } }) |s| {
        const drafted = try run(&.{.{ .prompt = &p1, .sampling = s }});
        defer free(drafted);
        const plain = try run(&.{.{ .prompt = &p1, .sampling = s, .drafts = false }});
        defer free(plain);
        try std.testing.expectEqualSlices(u32, plain[0], drafted[0]);
        // and the fake target's own decode
        var history: std.ArrayList(u32) = .empty;
        defer history.deinit(gpa);
        try history.appendSlice(gpa, &p1);
        for (drafted[0]) |t| {
            try std.testing.expectEqual(fake.next(history.items, s, history.items.len), t);
            try history.append(gpa, t);
        }
    }
}

test "shared rounds commit what each stream commits alone" {
    const sampled: Sampling = .{ .seed = 9, .temperature = 1.0, .top_k = 0, .top_p = 0.9 };
    const together = try run(&.{ .{ .prompt = &p1 }, .{ .prompt = &p2, .sampling = sampled, .max_new = 30 } });
    defer free(together);
    const one = try run(&.{.{ .prompt = &p1 }});
    defer free(one);
    const two = try run(&.{.{ .prompt = &p2, .sampling = sampled, .max_new = 30 }});
    defer free(two);
    try std.testing.expectEqualSlices(u32, one[0], together[0]);
    try std.testing.expectEqualSlices(u32, two[0], together[1]);
}

test "the thinking budget's forced close is the same drafted, plain and shared" {
    const drafted = try run(&.{.{ .prompt = &p2, .think_budget = 9 }});
    defer free(drafted);
    const plain = try run(&.{.{ .prompt = &p2, .think_budget = 9, .drafts = false }});
    defer free(plain);
    const shared = try run(&.{ .{ .prompt = &p2, .think_budget = 9 }, .{ .prompt = &p1, .drafts = false } });
    defer free(shared);
    try std.testing.expectEqualSlices(u32, plain[0], drafted[0]);
    try std.testing.expectEqualSlices(u32, plain[0], shared[0]);
    try std.testing.expectEqual(@as(u32, 90), drafted[0][8]);
}

test "the loop guard matches drafted and plain, reports its period, and answers after close" {
    const plain = try runGuard(&.{.{ .prompt = &p2, .drafts = false }}, 5000);
    defer freeGuard(plain);
    const drafted = try runGuard(&.{.{ .prompt = &p2 }}, 5000);
    defer freeGuard(drafted);
    try std.testing.expectEqualSlices(u32, plain[0].emitted, drafted[0].emitted);
    try std.testing.expectEqual(@as(?u32, 3), drafted[0].period);
    try std.testing.expect(plain[0].finished and plain[0].reason == .stop);
    try std.testing.expectEqualSlices(u32, &.{ 90, 91, 92, 40, 41, 42, 96 }, plain[0].emitted[329..]);

    const off = try runGuard(&.{.{ .prompt = &p2, .loop_guard = false, .think_open = true }}, 5000);
    defer freeGuard(off);
    try std.testing.expect(off[0].finished and off[0].reason == .length and off[0].emitted.len == 400);
}

test "a cycle in the answer does not refire or reclose" {
    const plain = try runGuard(&.{.{ .prompt = &p2, .drafts = false, .max_new = 800, .answer_cycles = true }}, 5000);
    defer freeGuard(plain);
    const drafted = try runGuard(&.{.{ .prompt = &p2, .max_new = 800, .answer_cycles = true }}, 5000);
    defer freeGuard(drafted);
    try std.testing.expectEqualSlices(u32, plain[0].emitted, drafted[0].emitted);
    try std.testing.expectEqual(@as(?u32, 3), plain[0].period);
    try std.testing.expect(plain[0].finished and plain[0].reason == .length);
    var close_count: usize = 0;
    for (plain[0].emitted) |token| close_count += @intFromBool(token == 90);
    try std.testing.expectEqual(@as(usize, 1), close_count);
}

test "logprobs: each emitted token scored from the row that drew it, forced ones too; the tokens unchanged" {
    // the verify rounds score; one-token rounds queued a step ahead (pipelined) leave their tokens unscored
    var cfg = try modelWith(false);
    defer cfg.deinit(gpa);
    const force = [_]u32{ 50, 51, 52, 53, 54, 55, 56, 57, 58 }; // teacher-forced: more than a window's rows
    for ([_]bool{ true, false }) |drafts| for ([_][]const u32{ &.{}, &force }) |forced| {
        var emitted: [2][]u32 = undefined;
        for ([_]?u8{ null, 3 }, 0..) |lp, i| {
            var target: fake.Fake = .{ .gpa = gpa, .reports = true };
            defer target.deinit();
            var clock: fake.FixedClock = .{};
            var engine = Engine.init(gpa, &cfg, target.backend(), clock.clock());
            defer engine.deinit();
            var p = try SuffixLookup.init(gpa, .{ .min_match = 4 });
            defer p.deinit();
            var s = try sm.Stream.init(gpa, .{ .id = "lp", .prompt = &p2, .max_new = 40, .eos = &.{96}, .drafts = drafts, .proposer = p.proposer(), .logprobs = lp, .force = forced });
            defer s.deinit(gpa);
            try engine.addStream(&s);
            while (engine.activeCount() > 0) try engine.step();
            emitted[i] = try gpa.dupe(u32, s.emitted());
            if (lp == null) {
                try std.testing.expectEqual(@as(usize, 0), s.scored.items.len);
                continue;
            }
            try std.testing.expectEqual(s.emitted().len, s.scored.items.len);
            try std.testing.expectEqualSlices(u32, forced, s.emitted()[0..forced.len]);
            var history: std.ArrayList(u32) = .empty;
            defer history.deinit(gpa);
            try history.appendSlice(gpa, &p2);
            for (s.emitted(), s.scored.items) |t, rep| {
                // the row that drew at this position: the fake's draw after the history
                const drew = fake.next(history.items, null, history.items.len);
                const want = fake.reportFor(drew, 3, t);
                try std.testing.expectEqual(t, rep.token);
                try std.testing.expect(rep.scored());
                try std.testing.expectEqual(want.forToken(t).logprob, rep.logprob);
                try std.testing.expectEqualSlices(u32, want.ids[0..3], rep.top());
                if (t == drew) try std.testing.expectEqual(rep.lps[0], rep.logprob);
                try history.append(gpa, t);
            }
        }
        defer for (emitted) |e| gpa.free(e);
        try std.testing.expectEqualSlices(u32, emitted[0], emitted[1]);
    };
}

test "logprobs in shared rounds: the asking stream's reports are its own rows', the other stream does no extra work" {
    var cfg = try modelWith(false);
    defer cfg.deinit(gpa);
    var target: fake.Fake = .{ .gpa = gpa, .reports = true };
    defer target.deinit();
    var clock: fake.FixedClock = .{};
    var engine = Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer engine.deinit();
    var pa = try SuffixLookup.init(gpa, .{ .min_match = 4 });
    defer pa.deinit();
    var pb = try SuffixLookup.init(gpa, .{ .min_match = 4 });
    defer pb.deinit();
    var a = try sm.Stream.init(gpa, .{ .id = "a", .prompt = &p1, .max_new = 30, .eos = &.{96}, .proposer = pa.proposer(), .logprobs = 2, .force = &.{ 60, 61, 62 } });
    defer a.deinit(gpa);
    var b = try sm.Stream.init(gpa, .{ .id = "b", .prompt = &p2, .max_new = 30, .eos = &.{96}, .proposer = pb.proposer() });
    defer b.deinit(gpa);
    try engine.addStream(&a);
    try engine.addStream(&b);
    while (engine.activeCount() > 0) try engine.step();
    try std.testing.expectEqual(@as(usize, 0), b.scored.items.len);
    try std.testing.expectEqual(a.emitted().len, a.scored.items.len);
    var history: std.ArrayList(u32) = .empty;
    defer history.deinit(gpa);
    try history.appendSlice(gpa, &p1);
    for (a.emitted(), a.scored.items) |t, rep| {
        const want = fake.reportFor(fake.next(history.items, null, history.items.len), 2, t);
        try std.testing.expect(rep.scored());
        try std.testing.expectEqual(want.forToken(t).logprob, rep.logprob);
        try history.append(gpa, t);
    }
    // the other stream's reply is the one it makes alone
    const alone = try run(&.{.{ .prompt = &p2, .max_new = 30 }});
    defer free(alone);
    try std.testing.expectEqualSlices(u32, alone[0], b.emitted());
}
