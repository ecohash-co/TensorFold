//! Host checks of the 8-aligned int64 pointer coverage (tri.proven, aot_ptr8.zig) against the captured kit:
//! stage ZF2's refused _sparse_attn_part launch, and every captured variant's launch with every subset of its
//! 16-aligned int64 pointers 8 bytes off. Reads the kit's aot.json (TF_TEST_AOT_JSON; skipped when unset);
//! without one the kit checks are skipped.
const std = @import("std");
const aot = @import("cuda").aot;
const tri = @import("tri.zig");
const tri_attn = @import("tri_attn.zig");
const ptr8 = @import("aot_ptr8.zig");

fn loadKit(gpa: std.mem.Allocator) !?std.json.Parsed(aot.SetJson) {
    const path: []const u8 = if (std.c.getenv("TF_TEST_AOT_JSON")) |v| std.mem.span(v) else return null;
    const text = std.Io.Dir.cwd().readFileAlloc(std.testing.io, path, gpa, .limited(1 << 24)) catch return null;
    defer gpa.free(text);
    return try aot.parseSet(gpa, text);
}

/// Stage ZF2's launch (soak-log-castor.txt, frame 14855): a prompt's last chunk past the replay cut of 16 rows or
/// fewer (ring mode, eight splits), pos `first` rows into the chunk's positions (prompt.replayCut: first odd).
fn zf2(h: usize, pos: u64) tri_attn.Attn {
    return .{
        .q = 0xf044b8400000,
        .out = 0xf04519400000,
        .rows = 9,
        .h = h,
        .hd = 512,
        .sink = 0xf04a2fbffe00,
        .wsrc = 0xf049309c0000,
        .wsrc_rows = 144,
        .wlo = 0xf041ae4f8d00,
        .ring = true,
        .comp = .{ .fp4 = .{ .codes = 0xf04041400000, .scales = 0xf04060480000 } },
        .idx = 0xf04635c50200,
        .n_idx = 512,
        .pos = pos,
        .scale = 0.044194173,
        .window = 128,
        .parts = .{ .pm = 0xf041af501400, .pl = 0xf041af503400, .po = 0xf041af505400 },
    };
}

test "stage ZF2's refused _sparse_attn_part launch (POS = RBASE = CBASE 8 bytes off 16) runs a captured variant" {
    const gpa = std.testing.allocator;
    const kit = (try loadKit(gpa)) orelse return;
    defer kit.deinit();
    const ks = kit.value.kernels;
    const want = [_][2][]const u8{ .{ "32", "f15e153358" }, .{ "16", "462931f085" } };
    for ([_]usize{ 32, 16 }, want) |h, w| {
        _ = w[0];
        const pt = try tri_attn.part(zf2(h, 0xf041a8c550b8), aot.ci("PDL", 1));
        try std.testing.expectEqual(@as(i32, 144), pt.args[15].value.i32);
        try std.testing.expectEqual(pt.args[6].value.ptr.addr, pt.args[13].value.ptr.addr); // RBASE: pos
        try std.testing.expectEqual(pt.args[6].value.ptr.addr, pt.args[14].value.ptr.addr); // CBASE: pos
        // night5 edf22ac: none
        try std.testing.expect(aot.pick(ks, "_sparse_attn_part", &pt.args, &pt.consts, .{}, false) == null);
        const got = aot.pick(ks, "_sparse_attn_part", &pt.args, &pt.consts, tri.proven, false).?;
        try std.testing.expect(!got.exact and std.mem.startsWith(u8, ks[got.index].hash, w[1]));
        // its 16-aligned twin: that variant exactly, as before
        const al = try tri_attn.part(zf2(h, 0xf041a8c550c0), aot.ci("PDL", 1));
        const ex = aot.pick(ks, "_sparse_attn_part", &al.args, &al.consts, tri.proven, false).?;
        try std.testing.expect(ex.exact and std.mem.startsWith(u8, ks[ex.index].hash, w[1]));
    }
}

/// prompt.replayCut at the last chunk the replay (len - window) falls in: the rows past it, `first` rows into the
/// chunk (pos then first * 8 bytes in). The chunk is 2,048 rows unless it is the prompt's last.
fn replayTail(len: usize) ?struct { first: usize, n: usize } {
    if (len <= 128) return null;
    const replay = len - 128;
    const start = replay / 2048 * 2048;
    const end = @min(start + 2048, len);
    return .{ .first = replay - start, .n = end - replay };
}

test "every prompt length to 1M whose replay cut leaves a ring-mode tail at an odd row runs _sparse_attn_part" {
    // ZF2's two refusals: 24,695 and 145,529 tokens (9 and 7 rows, 2,039 and 2,041 rows in)
    try std.testing.expectEqual(@as(usize, 9), replayTail(24695).?.n);
    try std.testing.expectEqual(@as(usize, 2039), replayTail(24695).?.first);
    try std.testing.expectEqual(@as(usize, 7), replayTail(145529).?.n);
    const gpa = std.testing.allocator;
    const kit = (try loadKit(gpa)) orelse return;
    defer kit.deinit();
    var hits: usize = 0;
    var len: usize = 129;
    while (len <= 1 << 20) : (len += 1) {
        const t = replayTail(len).?;
        if (t.first == 0 or t.n > 16) continue; // a cut chunk of more than 16 rows: one split, as captured
        for ([_]usize{ 16, 32 }) |h| {
            var a = zf2(h, 0xf041a8c55000 + t.first * 8);
            a.rows = t.n;
            const pt = try tri_attn.part(a, aot.ci("PDL", 1));
            if (aot.pick(kit.value.kernels, "_sparse_attn_part", &pt.args, &pt.consts, tri.proven, false) == null) {
                std.debug.print("no _sparse_attn_part variant at prompt length {d}\n", .{len});
                return error.TestUnexpectedResult;
            }
        }
        if (t.first % 2 == 1) hits += 1;
    }
    try std.testing.expect(hits > 0); // 776 lengths to 200K (2,161 the first: len % 2048 odd in 113 .. 127)
}

test "the proven table names only captured variants and their 16-aligned int64 parameters" {
    const gpa = std.testing.allocator;
    const kit = (try loadKit(gpa)) orelse return;
    defer kit.deinit();
    for (ptr8.variants) |v| {
        const k = for (kit.value.kernels) |k| {
            if (std.mem.eql(u8, k.hash, v.hash)) break k;
        } else return error.TestUnexpectedResult;
        for (v.ptrs8) |name| {
            const prm = for (k.params) |q| {
                if (std.mem.eql(u8, q.name, name)) break q;
            } else return error.TestUnexpectedResult;
            try std.testing.expect(std.mem.eql(u8, prm.type, "*i64") and prm.div16);
        }
    }
}

/// The (function, parameter) pairs whose 8-aligned launches the kit cannot run (no variant takes them, the proven
/// table included: their variants read them with 128-bit accesses), each with why no call site passes such a pointer.
pub const uncovered = [_]struct { f: []const u8, p: []const u8, why: []const u8 }{
    .{ .f = "_gather_tiles", .p = "OUT", .why = "PrunedBufs.cand, a buffer base" },
    .{ .f = "_score_keys", .p = "OUT", .why = "rd.keys, a buffer base" },
    .{ .f = "_sparse_attn_part", .p = "IDX", .why = "ch.cidx(p) / rd.cidx / dr.bidx, buffer bases (rows of kk * 8 bytes)" },
    .{ .f = "_topk_finish", .p = "TOP", .why = "TopK.top (ch.ktop / rd top), a buffer base" },
    .{ .f = "_topk_finish", .p = "OUT", .why = "cidx + row * kk * 8 (kk 512: 16-aligned) or rd.cidx" },
    .{ .f = "_topk_sel", .p = "KEYS", .why = "ch.keys / rd.keys / PrunedBufs.cand, buffer bases" },
    .{ .f = "_topk_sel", .p = "OUT", .why = "cidx + row * kk * 8 (kk a power of two >= 2: 16-aligned) or PrunedBufs.tpos" },
};

fn uncoveredAt(f: []const u8, p: []const u8) ?usize {
    for (uncovered, 0..) |u, i| if (std.mem.eql(u8, u.f, f) and std.mem.eql(u8, u.p, p)) return i;
    return null;
}

test "every captured variant's launch with any of its 16-aligned int64 pointers 8 bytes off finds a variant" {
    const gpa = std.testing.allocator;
    const kit = (try loadKit(gpa)) orelse return;
    defer kit.deinit();
    const ks = kit.value.kernels;
    var args: std.ArrayList(aot.Arg) = .empty;
    defer args.deinit(gpa);
    var consts: std.ArrayList(aot.Const) = .empty;
    defer consts.deinit(gpa);
    var launches: usize = 0;
    var rescued: usize = 0;
    var needed: [uncovered.len]usize = @splat(0); // launches each entry alone stands for
    for (ks) |k| {
        var ptrs: [8]usize = undefined; // indexes into args of the 16-aligned int64 pointers
        var np: usize = 0;
        args.clearRetainingCapacity();
        consts.clearRetainingCapacity();
        for (k.params, 0..) |prm, i| {
            const base: u64 = 0x100000 * (i + 1);
            if (prm.type[0] == '*') {
                if (std.mem.eql(u8, prm.type, "*i64") and prm.div16) {
                    ptrs[np] = args.items.len;
                    np += 1;
                }
                try args.append(gpa, aot.ptr(prm.name, prm.type, if (prm.div16) base else base + 8));
            } else if (std.mem.eql(u8, prm.type, "i32")) {
                try args.append(gpa, aot.int(prm.name, if (prm.div16) 32 else 17));
            } else if (std.mem.eql(u8, prm.type, "fp32")) {
                try args.append(gpa, aot.float(prm.name, 0.5));
            } else if (std.mem.eql(u8, prm.type, "u64")) {
                try args.append(gpa, aot.word(prm.name, if (prm.div16) 32 else 17));
            } else return error.TestUnexpectedResult;
        }
        // constexprs; an int Triton folded (a runtime argument of value 1: lower-case names) is passed as one
        var it = k.consts.map.iterator();
        while (it.next()) |e| {
            const name = e.key_ptr.*;
            if (std.ascii.isLower(name[0])) {
                try args.append(gpa, aot.int(name, @intCast(e.value_ptr.int.?)));
            } else if (e.value_ptr.int) |x| {
                try consts.append(gpa, aot.ci(name, x));
            } else try consts.append(gpa, .{ .name = name, .f32 = @bitCast(e.value_ptr.f32.?) });
        }
        // the variant's own launch is exact
        const own = aot.pick(ks, k.@"fn", args.items, consts.items, .{}, false).?;
        try std.testing.expect(own.exact);
        var mask: usize = 1;
        while (mask < (@as(usize, 1) << @intCast(np))) : (mask += 1) {
            for (0..np) |j| {
                const a = &args.items[ptrs[j]];
                const al = a.value.ptr.addr & ~@as(u64, 15);
                a.value.ptr.addr = if (mask >> @intCast(j) & 1 != 0) al + 8 else al;
            }
            launches += 1;
            if (aot.pick(ks, k.@"fn", args.items, consts.items, .{}, false) != null) continue;
            if (aot.pick(ks, k.@"fn", args.items, consts.items, tri.proven, false) != null) {
                rescued += 1;
                continue;
            }
            // not runnable: every pointer off must be a documented buffer base (never a row slice)
            var known = false;
            var only: ?usize = null;
            var offs: usize = 0;
            for (0..np) |j| if (mask >> @intCast(j) & 1 != 0) {
                offs += 1;
                if (uncoveredAt(k.@"fn", args.items[ptrs[j]].name)) |u| {
                    known = true;
                    only = u;
                }
            };
            if (offs == 1) if (only) |u| {
                needed[u] += 1;
            };
            if (!known) {
                std.debug.print("no variant: {s} ({s}) with", .{ k.@"fn", k.hash[0..10] });
                for (0..np) |j| if (mask >> @intCast(j) & 1 != 0) std.debug.print(" {s}", .{args.items[ptrs[j]].name});
                std.debug.print(" 8 bytes off\n", .{});
                return error.TestUnexpectedResult;
            }
        }
        for (0..np) |j| {
            const a = &args.items[ptrs[j]];
            a.value.ptr.addr &= ~@as(u64, 15);
        }
    }
    std.debug.print("int64 pointer subsets: {d} launches, {d} run only by the proven table\n", .{ launches, rescued });
    try std.testing.expect(launches > 0 and rescued > 0);
    // the list is no longer than it must be: each entry alone leaves some launch without a variant
    var extra = false;
    for (uncovered, needed) |u, n| if (n == 0) {
        std.debug.print("uncovered entry not needed: {s} {s}\n", .{ u.f, u.p });
        extra = true;
    };
    if (extra) return error.TestUnexpectedResult;
}
