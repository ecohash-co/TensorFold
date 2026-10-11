//! TF_DS_PROMPT_ATTN (stage ZT1): an SP prompt chunk's attention without the work every node of the four repeats.
//! Exact by construction: every kernel keeps its operands, specialization class and per-element summation order; a
//! launch only covers fewer rows (the kernels involved are row-local: each output row depends on its own input row
//! alone), and the rows another node computed arrive as bytes.
//!
//!   front   (layers that read x for the attention front alone: no kv source, no indexer) wq_a, wkv and q_norm on this
//!           node's own rows only (each node computes the rows it mixed), then qr and kv (1,792 bf16 a row) all-gathered
//!           in place instead of x (5,120 a row); wq_b of the own rows runs while that gather is in flight (on the
//!           OVL comm stream when there is one), then of the other rows. Every node used to compute wq_a / wkv / q_norm
//!           of every row.
//!   isplit  (the indexer layers: the kv sources' index queries) the indexer's queries (idx_wq_b, RoPE, fp4_qd) and its
//!           selection (scores, tile maxima, top-k) for this node's own rows only, then the selection (int64 [rows, k])
//!           all-gathered in place; the head weights (cuBLAS, whose algorithm may depend on M) stay on every row as
//!           today. Every node used to score and select every row against every compressed key.
//!
//!   tsel    (any prompt chunk's keyed selection) the indexer's top-k of a row block in one launch a row
//!           (attn_prompt.cu tf_ds_tsel_kernel) in place of topk_select_pruned's chain (topk_select of the tile
//!           maxima, _gather_tiles, torch's top-k of the gathered keys, _topk_finish) and of topk_select's torch path:
//!           the same set of the k largest (unique) keys, so the same sorted positions.
//!
//! front and isplit need SP rows (TF_DS_PROMPT_SP) on the four-node split, act on SP chunks' encoder layers only, and
//! only when every owner's rows exceed the prompt GEMM's 128 (else that chunk keeps today's launches, the same choice
//! on every rank: it depends only on n). Unset: today's launches.
const std = @import("std");
const cuda = @import("cuda");
const ovl = @import("ovl.zig");
const Owners = @import("prompt2d.zig").Owners;

pub const Levers = struct {
    front: bool = false,
    isplit: bool = false,
    tsel: bool = false,

    pub fn any(l: Levers) bool {
        return l.front or l.isplit or l.tsel;
    }

    /// A lever that needs the four-node split's SP rows.
    pub fn sp(l: Levers) bool {
        return l.front or l.isplit;
    }

    /// The active levers' names, as the start line prints them.
    pub fn names(l: Levers, buf: []u8) []const u8 {
        var n: usize = 0;
        inline for (.{ "front", "isplit", "tsel" }) |f| if (@field(l, f)) {
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

/// TF_DS_PROMPT_ATTN's value: unset, "" or "off": none; else a comma list of front, isplit, tsel, or "all" (the
/// three); any other word is refused (the engine does not start).
pub fn parse(v: ?[]const u8) !Levers {
    const s = v orelse return .{};
    if (s.len == 0 or std.mem.eql(u8, s, "off")) return .{};
    var l: Levers = .{};
    var it = std.mem.splitScalar(u8, s, ',');
    while (it.next()) |w| {
        if (std.mem.eql(u8, w, "front")) l.front = true else if (std.mem.eql(u8, w, "isplit")) l.isplit = true else if (std.mem.eql(u8, w, "tsel")) l.tsel = true else if (std.mem.eql(u8, w, "all")) {
            l.front = true;
            l.isplit = true;
            l.tsel = true;
        } else return error.BadPromptAttn;
    }
    return l;
}

/// front and isplit act on an SP chunk only when its four owners each have more than the prompt GEMM's 128 rows (each
/// own-row launch stays the prompt GEMM, as on the whole chunk): a function of n alone, so the same answer on every
/// rank (prompt.attnLevers).
pub fn ownersFit(o: Owners) bool {
    if (!o.valid()) return false;
    for (0..4) |h| if (o.rows(h) <= ovl.gemm_min_rows) return false;
    return true;
}

/// The engine's TF_DS_PROMPT_ATTN state. `emulate` (the one-GPU benchmark alone): instead of computing this node's
/// rows and gathering the others', compute every owner's rows one owner after another with the same launches (the
/// gathers' bytes, without NCCL).
pub const State = struct {
    lv: Levers = .{},
    k: ?*const Kernels = null, // tsel's kernel (loaded when tsel is asked for)
    emulate: bool = false,
    /// the benchmark's timing of one node alone: its own rows, no gather (the other owners' rows left as they are)
    no_gather: bool = false,
};

/// attn_prompt.cu's image: the selection kernel and its shared memory.
pub const Kernels = struct {
    module: cuda.Module,
    tsel_fn: cuda.Function,
    smem: u32,

    pub fn load(d: *const cuda.Driver, image: []const u8) !Kernels {
        var m = try cuda.Module.load(d, image);
        errdefer m.unload();
        const g = try m.global("tf_ds_tsel_smem");
        var smem: c_int = 0;
        try d.check(d.api.cuMemcpyDtoH_v2(@ptrCast(&smem), g.ptr, @sizeOf(c_int)), "cuMemcpyDtoH");
        if (smem <= 0 or smem > 99 * 1024) return error.BadImage;
        const f = try m.function("tf_ds_tsel_kernel");
        try f.allowDynamicShared(@intCast(smem));
        return .{ .module = m, .tsel_fn = f, .smem = @intCast(smem) };
    }

    pub fn unload(k: *Kernels) void {
        k.module.unload();
        k.* = undefined;
    }
};

/// The kernel's k (kernels.py index_topk, the engine's only k for prompt chunks).
pub const tsel_k = 512;

/// tsel: rows of keys int64 [rows, ks] (n keys a row) -> out int64 [rows, 512]: each row's 512 largest keys' positions
/// sorted ascending, -1 at or past vis[r]; with tmax (int64 [rows, nt]) from the keys of the tiles holding the 512
/// largest tile maxima (topk_select_pruned), without it from the row (topk_select).
pub fn tsel(k: *const Kernels, s: cuda.Stream, keys: u64, ks: usize, n: usize, tmax: ?u64, nt: usize, vis: u64, out: u64, rows: usize) !void {
    if (rows == 0) return;
    if (n < tsel_k or n > std.math.maxInt(c_int) or nt > std.math.maxInt(c_int)) return error.BadTopK;
    var a: cuda.Args = .{};
    a.add(keys);
    a.add(@as(c_longlong, @intCast(ks)));
    a.add(@as(c_int, @intCast(n)));
    a.add(tmax orelse @as(u64, 0));
    a.add(@as(c_int, @intCast(nt)));
    a.add(vis);
    a.add(out);
    try cuda.launch.launch(k.tsel_fn, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 1024 }, .shared = k.smem }, s, &a);
}

test "TF_DS_PROMPT_ATTN: unset / off is today's; lists and all; bad words fail" {
    try std.testing.expectEqual(Levers{}, try parse(null));
    try std.testing.expectEqual(Levers{}, try parse(""));
    try std.testing.expectEqual(Levers{}, try parse("off"));
    try std.testing.expectEqual(Levers{ .front = true }, try parse("front"));
    try std.testing.expectEqual(Levers{ .isplit = true }, try parse("isplit"));
    try std.testing.expectEqual(Levers{ .front = true, .isplit = true }, try parse("front,isplit"));
    try std.testing.expectEqual(Levers{ .front = true, .isplit = true, .tsel = true }, try parse("all"));
    try std.testing.expectEqual(Levers{ .tsel = true }, try parse("tsel"));
    try std.testing.expect(!(Levers{ .tsel = true }).sp());
    try std.testing.expectError(error.BadPromptAttn, parse("fronts"));
    try std.testing.expectError(error.BadPromptAttn, parse("front,"));
    var buf: [32]u8 = undefined;
    try std.testing.expectEqualStrings("front,isplit", (Levers{ .front = true, .isplit = true }).names(&buf));
}

test "TF_DS_PROMPT_ATTN with TF_DS_PROMPT_OVL (stage ZC2): front's own-first split exists on every rank wherever front acts; the served passes take front, isplit and fxag together" {
    // front (frontSp) replaces xag's x all-gather on its layers with the qr / kv gathers on the comm stream (Kind
    // .xag) and runs wq_b own rows first by ovl.ownFirst: where ownersFit holds, ownFirst must exist on all four ranks
    // (else one rank would fail NotAnSpChunk mid-chunk), with this node's rows first and every row once
    var n: usize = 64;
    while (n <= 4096) : (n += 16) {
        const o = Owners.of(n);
        const fit = ownersFit(o);
        for (0..4) |me| {
            const ps = ovl.ownFirst(o, me);
            if (!fit) continue;
            const p = ps orelse return error.TestUnexpectedResult;
            try std.testing.expectEqual(o.r0(me), p[0].r0);
            try std.testing.expectEqual(o.rows(me), p[0].rows);
            var hit: [4096]u8 = @splat(0);
            for (p) |r| for (r.r0..r.r0 + r.rows) |x| {
                hit[x] += 1;
            };
            for (hit[0..n]) |h| try std.testing.expectEqual(@as(u8, 1), h);
        }
        // the fxag plan (its pairs' rows = the router's runs) only on passes where front / isplit act as well
        if (ovl.fxagFits(o, 2048)) try std.testing.expect(fit);
    }
    // the served 2,048-row chunk and 4,096-row pass: front / isplit act; the pass also takes fxag's split
    try std.testing.expect(ownersFit(Owners.of(2048)));
    try std.testing.expect(ownersFit(Owners.of(4096)));
    try std.testing.expect(ovl.fxagFits(Owners.of(4096), 2048));
    // small chunks keep today's attention (owners of 128 rows or fewer)
    try std.testing.expect(!ownersFit(Owners.of(512)));
    try std.testing.expect(ownersFit(Owners.of(576)));
    // the two env values the combined arms set parse independently (no word of one is a word of the other)
    try std.testing.expectEqual(Levers{ .front = true, .isplit = true }, try parse("front,isplit"));
    const ol = try ovl.parse("all,pack3,dn");
    try std.testing.expect(ol.xag and ol.pack3 and ol.dn and !ol.fxag);
    try std.testing.expectError(error.BadPromptAttn, parse("xag"));
    try std.testing.expectError(error.BadPromptOvl, ovl.parse("front"));
}
