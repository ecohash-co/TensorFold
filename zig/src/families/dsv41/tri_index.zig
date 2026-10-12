//! kernels.py's indexer launches: the compressed keys' scores (a row a program in decode windows, row tiles in prompt
//! chunks), their top-k keys and tile maxima, the pool-only keys and the selection, each as its Python wrapper does.
const std = @import("std");
const cuda = @import("cuda");
const tri = @import("tri.zig");
const aot = cuda.aot;
const Tri = tri.Tri;
const u = tri.u;
const p = aot.ptr;

/// kernels.DECODE_ROWS (TF_DS_DECODE_ROWS unset): calls of at most this many rows take the row-invariant kernels.
pub const decode_rows = 16;
/// The score kernels' key tile: index_score's bn and TOPK_PRUNE_BN (the tile maxima's).
pub const tile = 64;
/// kernels.TOPK_FUSED_MAX (TF_DS_TOPK_FUSED_MAX unset): keys a row the fused selection takes at most.
pub const topk_fused_max = 4096;
/// kernels.IDX_RB with IDX_TREE on (TF_DS_IDX_TREE / TF_DS_IDX_RB unset): prompt chunks' scores one row a program.
const idx_rb = 1;
/// What topk_select_pruned's `every` holds in each row: a visible count past every tile position.
pub const every_vis: i64 = 1 << 40;

fn ci(name: []const u8, v: usize) aot.Const {
    return aot.ci(name, @intCast(v));
}

fn cb(name: []const u8, v: bool) aot.Const {
    return aot.ci(name, @intFromBool(v));
}

fn int(name: []const u8, v: usize) aot.Arg {
    return aot.int(name, @intCast(v));
}

/// kernels._pow2.
fn pow2(n: usize) bool {
    return n > 0 and n & (n - 1) == 0;
}

/// index_score's k: the indexer's keys, a packed FP4 pair (codes u8 [N, ID/2], E8M0 scales u8 [N, ID/32]) or bf16 [N, ID].
pub const IndexK = union(enum) {
    fp4: struct { codes: u64, scales: u64 },
    bf16: u64,

    fn ty(k: IndexK) []const u8 {
        return if (k == .fp4) "*u8" else "*bf16";
    }

    fn codes(k: IndexK) u64 {
        return switch (k) {
            .fp4 => |f| f.codes,
            .bf16 => |a| a,
        };
    }

    fn scales(k: IndexK) u64 {
        return switch (k) {
            .fp4 => |f| f.scales,
            .bf16 => |a| a,
        };
    }
};

/// index_keys' cand: the candidate pool's block mask (bool [R, >= cdiv(n, cand_block)], read as u8) and its row stride.
pub const Cand = struct { mask: u64, stride: usize };

/// index_score without bases on FP4 keys (fp32 scores; int64 keys with tile maxima): beyond aot's fallback (n, nt div
/// 16 onto any-value variants), VIS, KB and CAND 8-aligned onto variants that assumed 16. _index_score's variants of
/// these two configurations (HAS_BASE=0, HAS_CAND=0, PACKED=1; KEYS=0 HAS_TMAX=0, or KEYS=1 HAS_TMAX=1) do not depend
/// on it: they read VIS as one int64 a row (LDC.64 c[0x3a0], + ctaid.x * 8, LDG.E.64) and KB, CAND and cs not at all
/// (no load of c[0x3b8], c[0x3c0], c[0x3c8]), and the captured pairs that differ only in VIS/KB/CAND's alignment have
/// byte-identical SASS (KEYS=1: 9f83460754 and 45c81d5893; KEYS=0: missing-variant.md, stage ZG).
const score_relax: aot.Relax = .{ .ptrs8 = &.{ "VIS", "KB", "CAND" } };

/// The relaxation index_score's _index_score launch takes (score_relax where its SASS says the alignment of VIS, KB
/// and CAND is not read; else none): a prompt row block of at most decode_rows rows puts vis 8 bytes in when it starts
/// at an odd row (rb odd: n_comp_end in 162K..), fp32 scores (layers 20-39) and keyed ones (layers 2-19) alike.
pub fn scoreRelax(k: IndexK, base: ?u64, keys: bool, tmax: ?u64) aot.Relax {
    if (base != null or k != .fp4) return .{};
    return if (!keys or tmax != null) score_relax else .{};
}

/// _index_score's arguments and constexprs as index_score passes them (rows <= decode_rows or rowwise).
pub fn scoreArgs(q: u64, k: IndexK, w: u64, vis: u64, n: usize, out: u64, base: ?u64, keys: bool, tmax: ?u64) [12]aot.Arg {
    const nt = tri.cdiv(n, tile);
    const ot: []const u8 = if (keys) "*i64" else "*fp32";
    // tmax [R, cdiv(n, 64)] int64 (its maxima only with keys); without it the kernel gets out there
    const tm = p("TMAX", if (tmax != null) "*i64" else ot, tmax orelse out);
    // base [R]: each row's first pool row (concurrent rounds); vis stands in for it and for the candidate mask
    return .{
        p("Q", "*bf16", q),    p("K", k.ty(), k.codes()), p("KS", k.ty(), k.scales()), p("Wt", "*bf16", w),
        p("VIS", "*i64", vis), p("OUT", ot, out),         int("n", n),                 p("KB", "*i64", base orelse vis),
        p("CAND", "*i64", vis), int("cs", 0),             tm,                          int("nt", nt),
    };
}

pub fn scoreConsts(k: IndexK, base: ?u64, keys: bool, tmax: ?u64, ih: usize, id: usize, pdl: aot.Const) [12]aot.Const {
    // HAS_CAND, CB and DEAD_PAD at the kernel's defaults (index_score passes none of them)
    // SKIP_DEAD: switch "tile_skip" (served on), a tile wholly at or past vis skips its dots
    return .{
        ci("IH", ih),          ci("ID", id), ci("BN", tile),   cb("PACKED", k == .fp4),               cb("HAS_BASE", base != null), pdl,
        cb("HAS_CAND", false), ci("CB", 8),  cb("KEYS", keys), cb("HAS_TMAX", tmax != null and keys), cb("SKIP_DEAD", true),        cb("DEAD_PAD", false),
    };
}

/// index_score: q [R, IH, ID], k, w [R, IH], vis [R] -> out [R, n] fp32 scores; `keys`: int64 keys; `tmax`: tile maxima.
pub fn indexScore(t: Tri, q: u64, k: IndexK, w: u64, vis: u64, n: usize, out: u64, base: ?u64, keys: bool, tmax: ?u64, rowwise: bool, rows: usize, ih: usize, id: usize) !void {
    if (rows <= tri.decode_rows or rowwise) {
        const nt = tri.cdiv(n, tile);
        const args = scoreArgs(q, k, w, vis, n, out, base, keys, tmax);
        const consts = scoreConsts(k, base, keys, tmax, ih, id, t.pdlConst());
        return t.runRelaxed("_index_score", .{ u(rows), nt, 1 }, &args, &consts, scoreRelax(k, base, keys, tmax));
    }
    std.debug.assert(base == null);
    return indexScoreRows(t, q, k, w, vis, n, out, keys, tmax, rows, ih, id);
}

/// index_score's prompt-chunk launch (_index_score_rows) whatever the row count: rows of a block of more than
/// decode_rows rows (TF_DS_PROMPT_ATTN=isplit: a block's rows one node owns; each row's scores are its own, one row a
/// program). rows >= 2 (the kit has no rows == 1 specialization).
pub fn indexScoreRows(t: Tri, q: u64, k: IndexK, w: u64, vis: u64, n: usize, out: u64, keys: bool, tmax: ?u64, rows: usize, ih: usize, id: usize) !void {
    const nt = tri.cdiv(n, tile);
    const ot: []const u8 = if (keys) "*i64" else "*fp32";
    const tm = p("TMAX", if (tmax != null) "*i64" else ot, tmax orelse out);
    try t.run("_index_score_rows", .{ tri.cdiv(rows, idx_rb), nt, 1 }, &.{
        p("Q", "*bf16", q),    p("K", k.ty(), k.codes()), p("KS", k.ty(), k.scales()), p("Wt", "*bf16", w),
        p("VIS", "*i64", vis), p("OUT", ot, out),         int("n", n),                 int("rows", rows),
        tm,                    int("nt", nt),
    }, &.{
        ci("IH", ih),     ci("ID", id),                          ci("BN", tile),   ci("RB", idx_rb), cb("PACKED", k == .fp4),
        cb("KEYS", keys), cb("HAS_TMAX", tmax != null and keys), cb("TREE", true),
    });
}

/// index_keys: index_score's int64 keys (decode windows), apply_candidates' mask folded in; pruned_k: topk_select_pruned's k.
pub fn indexKeys(t: Tri, q: u64, k: IndexK, w: u64, vis: u64, n: usize, base: ?u64, cand: ?Cand, cand_block: usize, tmax: ?u64, pruned_k: usize, out: u64, rows: usize, ih: usize, id: usize) !void {
    std.debug.assert(rows <= tri.decode_rows);
    const nt = tri.cdiv(n, tile);
    // switch "tile_skip" (served on): dead tiles skip their dots, and write no keys when only the pruned selection reads
    const skip = true;
    // the mask as u8 (cand.view(torch.uint8)), its row stride; without one vis stands in, stride 0
    const cv = if (cand) |m| p("CAND", "*u8", m.mask) else p("CAND", "*i64", vis);
    const cs: usize = if (cand) |m| m.stride else 0;
    try t.run("_index_score", .{ u(rows), nt, 1 }, &.{
        p("Q", "*bf16", q),    p("K", k.ty(), k.codes()), p("KS", k.ty(), k.scales()),        p("Wt", "*bf16", w),
        p("VIS", "*i64", vis), p("OUT", "*i64", out),     int("n", n),                        p("KB", "*i64", base orelse vis),
        cv,                    int("cs", cs),             p("TMAX", "*i64", tmax orelse out), int("nt", nt),
    }, &.{
        ci("IH", ih),                 ci("ID", id),         ci("BN", tile),   cb("PACKED", k == .fp4),      cb("HAS_BASE", base != null), t.pdlConst(),
        cb("HAS_CAND", cand != null), ci("CB", cand_block), cb("KEYS", true), cb("HAS_TMAX", tmax != null), cb("SKIP_DEAD", skip),        cb("DEAD_PAD", skip and tmax != null and prunes(n, pruned_k)),
    });
}

/// index_keys_cand: the keys of the pool's blocks only, cblk [R, nblk] i32 (-1: none; row stride cbs) -> out [R, nblk * CB].
pub fn indexKeysCand(t: Tri, q: u64, k: IndexK, w: u64, vis: u64, n: usize, cblk: u64, cbs: usize, nblk: usize, cand_block: usize, base: ?u64, out: u64, rows: usize, ih: usize, id: usize) !void {
    std.debug.assert(rows <= tri.decode_rows);
    const nout = nblk * cand_block;
    try t.run("_index_score_cand", .{ u(rows), tri.cdiv(nout, tile), 1 }, &.{
        p("Q", "*bf16", q),               p("K", k.ty(), k.codes()), p("KS", k.ty(), k.scales()), p("Wt", "*bf16", w),
        p("VIS", "*i64", vis),            p("OUT", "*i64", out),     int("n", n),                 int("nout", nout),
        p("KB", "*i64", base orelse vis), p("CBLK", "*i32", cblk),   int("cbs", cbs),
    }, &.{ ci("IH", ih), ci("ID", id), ci("BN", tile), cb("PACKED", k == .fp4), cb("HAS_BASE", base != null), ci("CB", cand_block), t.pdlConst() });
}

/// torch's top-k, topk_select's one step that is no Triton launch: keys.topk(k, sorted=False).values into top [R, k] int64.
pub const TopK = struct {
    top: u64,
    ctx: ?*anyopaque = null,
    run: *const fn (ctx: ?*anyopaque, t: Tri, keys: u64, ks: usize, rows: usize, n: usize, k: usize, top: u64) anyerror!void,
};

/// topk_select: keys int64 [R, n] (row stride ks) -> out [R, k] int64, the k largest keys' indices sorted, -1 past vis[r].
pub fn topkSelect(t: Tri, keys: u64, ks: usize, k: usize, vis: u64, out: u64, top: TopK, rows: usize, n: usize) !void {
    // switch "topk_fused" (served on): one launch a row for power-of-two k <= n <= TOPK_FUSED_MAX, contiguous int64 keys
    if (fusedSelect(n, k, ks, rows)) {
        return t.run("_topk_sel", .{ u(rows), 1, 1 }, &.{ p("KEYS", "*i64", keys), p("VIS", "*i64", vis), p("OUT", "*i64", out) }, &.{ ci("N", n), ci("KK", k), t.pdlConst() });
    }
    try top.run(top.ctx, t, keys, ks, rows, n, k, top.top);
    try t.run("_topk_finish", .{ u(rows), 1, 1 }, &.{ p("TOP", "*i64", top.top), p("VIS", "*i64", vis), p("OUT", "*i64", out) }, &.{ ci("KK", k), t.pdlConst() });
}

/// topk_select takes the fused selection (_topk_sel, switch "topk_fused") for these keys (else torch's top-k).
pub fn fusedSelect(n: usize, k: usize, ks: usize, rows: usize) bool {
    return pow2(n) and pow2(k) and k <= n and n <= topk_fused_max and (ks == n or rows <= 1);
}

/// prunes: topk_select_pruned searches tiles for rows of n keys (more than TOPK_FUSED_MAX keys and more than k tiles).
pub fn prunes(n: usize, k: usize) bool {
    // switch "topk_prune" (served on)
    return k > 0 and n > topk_fused_max and tri.cdiv(n, tile) > k;
}

/// topk_select_pruned's buffers: tpos [R, k], cand [R, k * 64], every [>= R] all every_vis (constant: filled once).
pub const PrunedBufs = struct { tpos: u64, cand: u64, every: u64 };

/// topk_select_pruned: topk_select(keys, k, vis) from the k 64-key tiles whose maxima tmax [R, cdiv(n, 64)] are largest.
pub fn topkSelectPruned(t: Tri, keys: u64, ks: usize, tmax: u64, k: usize, vis: u64, out: u64, bufs: PrunedBufs, top: TopK, rows: usize, n: usize) !void {
    // kernels.py also falls back to topk_select when keys.stride(1) != 1: every caller's keys have unit column stride
    if (!prunes(n, k)) return topkSelect(t, keys, ks, k, vis, out, top, rows, n);
    const nt = tri.cdiv(n, tile);
    try topkSelect(t, tmax, nt, k, bufs.every, bufs.tpos, top, rows, nt);
    try t.run("_gather_tiles", .{ u(rows), u(k), 1 }, &.{
        p("KEYS", "*i64", keys), p("TPOS", "*i64", bufs.tpos), p("OUT", "*i64", bufs.cand), int("n", n), int("ks", ks),
    }, &.{ ci("BN", tile), ci("KK", k), t.pdlConst() });
    try topkSelect(t, bufs.cand, k * tile, k, vis, out, top, rows, k * tile);
}

/// score_keys: fp32 scores [R, n] (contiguous) -> out [R, n] int64 keys; tmax: each 64-key tile's largest [R, cdiv(n, 64)].
pub fn scoreKeys(t: Tri, score: u64, tmax: ?u64, out: u64, rows: usize, n: usize) !void {
    try t.run("_score_keys", .{ u(rows), tri.cdiv(n, 1024), 1 }, &.{
        p("S", "*fp32", score), p("OUT", "*i64", out), int("n", n), p("TMAX", "*i64", tmax orelse out), int("nt", tri.cdiv(n, tile)),
    }, &.{ ci("BN", 1024), cb("HAS_TMAX", tmax != null), ci("TB", tile), t.pdlConst() });
}

// -- conformance with the recorded launches ----------------------------------------------------------------------

const Case = tri.Case;

/// config.json's index_topk: rounds.py's k (and index_keys' pruned_k) is min(index_topk, n).
const index_topk = 512;

/// The recorded key cache: FP4 codes [N, ID/2] with their scales, or bf16 rows [N, ID].
fn caseK(c: Case) IndexK {
    if (2 * c.dim("K", 1) == c.dim("Q", 2)) return .{ .fp4 = .{ .codes = c.ptr("K"), .scales = c.ptr("KS") } };
    return .{ .bf16 = c.ptr("K") };
}

/// A launch of a decode round (rounds.py passes the pool's per-row bases), not of the prompt forward (attention_k: none).
fn inRound(c: Case) bool {
    return std.mem.startsWith(u8, c.v.get("phase").?.string, "round:");
}

/// A recorded int argument, passed or folded by Triton (the value 1) into a constexpr.
fn num(c: Case, name: []const u8) usize {
    return @intCast(if (c.hasConst(name)) c.constInt(name) else c.int(name));
}

/// The tests' torch top-k: nothing (no Triton launch to check).
fn noTopK(_: ?*anyopaque, _: Tri, _: u64, _: usize, _: usize, _: usize, _: usize, _: u64) anyerror!void {}

test "index_score and index_keys launch _index_score as recorded" {
    try tri.conform("_index_score", struct {
        fn call(t: Tri, c: Case) !void {
            const rows = c.dim("Q", 0);
            const n = c.dim("OUT", 1);
            const keys = c.constInt("KEYS") != 0;
            const tmax: ?u64 = if (c.constInt("HAS_TMAX") != 0) c.ptr("TMAX") else null;
            const base: ?u64 = if (inRound(c)) c.ptr("KB") else null;
            if (keys and inRound(c)) {
                // rounds.py's index_keys: a pool mask is [R, blocks] (vis [R] stands in without one)
                const cand: ?Cand = if (c.rank("CAND") == 2) .{ .mask = c.ptr("CAND"), .stride = c.stride("CAND", 0) } else null;
                return indexKeys(t, c.ptr("Q"), caseK(c), c.ptr("Wt"), c.ptr("VIS"), n, base, cand, @intCast(c.constInt("CB")), tmax, @min(index_topk, n), c.ptr("OUT"), rows, c.dim("Q", 1), c.dim("Q", 2));
            }
            try indexScore(t, c.ptr("Q"), caseK(c), c.ptr("Wt"), c.ptr("VIS"), n, c.ptr("OUT"), base, keys, tmax, false, rows, c.dim("Q", 1), c.dim("Q", 2));
        }
    }.call);
}

/// Stage ZS2's refused launch (helen's log, results/stageZS2/hang-log-helen.txt): a keyed prompt row block's tail of at
/// most decode_rows rows at an odd row (rb 207 at n_comp_end 80809), so vis 8 bytes off 16; n and nt not div 16.
fn zs2(kernels: []const aot.KernelJson, vis: u64, n: usize) !?struct { hash: []const u8, exact: bool, night4: bool } {
    const k: IndexK = .{ .fp4 = .{ .codes = 0xe3dd526d0000, .scales = 0xe3dd2b201300 } };
    const tmax: ?u64 = 0xe3e2b5c48200;
    const args = scoreArgs(0xe3e2b4a0e200, k, 0xe3e2b5c1d3c0, vis, n, 0xe3e36f800000, null, true, tmax);
    const consts = scoreConsts(k, null, true, tmax, 32, 128, aot.ci("PDL", 1));
    const night4 = aot.pick(kernels, "_index_score", &args, &consts, .{}, false) != null; // night4: no relaxation when keyed
    const got = aot.pick(kernels, "_index_score", &args, &consts, scoreRelax(k, null, true, tmax), false) orelse return null;
    return .{ .hash = kernels[got.index].hash, .exact = got.exact, .night4 = night4 };
}

fn expectZs2(kernels: []const aot.KernelJson) !void {
    const expect = std.testing.expect;
    const vis: u64 = 0xe3e2b5c43c38;
    // the logged launch: night4 refused it (MissingTritonVariant); now fa1370b0f6 (n, nt any, VIS/KB/CAND 16) runs it
    const a = (try zs2(kernels, vis, 80809)).?;
    try expect(!a.night4 and !a.exact and std.mem.startsWith(u8, a.hash, "fa1370b0f6"));
    // the same block 16-aligned: that variant exactly, as before
    const b = (try zs2(kernels, vis + 8, 80809)).?;
    try expect(b.night4 and b.exact and std.mem.startsWith(u8, b.hash, "fa1370b0f6"));
    // n and nt div 16 with vis 8 in: the captured 45c81d5893 exactly (no relaxation needed or taken)
    const c = (try zs2(kernels, vis, 1024 * 64)).?;
    try expect(c.night4 and c.exact and std.mem.startsWith(u8, c.hash, "45c81d5893"));
    // 4-aligned never
    try expect(try zs2(kernels, vis + 4, 80809) == null);
}

test "stage ZS2's refused _index_score launch (keyed, VIS 8 bytes in, n 80809, nt 1263) runs the captured fa1370b0f6" {
    const gpa = std.testing.allocator;
    // the kit's KEYS=1 HAS_BASE=0 HAS_TMAX=1 variants (aot.json entries, cut to the fields a pick reads)
    const set = try aot.parseSet(gpa, @embedFile("fixtures/index_score_keyed_variants.json"));
    defer set.deinit();
    try expectZs2(set.value.kernels);
    // the args as logged
    const k: IndexK = .{ .fp4 = .{ .codes = 1, .scales = 2 } };
    const args = scoreArgs(0, k, 0, 0xe3e2b5c43c38, 80809, 0, null, true, 3);
    try std.testing.expectEqual(@as(i32, 1263), args[11].value.i32);
    try std.testing.expectEqual(args[4].value.ptr.addr, args[7].value.ptr.addr); // KB, CAND: vis
    try std.testing.expectEqual(args[4].value.ptr.addr, args[8].value.ptr.addr);
    // the relaxation only where the SASS was read: never with bases, bf16 keys, or keys without tile maxima
    try std.testing.expectEqual(@as(usize, 3), scoreRelax(k, null, true, 3).ptrs8.len);
    try std.testing.expectEqual(@as(usize, 3), scoreRelax(k, null, false, null).ptrs8.len);
    try std.testing.expectEqual(@as(usize, 0), scoreRelax(k, 5, true, 3).ptrs8.len);
    try std.testing.expectEqual(@as(usize, 0), scoreRelax(k, null, true, null).ptrs8.len);
    try std.testing.expectEqual(@as(usize, 0), scoreRelax(.{ .bf16 = 1 }, null, false, null).ptrs8.len);
    // with the whole captured kit when it is on this host (TF_TEST_AOT_JSON; skipped when unset)
    const path: []const u8 = if (std.c.getenv("TF_TEST_AOT_JSON")) |v| std.mem.span(v) else return;
    const text = std.Io.Dir.cwd().readFileAlloc(std.testing.io, path, gpa, .limited(1 << 24)) catch return;
    defer gpa.free(text);
    const kit = try aot.parseSet(gpa, text);
    defer kit.deinit();
    try expectZs2(kit.value.kernels);
}

test "every keyed prompt row block of at most decode_rows rows finds a variant, at every prompt length to 1M" {
    // prompt.selectRows(Part)'s blocks: the passes' 2,048-row runs [s0, s0 + sn) (s0 % 2048 == 0), the keyed layers'
    // n_comp_end (s0 + sn) / 2, blocks of rowBlock rows; a block of at most decode_rows rows past the first starts at
    // row s0 + k rb, its vis k rb int64s into the run. Only a prompt's last, partial run can leave one (whole runs:
    // n_comp_end % 1024 == 0, n and nt div 16, the captured 45c81d5893 / 9f83460754 exactly). night4 refused 2,294
    // prompt lengths up to 200K this way (18,226 the first; stage ZS2's 161,618-161,619), night3 3,085.
    const gpa = std.testing.allocator;
    const set = try aot.parseSet(gpa, @embedFile("fixtures/index_score_keyed_variants.json"));
    defer set.deinit();
    const rowBlock = @import("prompt.zig").rowBlock;
    const k: IndexK = .{ .fp4 = .{ .codes = 0x10000, .scales = 0x20000 } };
    const tmax: ?u64 = 0x30000;
    var tails: usize = 0;
    var night4: usize = 0;
    var len: usize = 1;
    while (len <= 1 << 20) : (len += 1) {
        const sn = len % 2048;
        if (sn == 0) continue;
        const nce = len / 2;
        const rb = rowBlock(sn, nce);
        const blocks = sn / rb;
        const m = sn - blocks * rb;
        if (blocks == 0 or m == 0 or m > decode_rows) continue;
        tails += 1;
        const vis: u64 = 0x40000 + blocks * rb * 8;
        const args = scoreArgs(0x50000, k, 0x60000, vis, nce, 0x70000, null, true, tmax);
        const consts = scoreConsts(k, null, true, tmax, 32, 128, aot.ci("PDL", 1));
        if (aot.pick(set.value.kernels, "_index_score", &args, &consts, .{}, false) == null) night4 += 1;
        if (aot.pick(set.value.kernels, "_index_score", &args, &consts, scoreRelax(k, null, true, tmax), false) == null) {
            std.debug.print("no variant at prompt length {d}\n", .{len});
            return error.TestUnexpectedResult;
        }
    }
    try std.testing.expect(tails > 0 and night4 > 0);
}

test "index_score (prompt chunks) launches _index_score_rows as recorded" {
    try tri.conform("_index_score_rows", struct {
        fn call(t: Tri, c: Case) !void {
            const tmax: ?u64 = if (c.constInt("HAS_TMAX") != 0) c.ptr("TMAX") else null;
            try indexScore(t, c.ptr("Q"), caseK(c), c.ptr("Wt"), c.ptr("VIS"), c.dim("OUT", 1), c.ptr("OUT"), null, c.constInt("KEYS") != 0, tmax, false, c.dim("Q", 0), c.dim("Q", 1), c.dim("Q", 2));
        }
    }.call);
}

test "index_keys_cand launches _index_score_cand as recorded" {
    try tri.conform("_index_score_cand", struct {
        fn call(t: Tri, c: Case) !void {
            const base: ?u64 = if (inRound(c)) c.ptr("KB") else null;
            try indexKeysCand(t, c.ptr("Q"), caseK(c), c.ptr("Wt"), c.ptr("VIS"), num(c, "n"), c.ptr("CBLK"), c.stride("CBLK", 0), c.dim("CBLK", 1), @intCast(c.constInt("CB")), base, c.ptr("OUT"), c.dim("Q", 0), c.dim("Q", 1), c.dim("Q", 2));
        }
    }.call);
}

test "topk_select (fused) launches _topk_sel as recorded" {
    try tri.conform("_topk_sel", struct {
        fn call(t: Tri, c: Case) !void {
            const top: TopK = .{ .top = 0x7e00_0000_0000, .run = noTopK };
            try topkSelect(t, c.ptr("KEYS"), c.stride("KEYS", 0), c.dim("OUT", 1), c.ptr("VIS"), c.ptr("OUT"), top, c.dim("KEYS", 0), c.dim("KEYS", 1));
        }
    }.call);
}

test "topk_select (torch's top-k) launches _topk_finish as recorded" {
    try tri.conform("_topk_finish", struct {
        fn call(t: Tri, c: Case) !void {
            // the keys are not in the launch (torch's top-k read them): a width past the fused selection's stands in
            const n = 2 * topk_fused_max;
            const top: TopK = .{ .top = c.ptr("TOP"), .run = noTopK };
            try topkSelect(t, 0x7e00_0000_0000, n, c.dim("TOP", 1), c.ptr("VIS"), c.ptr("OUT"), top, c.dim("TOP", 0), n);
        }
    }.call);
}

test "topk_select_pruned launches _gather_tiles as recorded" {
    try tri.conform("_gather_tiles", struct {
        fn call(t: Tri, c: Case) !void {
            const bufs: PrunedBufs = .{ .tpos = c.ptr("TPOS"), .cand = c.ptr("OUT"), .every = 0x7e00_0000_0000 };
            const top: TopK = .{ .top = 0x7e00_0000_0100, .run = noTopK };
            try topkSelectPruned(t, c.ptr("KEYS"), c.stride("KEYS", 0), 0x7e00_0000_0200, c.dim("TPOS", 1), 0x7e00_0000_0300, 0x7e00_0000_0400, bufs, top, c.dim("KEYS", 0), c.dim("KEYS", 1));
        }
    }.call);
}

test "score_keys launches _score_keys as recorded" {
    try tri.conform("_score_keys", struct {
        fn call(t: Tri, c: Case) !void {
            const tmax: ?u64 = if (c.constInt("HAS_TMAX") != 0) c.ptr("TMAX") else null;
            try scoreKeys(t, c.ptr("S"), tmax, c.ptr("OUT"), c.dim("S", 0), c.dim("S", 1));
        }
    }.call);
}
