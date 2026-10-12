//! Stage MX3 (milestone 1: kernels, slice loader and tf-dsv41-mx-bench; no layer calls this yet): a 2D node's prompt-chunk
//! experts on the original MXFP4 weights (TIER-2 lever "MXFP4 prefill copy of the encoder experts", stageMX0/README.md)
//! in place of the EXL3 chain, for layers that hold a converted slice (convert_mxfp4_slices.py, one safetensors file a
//! (layer, node)): TensorFold's O2 prefill kernel (zig/kernels/cuda/dsv41/mx_prompt.cu, upstream's template as it is)
//! for the routed experts, the shared expert on its exact bf16 cuts through cuBLAS, a plain slot combine.
//!
//! The chain on a chunk of n rows, P = n * slots pairs (slots = top_k + 1, the shared expert in the last slot):
//!   route        picks -> the routed plan (E + 1 experts: the shared id is in it and skipped by the GEMMs) and, through
//!                ops.remapPicks with the slice's rest map, the rest plan (the rest experts + 1: every expert another
//!                pair computes the rest of, and the shared expert, map to the last id, skipped)
//!   gate main    O2 EPI 2 on x (the plain ffn-normed bf16 rows: no rotation here), K 5120 x N 512: the pack's main rows
//!   gate rest    O2 EPI 2, N 128, the parity experts' rows of the pack's rest part (other rows untouched)
//!   shared g/u   cuBLAS x [n, D] x [gate; up]^T -> fp32 [n, 2 sw], experts.cuh's swiglu -> the pack's slot rows
//!   (exchange)   the pack (bf16, the EXL3 pack's bytes a row: main [P, 512] then rest [P, 128]) as today
//!                (Two.assembleParity: packExchange + assembleRest with the same owner table) -> xd_full [P, 1152]
//!   down         O2 EPI 0, K 1152 x N 2560 -> y fp32 [P, 2560] (the shared slot's rows untouched)
//!   shared down  cuBLAS xd_full's slot rows x W_d^T -> y's slot rows (row strides slots x 1152 / slots x 2560)
//!   combine      out[r] = sum over the slots in order of wts[r, s] y[r slots + s] (an fp32 fmaf chain from 0)
//! Every routed pair's bits are the Python kernel's (bench_mx_zigsplit.py, stage MX2): the same template, the same
//! flags, a pair's bits independent of its neighbours and of the geometry.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const plan_mod = @import("plan.zig");
const Config = @import("config.zig").Config;
const cublas = @import("cublas.zig");
const ops_mod = @import("ops.zig");

const dio = core.direct_io;

/// int32 words of a (32 columns, 32 inputs) "mx" block, and pairs an O2 item holds.
pub const block_words = 136;
pub const tile = 64;
/// tf_mx_plan_small's limit (pairs) and the wide plan's block (PLAN_THREADS).
pub const plan_small = 1024;
pub const plan_threads = 1024;
/// The plan kernels' expert limit (EMAX).
pub const plan_emax = 1024;

/// O2 geometries built (upstream TF_GEOS ids): 2 gate / up's default at NB 16, 3 six column-block warps, 4 down's default.
pub const geos = [_]u32{ 2, 3, 4 };
pub const Geos = struct { main: u32 = 2, rest: u32 = 2, down: u32 = 4 };

pub fn geoIndex(g: u32) !usize {
    for (geos, 0..) |x, i| if (x == g) return i;
    return error.NoSuchGeometry;
}

/// The (RT, WM, WN, CN, SG, stages) of geometry g, upstream's table.
pub fn geoDesc(g: u32) ![6]u32 {
    return switch (g) {
        2 => .{ 2, 2, 4, 1, 2, 2 },
        3 => .{ 2, 2, 6, 1, 2, 2 },
        4 => .{ 2, 2, 4, 2, 2, 2 },
        else => error.NoSuchGeometry,
    };
}

/// tf_mx::nvfp4_prefill_kernel<FMT 1, M, EPI, geometry>'s mangled name (mx_prompt.cu's explicit instantiations).
fn symbol(comptime m: u32, comptime epi: u32, comptime g: u32) [:0]const u8 {
    const d = geoDesc(g) catch unreachable;
    return std.fmt.comptimePrint("_ZN5tf_mx20nvfp4_prefill_kernelILi1ELi{d}ELi{d}ELi{d}ELi{d}ELi{d}ELi{d}ELi{d}ELi{d}EEEvPK13__nv_bfloat16iiPK5uint4PKfiiPKiSA_SA_Pvifi", .{ m, epi, d[0], d[1], d[2], d[3], d[4], d[5] });
}

/// The entries in tf_mx_entry's order: gate / up (M 2, EPI 2) at geometries 2, 3, 4, then down (M 1, EPI 0).
pub const entry_symbols = [6][:0]const u8{ symbol(2, 2, 2), symbol(2, 2, 3), symbol(2, 2, 4), symbol(1, 0, 2), symbol(1, 0, 3), symbol(1, 0, 4) };

/// An entry's launch shape (tf_mx_entry): threads, dynamic shared bytes, column blocks a CTA, pairs an item.
pub const Shape = struct { threads: u32, smem: u32, cb: u32, bm: u32 };

/// The shape mx_prompt.cu's Tile gives (for the host tests: the image's table must say the same).
pub fn tileShape(m: u32, g: u32) !Shape {
    const d = try geoDesc(g);
    const rt = d[0];
    const wm = d[1];
    const wn = d[2];
    const cn = d[3];
    const sg = d[4];
    const st = d[5];
    const bm = 16 * rt * wm;
    const cb = wn * cn;
    const xu = bm * sg * 4;
    const wu = cb * sg * m * (block_words / 4);
    return .{ .threads = wm * wn * 32, .smem = st * (xu + wu) * 16, .cb = cb, .bm = bm };
}

pub const Kernels = struct {
    module: cuda.Module,
    gu: [3]cuda.Function, // by geoIndex
    dn: [3]cuda.Function,
    shape: [6]Shape, // gu then dn
    plan_small: cuda.Function,
    plan_rank: cuda.Function,
    plan_offsets: cuda.Function,
    plan_scatter: cuda.Function,
    combine: cuda.Function,
    shared_swiglu: cuda.Function,

    pub fn load(d: *const cuda.Driver, image: []const u8) !Kernels {
        var m = try cuda.Module.load(d, image);
        errdefer m.unload();
        const gf = try m.global("tf_mx_format");
        var fmt: [2]c_int = undefined;
        try d.check(d.api.cuMemcpyDtoH_v2(@ptrCast(&fmt), gf.ptr, @sizeOf([2]c_int)), "cuMemcpyDtoH");
        if (fmt[0] != block_words or fmt[1] != tile) return error.BadImage;
        const ge = try m.global("tf_mx_entry");
        var tab: [6][4]c_int = undefined;
        try d.check(d.api.cuMemcpyDtoH_v2(@ptrCast(&tab), ge.ptr, @sizeOf([6][4]c_int)), "cuMemcpyDtoH");
        var k: Kernels = undefined;
        k.module = m;
        for (0..6) |i| {
            k.shape[i] = .{ .threads = @intCast(tab[i][0]), .smem = @intCast(tab[i][1]), .cb = @intCast(tab[i][2]), .bm = @intCast(tab[i][3]) };
            const want = try tileShape(if (i < 3) 2 else 1, geos[i % 3]);
            if (!std.meta.eql(k.shape[i], want) or want.bm != tile) return error.BadImage;
            const f = try m.function(entry_symbols[i]);
            try f.allowDynamicShared(k.shape[i].smem);
            if (i < 3) k.gu[i] = f else k.dn[i - 3] = f;
        }
        k.plan_small = try m.function("tf_mx_plan_small");
        k.plan_rank = try m.function("tf_mx_plan_rank");
        k.plan_offsets = try m.function("tf_mx_plan_offsets");
        k.plan_scatter = try m.function("tf_mx_plan_scatter");
        k.combine = try m.function("tf_mx_combine");
        k.shared_swiglu = try m.function("tf_mx_shared_swiglu");
        return k;
    }

    pub fn unload(k: *Kernels) void {
        k.module.unload();
        k.* = undefined;
    }
};

// -- the grouping ---------------------------------------------------------------------------------------------------

/// grouped.max_items: items a plan of `pairs` over `experts` holds at most (an item a used expert, plus one a `t` pairs).
pub fn maxItems(pairs: usize, experts: usize, t: usize) usize {
    return @min(pairs, experts) + pairs / t;
}

/// A plan's device scratch (grouped.Plan): members int32 [pairs], items int32 [maxItems, 3], counts int32 [2], rank
/// int32 [pairs] and hist int32 [ceil(pairs / 1024), experts] (the wide plan's).
pub const PlanScratch = struct {
    pairs: usize, // the most pairs it holds
    experts: usize,
    members: u64,
    items: u64,
    counts: u64,
    rank: u64,
    hist: u64,

    pub fn sizes(pairs: usize, experts: usize) [5]usize {
        return .{ 4 * @max(pairs, 1), 12 * maxItems(@max(pairs, 1), experts, tile), 8, 4 * @max(pairs, 1), 4 * cdiv(@max(pairs, 1), plan_threads) * experts };
    }
};

fn cdiv(a: usize, b: usize) usize {
    return (a + b - 1) / b;
}

fn i32a(a: *cuda.Args, v: usize) void {
    a.add(@as(c_int, @intCast(v)));
}

fn go(f: cuda.Function, s: cuda.Stream, grid: usize, block: usize, shared: u32, a: *cuda.Args) !void {
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(grid) }, .block = .{ .x = @intCast(block) }, .shared = shared }, s, a);
}

/// grouped.route (experts_plan_cuda): picks int32 [p] (each < ps.experts) -> ps's members, items and counts.
pub fn route(k: *const Kernels, s: cuda.Stream, picks: u64, p: usize, ps: PlanScratch) !void {
    if (p == 0) return;
    if (p > ps.pairs or ps.experts > plan_emax) return error.PlanTooSmall;
    var a: cuda.Args = .{};
    if (p <= plan_small) {
        a.add(picks);
        i32a(&a, p);
        i32a(&a, ps.experts);
        i32a(&a, tile);
        for ([_]u64{ ps.members, ps.items, ps.counts }) |v| a.add(v);
        return go(k.plan_small, s, 1, plan_threads, 0, &a);
    }
    const nblk = cdiv(p, plan_threads);
    a.add(picks);
    i32a(&a, p);
    i32a(&a, ps.experts);
    a.add(ps.rank);
    a.add(ps.hist);
    try go(k.plan_rank, s, nblk, plan_threads, 0, &a);
    a = .{};
    i32a(&a, nblk);
    i32a(&a, ps.experts);
    i32a(&a, tile);
    for ([_]u64{ ps.hist, ps.items, ps.counts }) |v| a.add(v);
    try go(k.plan_offsets, s, 1, plan_threads, 0, &a);
    a = .{};
    a.add(picks);
    i32a(&a, p);
    i32a(&a, ps.experts);
    a.add(ps.rank);
    a.add(ps.hist);
    a.add(ps.members);
    try go(k.plan_scatter, s, cdiv(p, 256), 256, 0, &a);
}

pub const Epi = enum { gate_up, down };

/// One O2 launch (upstream launch()): pair q reads x row q / slots (slots > 0) or q, writes out row q of n columns
/// (bf16 SwiGLU for gate_up, fp32 for down); items of expert `skip` (-1: none) return at once.
pub const Gemm = struct {
    epi: Epi,
    geo: u32,
    x: u64,
    x_stride: usize, // elements (a multiple of 8: 16-byte rows)
    slots: usize, // 0: pair q reads row q
    w: u64,
    scale: u64,
    kg: usize, // K / 32
    nb: usize, // N / 32
    out: u64,
    limit: f32,
    skip: i32,
    pairs: usize, // the plan's pairs (the grid's max items)
};

pub fn gemm(k: *const Kernels, s: cuda.Stream, g: Gemm, ps: PlanScratch) !void {
    if (g.pairs == 0) return;
    if (g.x_stride % 8 != 0 or g.x % 16 != 0) return error.UnalignedRows;
    const gi = try geoIndex(g.geo);
    const f = if (g.epi == .gate_up) k.gu[gi] else k.dn[gi];
    const sh = k.shape[if (g.epi == .gate_up) gi else 3 + gi];
    const items = maxItems(g.pairs, ps.experts, tile);
    const grid = items * (cdiv(g.nb, sh.cb));
    if (grid >= (@as(usize, 1) << 31)) return error.GridTooLarge;
    var a: cuda.Args = .{};
    a.add(g.x);
    i32a(&a, g.x_stride);
    i32a(&a, g.slots);
    a.add(g.w);
    a.add(g.scale);
    i32a(&a, g.kg);
    i32a(&a, g.nb);
    for ([_]u64{ ps.items, ps.counts, ps.members, g.out }) |v| a.add(v);
    i32a(&a, g.nb * 32);
    a.add(g.limit);
    a.add(@as(c_int, g.skip));
    try go(f, s, grid, sh.threads, sh.smem, &a);
}

// -- a layer's slice --------------------------------------------------------------------------------------------------

/// What a node's slice of a layer must hold (plan.zig's 2D parity split for that node, the converter's cuts).
pub const Expect = struct {
    layer: u32,
    node: u32, // g = 2 p + r
    tp2_rank: u32,
    pair: u32,
    half: [2]usize,
    main: [2]usize,
    rest: [2]usize,
    dcols: [2]usize,
    hidden: usize,
    experts: usize,
    n_rest: usize, // experts whose rest this pair computes (e % 2 == pair)
    shared_rest: bool, // the shared expert's rest is this pair's (plan.restPair)

    pub fn mainW(x: Expect) usize {
        return x.main[1] - x.main[0];
    }
    pub fn restW(x: Expect) usize {
        return x.rest[1] - x.rest[0];
    }
    pub fn halfW(x: Expect) usize {
        return x.half[1] - x.half[0];
    }
    pub fn downN(x: Expect) usize {
        return x.dcols[1] - x.dcols[0];
    }
    /// The shared expert's gate / up rows here: the main columns, and the rest's on the pair that computes it.
    pub fn sharedW(x: Expect) usize {
        return x.mainW() + if (x.shared_rest) x.restW() else 0;
    }

    /// From the split the engine runs (TF_DS_2D_GU=parity on four nodes).
    pub fn of(c: Config, s: plan_mod.Split, layer: u32) !Expect {
        const p = s.pair orelse return error.NotA2DSplit;
        if (!s.parity) return error.MxNeedsTheParitySplit;
        const xp = s.expertParts(c);
        const rest = xp.rest orelse return error.MxNeedsTheParitySplit;
        const dcols = xp.dcols orelse return error.NotA2DSplit;
        var n_rest: usize = 0;
        for (0..c.experts) |e| n_rest += @intFromBool(plan_mod.restPair(e, c.experts) == p);
        return .{ .layer = layer, .node = s.rank + 2 * p, .tp2_rank = s.rank, .pair = p, .half = xp.half, .main = xp.gu, .rest = rest, .dcols = dcols, .hidden = c.hidden, .experts = c.experts, .n_rest = n_rest, .shared_rest = plan_mod.restPair(c.experts, c.experts) == p };
    }
};

pub const max_rank = 5;

/// A slice file's tensor (its own header parser: the blocks are rank 5, past core.safetensors' 4).
pub const Entry = struct {
    dtype: []const u8,
    rank: usize,
    shape: [max_rank]usize,
    begin: usize, // from the data region's start
    end: usize,

    pub fn is(e: Entry, dtype: []const u8, shape: []const usize) bool {
        return std.mem.eql(u8, e.dtype, dtype) and e.rank == shape.len and std.mem.eql(usize, e.shape[0..e.rank], shape);
    }
};

pub const Header = struct {
    names: std.StringArrayHashMapUnmanaged(Entry) = .empty,
    format: []const u8 = "",
    version: []const u8 = "",
    layer: ?u32 = null,
    experts: ?usize = null,
    split: ?SplitMeta = null,
};

/// The converter's `split` metadata (a JSON object inside a string).
pub const SplitMeta = struct { node: u32, tp2_rank: u32, pair: u32, half: [2]usize, main: [2]usize, rest: [2]usize, dcols: [2]usize };

fn dtypeSize(dt: []const u8) ?usize {
    const t = [_]struct { []const u8, usize }{ .{ "I32", 4 }, .{ "F32", 4 }, .{ "BF16", 2 }, .{ "U8", 1 }, .{ "I8", 1 }, .{ "F8_E8M0", 1 }, .{ "F8_E4M3", 1 }, .{ "F16", 2 }, .{ "I64", 8 } };
    for (t) |x| if (std.mem.eql(u8, x[0], dt)) return x[1];
    return null;
}

fn juint(v: std.json.Value) ?usize {
    return switch (v) {
        .integer => |i| std.math.cast(usize, i),
        .number_string => |s| std.fmt.parseInt(usize, s, 10) catch null,
        else => null,
    };
}

fn jrange(o: std.json.ObjectMap, key: []const u8) ![2]usize {
    const v = o.get(key) orelse return error.BadSliceMeta;
    if (v != .array or v.array.items.len != 2) return error.BadSliceMeta;
    return .{ juint(v.array.items[0]) orelse return error.BadSliceMeta, juint(v.array.items[1]) orelse return error.BadSliceMeta };
}

fn jint(o: std.json.ObjectMap, key: []const u8) !u32 {
    const v = o.get(key) orelse return error.BadSliceMeta;
    return std.math.cast(u32, juint(v) orelse return error.BadSliceMeta) orelse error.BadSliceMeta;
}

/// The header JSON of a slice file (any safetensors file whose tensors have rank <= 5); every entry's byte count must
/// equal its shape's and fit in `data_len` bytes.
pub fn parseHeader(arena: std.mem.Allocator, json: []const u8, data_len: usize) !Header {
    const v = try std.json.parseFromSliceLeaky(std.json.Value, arena, json, .{});
    if (v != .object) return error.BadSafetensors;
    var h: Header = .{};
    var it = v.object.iterator();
    while (it.next()) |kv| {
        if (std.mem.eql(u8, kv.key_ptr.*, "__metadata__")) {
            if (kv.value_ptr.* != .object) return error.BadSafetensors;
            const m = kv.value_ptr.object;
            if (m.get("format")) |f| if (f == .string) {
                h.format = f.string;
            };
            if (m.get("version")) |f| if (f == .string) {
                h.version = f.string;
            };
            if (m.get("layer")) |f| if (f == .string) {
                h.layer = std.fmt.parseInt(u32, f.string, 10) catch return error.BadSliceMeta;
            };
            if (m.get("experts")) |f| if (f == .string) {
                h.experts = std.fmt.parseInt(usize, f.string, 10) catch return error.BadSliceMeta;
            };
            if (m.get("split")) |f| if (f == .string) {
                const sv = try std.json.parseFromSliceLeaky(std.json.Value, arena, f.string, .{});
                if (sv != .object) return error.BadSliceMeta;
                const o = sv.object;
                h.split = .{ .node = try jint(o, "node"), .tp2_rank = try jint(o, "tp2_rank"), .pair = try jint(o, "pair"), .half = try jrange(o, "half"), .main = try jrange(o, "main"), .rest = try jrange(o, "rest"), .dcols = try jrange(o, "dcols") };
            };
            continue;
        }
        if (kv.value_ptr.* != .object) return error.BadSafetensors;
        const o = kv.value_ptr.object;
        const dt = o.get("dtype") orelse return error.BadSafetensors;
        if (dt != .string) return error.BadSafetensors;
        const size = dtypeSize(dt.string) orelse return error.UnsupportedDType;
        const sh = o.get("shape") orelse return error.BadSafetensors;
        if (sh != .array or sh.array.items.len > max_rank) return error.BadSafetensors;
        var e: Entry = .{ .dtype = dt.string, .rank = sh.array.items.len, .shape = @splat(1), .begin = 0, .end = 0 };
        var n: usize = size;
        for (sh.array.items, 0..) |d, i| {
            e.shape[i] = juint(d) orelse return error.BadSafetensors;
            n = std.math.mul(usize, n, e.shape[i]) catch return error.BadSafetensors;
        }
        const offs = try jrange(o, "data_offsets");
        e.begin = offs[0];
        e.end = offs[1];
        if (e.end < e.begin or e.end - e.begin != n or e.end > data_len) return error.BadSafetensors;
        try h.names.put(arena, kv.key_ptr.*, e);
    }
    return h;
}

/// The loader's checks of a slice header against the split the engine runs: the converter's format, the layer, the
/// split metadata equal to Split.expertParts' (node, pair, rank, every range), and every tensor's dtype and shape.
pub fn checkHeader(h: Header, x: Expect) !void {
    if (!std.mem.eql(u8, h.format, "mx")) return error.MxSliceNotMx;
    if (h.layer == null or h.layer.? != x.layer) return error.MxSliceWrongLayer;
    if (h.experts == null or h.experts.? != x.experts) return error.MxSliceWrongExperts;
    const sp = h.split orelse return error.MxSliceNoSplit;
    if (sp.node != x.node or sp.pair != x.pair or sp.tp2_rank != x.tp2_rank) return error.MxSliceWrongNode;
    for ([_][2][2]usize{ .{ sp.half, x.half }, .{ sp.main, x.main }, .{ sp.rest, x.rest }, .{ sp.dcols, x.dcols } }) |r|
        if (r[0][0] != r[1][0] or r[0][1] != r[1][1]) return error.MxSliceSplitMismatch;
    const e = x.experts;
    const mb = x.mainW() / 32;
    const rb = x.restW() / 32;
    const kg = x.hidden / 32;
    const want = [_]struct { []const u8, []const u8, []const usize }{
        .{ "gu_main", "I32", &.{ e, mb, kg, 2, block_words } },
        .{ "gu_main_scale", "F32", &.{ e, 2 } },
        .{ "gu_rest", "I32", &.{ x.n_rest, rb, kg, 2, block_words } },
        .{ "gu_rest_ids", "I32", &.{x.n_rest} },
        .{ "gu_rest_scale", "F32", &.{ x.n_rest, 2 } },
        .{ "down", "I32", &.{ e, x.downN() / 32, x.halfW() / 32, 1, block_words } },
        .{ "down_scale", "F32", &.{ e, 1 } },
        .{ "shared_gate", "BF16", &.{ x.sharedW(), x.hidden } },
        .{ "shared_up", "BF16", &.{ x.sharedW(), x.hidden } },
        .{ "shared_down", "BF16", &.{ x.downN(), x.halfW() } },
    };
    for (want) |w| {
        const got = h.names.get(w[0]) orelse return error.MxSliceMissingTensor;
        if (!got.is(w[1], w[2])) {
            std.log.warn("mx slice {s}: {s} {any}, expected {s} {any}", .{ w[0], got.dtype, got.shape[0..got.rank], w[1], w[2] });
            return error.MxSliceShape;
        }
    }
}

/// The rest map (int32 [experts + 1]: expert -> its row in gu_rest, every other id -> n_rest, skipped) from gu_rest_ids,
/// which must be exactly the experts whose rest this pair computes, ascending.
pub fn restMap(out: []i32, ids: []const i32, x: Expect) !void {
    if (out.len != x.experts + 1 or ids.len != x.n_rest) return error.MxSliceRestIds;
    @memset(out, @intCast(x.n_rest));
    var j: usize = 0;
    for (0..x.experts) |e| {
        if (plan_mod.restPair(e, x.experts) != x.pair) continue;
        if (j >= ids.len or ids[j] != @as(i32, @intCast(e))) return error.MxSliceRestIds;
        out[e] = @intCast(j);
        j += 1;
    }
    if (j != ids.len) return error.MxSliceRestIds;
}

/// The owner table assembleRest reads (int32 [experts + 1]): the pair computing expert e's rest (plan.restPair).
pub fn ownerTable(out: []i32, experts: usize) void {
    for (out, 0..) |*o, e| o.* = @intCast(plan_mod.restPair(e, experts));
}

/// A layer's slice in device memory (cuMemAlloc, as the weights).
pub const Layer = struct {
    x: Expect,
    bufs: [10]cuda.DeviceBuffer = undefined,
    nbufs: usize = 0,
    bytes: usize = 0, // device bytes held
    gu_main: u64 = 0, // int32 [E, 16, 160, 2, 136]
    gu_main_scale: u64 = 0, // fp32 [E, 2]
    gu_rest: u64 = 0, // int32 [n_rest, 4, 160, 2, 136]
    gu_rest_scale: u64 = 0, // fp32 [n_rest, 2]
    down: u64 = 0, // int32 [E, 80, 36, 1, 136]
    down_scale: u64 = 0, // fp32 [E, 1]
    rest_map: u64 = 0, // int32 [E + 1]
    owner: u64 = 0, // int32 [E + 1]
    shared_gu: u64 = 0, // bf16 [2 sw, D]: the gate rows, then the up rows
    shared_down: u64 = 0, // bf16 [dn, half]
    host_rest_map: []i32 = &.{}, // (gpa) the rest map, for the bench's host checks

    pub fn deinit(l: *Layer, gpa: std.mem.Allocator) void {
        for (l.bufs[0..l.nbufs]) |*b| b.free();
        if (l.host_rest_map.len != 0) gpa.free(l.host_rest_map);
        l.* = undefined;
    }

    fn take(l: *Layer, d: *const cuda.Driver, n: usize) !cuda.DeviceBuffer {
        const b = try cuda.DeviceBuffer.alloc(d, n);
        l.bufs[l.nbufs] = b;
        l.nbufs += 1;
        l.bytes += n;
        return b;
    }
};

/// `dir`/layerLL.nodeG.mx.safetensors.
pub fn slicePath(buf: []u8, dir: []const u8, layer: u32, node: u32) ![:0]const u8 {
    return std.fmt.bufPrintSentinel(buf, "{s}/layer{d:0>2}.node{d}.mx.safetensors", .{ dir, layer, node }, 0);
}

/// The bounce buffer the loader reads through (direct I/O: 4 KiB aligned).
pub const bounce_bytes: usize = 64 << 20;

/// Reads a slice file (direct I/O where the file system allows it), checks it against `x` (checkHeader, restMap) and
/// uploads it: every tensor into its own cuMemAlloc buffer, the shared gate and up rows into one.
pub fn loadLayer(gpa: std.mem.Allocator, d: *const cuda.Driver, dir: []const u8, x: Expect) !Layer {
    var pb: [std.fs.max_path_bytes]u8 = undefined;
    const path = try slicePath(&pb, dir, x.layer, x.node);
    var f = dio.File.open(path) catch |err| {
        std.log.err("mx slice {s}: {s}", .{ path, @errorName(err) });
        return err;
    };
    defer f.close();
    const st = try fileSize(f.fd);
    const buf = try gpa.alignedAlloc(u8, .fromByteUnits(dio.alignment), bounce_bytes);
    defer gpa.free(buf);
    const n0 = try f.read(buf, 0, 8);
    const hlen: usize = @intCast(std.mem.readInt(u64, n0[0..8], .little));
    if (hlen > st - 8 or hlen > dio.fits(bounce_bytes)) return error.BadSafetensors;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const hj = try arena.allocator().dupe(u8, try f.read(buf, 8, hlen));
    const h = try parseHeader(arena.allocator(), hj, st - 8 - hlen);
    try checkHeader(h, x);
    const base = 8 + hlen;
    var l: Layer = .{ .x = x };
    errdefer l.deinit(gpa);
    // the rest ids -> the rest map (host, then device), and the owner table
    const ids_e = h.names.get("gu_rest_ids").?;
    const ids_b = try f.read(buf, base + ids_e.begin, ids_e.end - ids_e.begin);
    const ids = try arena.allocator().alloc(i32, x.n_rest);
    @memcpy(std.mem.sliceAsBytes(ids), ids_b);
    l.host_rest_map = try gpa.alloc(i32, x.experts + 1);
    try restMap(l.host_rest_map, ids, x);
    const rm = try l.take(d, 4 * (x.experts + 1));
    try rm.upload(0, std.mem.sliceAsBytes(l.host_rest_map));
    l.rest_map = rm.ptr;
    const own = try arena.allocator().alloc(i32, x.experts + 1);
    ownerTable(own, x.experts);
    const ob = try l.take(d, 4 * (x.experts + 1));
    try ob.upload(0, std.mem.sliceAsBytes(own));
    l.owner = ob.ptr;
    // the tensors
    const plain = [_]struct { []const u8, *u64 }{ .{ "gu_main", &l.gu_main }, .{ "gu_main_scale", &l.gu_main_scale }, .{ "gu_rest", &l.gu_rest }, .{ "gu_rest_scale", &l.gu_rest_scale }, .{ "down", &l.down }, .{ "down_scale", &l.down_scale }, .{ "shared_down", &l.shared_down } };
    for (plain) |p| {
        const e = h.names.get(p[0]).?;
        const b = try l.take(d, e.end - e.begin);
        try copyIn(f, buf, base + e.begin, e.end - e.begin, b, 0);
        p[1].* = b.ptr;
    }
    const sg = h.names.get("shared_gate").?;
    const su = h.names.get("shared_up").?;
    const sb = try l.take(d, (sg.end - sg.begin) + (su.end - su.begin));
    try copyIn(f, buf, base + sg.begin, sg.end - sg.begin, sb, 0);
    try copyIn(f, buf, base + su.begin, su.end - su.begin, sb, sg.end - sg.begin);
    l.shared_gu = sb.ptr;
    return l;
}

fn fileSize(fd: std.c.fd_t) !usize {
    const end = std.c.lseek(fd, 0, std.c.SEEK.END);
    if (end < 0) return error.ReadFailed;
    return @intCast(end);
}

/// `len` file bytes at `off` into device buffer `b` at `at`, through the bounce buffer in aligned pieces.
fn copyIn(f: dio.File, buf: dio.Buffer, off: usize, len: usize, b: cuda.DeviceBuffer, at: usize) !void {
    const step = dio.fits(buf.len) / dio.alignment * dio.alignment;
    var done: usize = 0;
    while (done < len) {
        const n = @min(step, len - done);
        const got = try f.read(buf, off + done, n);
        try b.upload(at + done, got);
        done += n;
    }
}

/// /proc/meminfo's MemAvailable in bytes (null: unreadable). The loader's caller refuses to go on below
/// TF_DS_MXFP4_MIN_AVAIL_GIB after loading (stageMX0 section 2: the slices take host memory on GB10).
pub fn memAvailable(text: []const u8) ?u64 {
    var it = std.mem.splitScalar(u8, text, '\n');
    while (it.next()) |line| {
        if (!std.mem.startsWith(u8, line, "MemAvailable:")) continue;
        var f = std.mem.tokenizeAny(u8, line["MemAvailable:".len..], " \t");
        const n = std.fmt.parseInt(u64, f.next() orelse return null, 10) catch return null;
        const unit = f.next() orelse return null;
        if (!std.mem.eql(u8, unit, "kB")) return null;
        return n * 1024;
    }
    return null;
}

// -- a chunk's launches ----------------------------------------------------------------------------------------------

/// A chunk's scratch (milestone 2 takes it from the prompt arena): the routed plan, the rest plan and its remapped
/// picks, and the shared expert's gate / up GEMM output.
pub const Scratch = struct {
    rows: usize,
    slots: usize,
    main: PlanScratch, // experts + 1 ids
    rest: PlanScratch, // n_rest + 1 ids
    rpick: u64, // int32 [rows * slots]
    g: u64, // fp32 [rows, 2 sw]

    /// Bytes of: main members, items, counts, rank, hist; rest the same five; rpick; g.
    pub fn sizes(rows: usize, slots: usize, x: Expect) [12]usize {
        const p = rows * slots;
        const m = PlanScratch.sizes(p, x.experts + 1);
        const r = PlanScratch.sizes(p, x.n_rest + 1);
        return .{ m[0], m[1], m[2], m[3], m[4], r[0], r[1], r[2], r[3], r[4], 4 * @max(p, 1), 4 * @max(rows, 1) * 2 * x.sharedW() };
    }

    /// From buffers of sizes() (in that order).
    pub fn of(rows: usize, slots: usize, x: Expect, ptrs: [12]u64) Scratch {
        const p = rows * slots;
        return .{ .rows = rows, .slots = slots, .main = .{ .pairs = p, .experts = x.experts + 1, .members = ptrs[0], .items = ptrs[1], .counts = ptrs[2], .rank = ptrs[3], .hist = ptrs[4] }, .rest = .{ .pairs = p, .experts = x.n_rest + 1, .members = ptrs[5], .items = ptrs[6], .counts = ptrs[7], .rank = ptrs[8], .hist = ptrs[9] }, .rpick = ptrs[10], .g = ptrs[11] };
    }
};

/// The pack (this node's gate / up output, the bytes the column partner receives): bf16 main rows [P, main_w], then
/// the rest rows [P, rest_w] (exl3_prompt's sc.xd layout at ig = main_w, ir = rest_w).
pub fn packRest(pack: u64, p: usize, x: Expect) u64 {
    return pack + p * x.mainW() * 2;
}

/// Both plans of a chunk: pick int32 [n, slots] (the shared expert, id E, in the last slot).
pub fn plan(k: *const Kernels, o: *const ops_mod.Ops, s: cuda.Stream, lay: *const Layer, sc: Scratch, pick: u64, n: usize) !void {
    const p = n * sc.slots;
    try route(k, s, pick, p, sc.main);
    try o.remapPicks(s, pick, lay.rest_map, sc.rpick, p);
    try route(k, s, sc.rpick, p, sc.rest);
}

/// Gate / up's main columns of every routed pair into the pack's main rows (the shared slot's rows: sharedGateUp).
pub fn gateMain(k: *const Kernels, s: cuda.Stream, lay: *const Layer, sc: Scratch, x: u64, x_stride: usize, n: usize, pack: u64, limit: f32, geo: u32) !void {
    const X = lay.x;
    try gemm(k, s, .{ .epi = .gate_up, .geo = geo, .x = x, .x_stride = x_stride, .slots = sc.slots, .w = lay.gu_main, .scale = lay.gu_main_scale, .kg = X.hidden / 32, .nb = X.mainW() / 32, .out = pack, .limit = limit, .skip = @intCast(X.experts), .pairs = n * sc.slots }, sc.main);
}

/// Gate / up's rest block of the pairs whose expert's rest this pair computes, into the pack's rest rows.
pub fn gateRest(k: *const Kernels, s: cuda.Stream, lay: *const Layer, sc: Scratch, x: u64, x_stride: usize, n: usize, pack: u64, limit: f32, geo: u32) !void {
    const X = lay.x;
    const p = n * sc.slots;
    try gemm(k, s, .{ .epi = .gate_up, .geo = geo, .x = x, .x_stride = x_stride, .slots = sc.slots, .w = lay.gu_rest, .scale = lay.gu_rest_scale, .kg = X.hidden / 32, .nb = X.restW() / 32, .out = packRest(pack, p, X), .limit = limit, .skip = @intCast(X.n_rest), .pairs = p }, sc.rest);
}

/// The shared expert's gate / up (one cuBLAS GEMM over [gate; up], fp32 out) and its SwiGLU into the pack's last-slot
/// rows: the main columns, and the rest columns on the pair that computes the shared expert's rest.
pub fn sharedGateUp(k: *const Kernels, blas: *const cublas.Blas, s: cuda.Stream, lay: *const Layer, sc: Scratch, x: u64, x_stride: usize, n: usize, pack: u64, limit: f32) !void {
    if (n == 0) return;
    const X = lay.x;
    const sw = X.sharedW();
    try blas.gemmBf16F32(x, x_stride, lay.shared_gu, sc.g, 2 * sw, n, X.hidden, 2 * sw);
    var a: cuda.Args = .{};
    a.add(sc.g);
    i32a(&a, sw);
    i32a(&a, sc.slots);
    i32a(&a, sc.slots - 1);
    a.add(pack);
    i32a(&a, X.mainW());
    a.add(packRest(pack, n * sc.slots, X));
    i32a(&a, X.restW());
    a.add(limit);
    try go(k.shared_swiglu, s, n, 256, 0, &a);
}

/// Everything before the pack exchange: both plans, the main columns, the rest block, the shared expert.
pub fn gateUp(k: *const Kernels, o: *const ops_mod.Ops, blas: *const cublas.Blas, s: cuda.Stream, lay: *const Layer, sc: Scratch, x: u64, x_stride: usize, pick: u64, n: usize, pack: u64, limit: f32, g: Geos) !void {
    try plan(k, o, s, lay, sc, pick, n);
    try gateMain(k, s, lay, sc, x, x_stride, n, pack, limit, g.main);
    try gateRest(k, s, lay, sc, x, x_stride, n, pack, limit, g.rest);
    try sharedGateUp(k, blas, s, lay, sc, x, x_stride, n, pack, limit);
}

/// Down of every routed pair: xd_full bf16 [P, half] -> y fp32 [P, dn] (the routed plan from plan()).
pub fn downRouted(k: *const Kernels, s: cuda.Stream, lay: *const Layer, sc: Scratch, xd_full: u64, n: usize, y: u64, geo: u32) !void {
    const X = lay.x;
    try gemm(k, s, .{ .epi = .down, .geo = geo, .x = xd_full, .x_stride = X.halfW(), .slots = 0, .w = lay.down, .scale = lay.down_scale, .kg = X.halfW() / 32, .nb = X.downN() / 32, .out = y, .limit = 0, .skip = @intCast(X.experts), .pairs = n * sc.slots }, sc.main);
}

/// The shared expert's down: xd_full's last-slot rows x W_d^T -> y's last-slot rows (cuBLAS, strided rows).
pub fn sharedDown(blas: *const cublas.Blas, lay: *const Layer, xd_full: u64, n: usize, slots: usize, y: u64) !void {
    if (n == 0) return;
    const X = lay.x;
    const last = slots - 1;
    try blas.gemmBf16F32(xd_full + last * X.halfW() * 2, slots * X.halfW(), lay.shared_down, y + last * X.downN() * 4, slots * X.downN(), n, X.halfW(), X.downN());
}

/// out fp32 [n, d] (row stride out_ld floats) = each row's slots of y weighted by wts [n, slots], in slot order.
pub fn combine(k: *const Kernels, s: cuda.Stream, y: u64, wts: u64, out: u64, out_ld: usize, n: usize, slots: usize, d: usize) !void {
    if (n == 0) return;
    if (d % 4 != 0 or out_ld % 4 != 0 or out % 16 != 0 or y % 16 != 0) return error.UnsupportedShape;
    const n4 = n * (d / 4);
    var a: cuda.Args = .{};
    a.add(y);
    a.add(wts);
    a.add(out);
    a.add(@as(c_longlong, @intCast(n4)));
    i32a(&a, d / 4);
    i32a(&a, slots);
    a.add(@as(c_longlong, @intCast(out_ld / 4)));
    try go(k.combine, s, cdiv(n4, 256), 256, 0, &a);
}

/// Everything after the exchange: down of the routed pairs and the shared slot into y, then the combine into out.
pub fn down(k: *const Kernels, blas: *const cublas.Blas, s: cuda.Stream, lay: *const Layer, sc: Scratch, xd_full: u64, wts: u64, n: usize, y: u64, out: u64, geo: u32) !void {
    try downRouted(k, s, lay, sc, xd_full, n, y, geo);
    try sharedDown(blas, lay, xd_full, n, sc.slots, y);
    try combine(k, s, y, wts, out, lay.x.downN(), n, sc.slots, lay.x.downN());
}

// -- host tests (no GPU) ----------------------------------------------------------------------------------------------

const testing = std.testing;

fn testConfig(a: std.mem.Allocator) !Config {
    const cfg = @import("config.zig");
    var why: cfg.Why = .{};
    return cfg.parse(a, cfg.test_config, &why);
}

fn split4(g: u32) plan_mod.Split {
    return .{ .rank = g % 2, .world = 2, .pair = g / 2, .parity = true };
}

test "the split each node's slice must carry is the converter's (convert_mxfp4_slices.split)" {
    const c = try testConfig(testing.allocator);
    // convert_mxfp4_slices.split(g): half [r 1152, +1152), main half0 + p 512 (+512), rest [half0 + 1024, half0 + 1152),
    // dcols [p 2560, +2560); rest experts e % 2 == p (192), the shared expert's rest on pair 1
    for (0..4) |gi| {
        const g: u32 = @intCast(gi);
        const x = try Expect.of(c, split4(g), 7);
        const r: usize = g % 2;
        const p: usize = g / 2;
        const h0 = r * 1152;
        try testing.expectEqual([2]usize{ h0, h0 + 1152 }, x.half);
        try testing.expectEqual([2]usize{ h0 + p * 512, h0 + (p + 1) * 512 }, x.main);
        try testing.expectEqual([2]usize{ h0 + 1024, h0 + 1152 }, x.rest);
        try testing.expectEqual([2]usize{ p * 2560, (p + 1) * 2560 }, x.dcols);
        try testing.expectEqual(@as(usize, 192), x.n_rest);
        try testing.expectEqual(p == 1, x.shared_rest);
        try testing.expectEqual(@as(usize, if (p == 1) 640 else 512), x.sharedW());
        try testing.expectEqual(g, x.node);
    }
    var s = split4(0);
    s.parity = false;
    try testing.expectError(error.MxNeedsTheParitySplit, Expect.of(c, s, 0));
    try testing.expectError(error.NotA2DSplit, Expect.of(c, .{ .rank = 0, .world = 2, .parity = true }, 0));
}

/// layer00.node0.mx.safetensors' header as the converter wrote it (stage MX0, castor; offsets real).
const real_header =
    \\{"gu_main":{"dtype":"I32","shape":[384,16,160,2,136],"data_offsets":[0,1069547520]},
    \\"gu_main_scale":{"dtype":"F32","shape":[384,2],"data_offsets":[1069547520,1069550592]},
    \\"gu_rest":{"dtype":"I32","shape":[192,4,160,2,136],"data_offsets":[1069550592,1203244032]},
    \\"gu_rest_ids":{"dtype":"I32","shape":[192],"data_offsets":[1203244032,1203244800]},
    \\"gu_rest_scale":{"dtype":"F32","shape":[192,2],"data_offsets":[1203244800,1203246336]},
    \\"down":{"dtype":"I32","shape":[384,80,36,1,136],"data_offsets":[1203246336,1804866816]},
    \\"down_scale":{"dtype":"F32","shape":[384,1],"data_offsets":[1804866816,1804868352]},
    \\"shared_gate":{"dtype":"BF16","shape":[512,5120],"data_offsets":[1804868352,1810111232]},
    \\"shared_up":{"dtype":"BF16","shape":[512,5120],"data_offsets":[1810111232,1815354112]},
    \\"shared_down":{"dtype":"BF16","shape":[2560,1152],"data_offsets":[1815354112,1821252352]},
    \\"__metadata__":{"format":"mx","version":"stageMX0-1","layer":"0","split":"{\"node\": 0, \"host\": \"castor\", \"tp2_rank\": 0, \"pair\": 0, \"half\": [0, 1152], \"main\": [0, 512], \"rest\": [1024, 1152], \"rest_experts\": \"e % 2 == pair\", \"dcols\": [0, 2560]}","source":"model-00003-of-00048.safetensors","experts":"384","layout":"see convert_mxfp4_slices.py docstring"}}
;
const real_data: usize = 1821252352;

test "a real slice header parses and passes node 0's checks, and only node 0's" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const c = try testConfig(a);
    const h = try parseHeader(a, real_header, real_data);
    try testing.expectEqual(@as(usize, 10), h.names.count());
    try testing.expectEqual(@as(?u32, 0), h.layer);
    try testing.expectEqual(@as(usize, 1069547520), h.names.get("gu_main").?.end);
    try checkHeader(h, try Expect.of(c, split4(0), 0));
    try testing.expectError(error.MxSliceWrongLayer, checkHeader(h, try Expect.of(c, split4(0), 1)));
    try testing.expectError(error.MxSliceWrongNode, checkHeader(h, try Expect.of(c, split4(1), 0)));
    try testing.expectError(error.MxSliceWrongNode, checkHeader(h, try Expect.of(c, split4(2), 0)));
    // a split whose node and pair agree but whose ranges do not (another TF_DS_2D_DOWN)
    var s = split4(0);
    s.down0 = 18;
    try testing.expectError(error.MxSliceSplitMismatch, checkHeader(h, try Expect.of(c, s, 0)));
    // a short file, a wrong byte count, a rank past 5
    try testing.expectError(error.BadSafetensors, parseHeader(a, real_header, real_data - 1));
    const bad = try std.mem.replaceOwned(u8, a, real_header, "[384,2],\"data_offsets\":[1069547520,1069550592]", "[384,3],\"data_offsets\":[1069547520,1069550592]");
    try testing.expectError(error.BadSafetensors, parseHeader(a, bad, real_data));
    const r6 = try std.mem.replaceOwned(u8, a, real_header, "[384,16,160,2,136]", "[384,16,160,2,136,1]");
    try testing.expectError(error.BadSafetensors, parseHeader(a, r6, real_data));
    // a shape the split does not give (pair 1's shared rows on pair 0)
    const sh = try std.mem.replaceOwned(u8, a, real_header, "\"shared_gate\":{\"dtype\":\"BF16\",\"shape\":[512,5120],\"data_offsets\":[1804868352,1810111232]}", "\"shared_gate\":{\"dtype\":\"BF16\",\"shape\":[512,5120],\"data_offsets\":[1804868352,1810111232]},\"x\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[0,1]}");
    try checkHeader(try parseHeader(a, sh, real_data), try Expect.of(c, split4(0), 0)); // extra tensors are ignored
    const fmt = try std.mem.replaceOwned(u8, a, real_header, "\"format\":\"mx\"", "\"format\":\"raw\"");
    try testing.expectError(error.MxSliceNotMx, checkHeader(try parseHeader(a, fmt, real_data), try Expect.of(c, split4(0), 0)));
    const sw = try std.mem.replaceOwned(u8, a, real_header, "\"shape\":[512,5120],\"data_offsets\":[1810111232", "\"shape\":[256,10240],\"data_offsets\":[1810111232");
    try testing.expectError(error.MxSliceShape, checkHeader(try parseHeader(a, sw, real_data), try Expect.of(c, split4(0), 0)));
}

test "the rest map and the owner table" {
    const c = try testConfig(testing.allocator);
    for ([_]u32{ 0, 2 }) |g| {
        const x = try Expect.of(c, split4(g), 0);
        var ids: [192]i32 = undefined;
        for (&ids, 0..) |*v, j| v.* = @intCast(2 * j + x.pair);
        var m: [385]i32 = undefined;
        try restMap(&m, &ids, x);
        for (0..384) |e| try testing.expectEqual(@as(i32, if (e % 2 == x.pair) @intCast(e / 2) else 192), m[e]);
        try testing.expectEqual(@as(i32, 192), m[384]); // the shared expert: never by the routed rest launch
        ids[5] += 2;
        try testing.expectError(error.MxSliceRestIds, restMap(&m, &ids, x));
    }
    var o: [385]i32 = undefined;
    ownerTable(&o, 384);
    try testing.expectEqual(@as(i32, 0), o[0]);
    try testing.expectEqual(@as(i32, 1), o[383]);
    try testing.expectEqual(@as(i32, 1), o[384]);
}

test "entry shapes, symbols and grids" {
    // upstream's Tile at geometries 2, 3, 4: gate / up (M 2) 33,792 / 42,496 / 51,200 B, down (M 1) 25,088 / 29,440 / 33,792 B
    try testing.expectEqual(Shape{ .threads = 256, .smem = 33792, .cb = 4, .bm = 64 }, try tileShape(2, 2));
    try testing.expectEqual(Shape{ .threads = 384, .smem = 42496, .cb = 6, .bm = 64 }, try tileShape(2, 3));
    try testing.expectEqual(Shape{ .threads = 256, .smem = 51200, .cb = 8, .bm = 64 }, try tileShape(2, 4));
    try testing.expectEqual(Shape{ .threads = 256, .smem = 25088, .cb = 4, .bm = 64 }, try tileShape(1, 2));
    try testing.expectEqual(Shape{ .threads = 384, .smem = 29440, .cb = 6, .bm = 64 }, try tileShape(1, 3));
    try testing.expectEqual(Shape{ .threads = 256, .smem = 33792, .cb = 8, .bm = 64 }, try tileShape(1, 4));
    try testing.expectError(error.NoSuchGeometry, tileShape(2, 1));
    // ptxas's names for the instantiations (cuobjdump of the built image)
    try testing.expectEqualStrings("_ZN5tf_mx20nvfp4_prefill_kernelILi1ELi2ELi2ELi2ELi2ELi4ELi1ELi2ELi2EEEvPK13__nv_bfloat16iiPK5uint4PKfiiPKiSA_SA_Pvifi", entry_symbols[0]);
    try testing.expectEqualStrings("_ZN5tf_mx20nvfp4_prefill_kernelILi1ELi1ELi0ELi2ELi2ELi4ELi2ELi2ELi2EEEvPK13__nv_bfloat16iiPK5uint4PKfiiPKiSA_SA_Pvifi", entry_symbols[5]);
    // grouped.max_items at 4,096 rows of 7 slots over 385 ids: 385 + 448
    try testing.expectEqual(@as(usize, 833), maxItems(4096 * 7, 385, 64));
    try testing.expectEqual(@as(usize, 8), maxItems(8, 385, 64));
    const ps = PlanScratch.sizes(4096 * 7, 385);
    try testing.expectEqual(@as(usize, 4 * 28 * 385), ps[4]);
}

test "MemAvailable from /proc/meminfo's text" {
    try testing.expectEqual(@as(?u64, 41 * 1024 * 1024 * 1024), memAvailable("MemTotal:       128000000 kB\nMemFree:  1 kB\nMemAvailable:   42991616 kB\n"));
    try testing.expectEqual(@as(?u64, null), memAvailable("MemTotal: 1 kB\n"));
}
