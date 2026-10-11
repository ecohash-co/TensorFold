//! TF_DS_EXL3_PROMPT=v2 (default: unset, today's launches): a 2D node's prompt-chunk routed experts with our image's
//! gate / up and down GEMMs (zig/kernels/cuda/dsv41/exl3_prompt.cu) in place of the served build's rot_in, grouped_mma
//! (gate / up), the parity rest's grouping and grouped_mma, and grouped_rows (down), and (TF_DS_EXL3_WQ_CACHE, on with
//! v2) the dense prompt GEMMs' W_q unpacked once at load. Every output element is the served build's: the same decoded
//! weight tiles, the same fp16 input rows (rot_in's arithmetic in the program, its contraction pinned), each K chain an
//! fp32 mma chain from zero over the same k tiles in the same order, the chains and the splits added in the served
//! order; the served grouping, epilogues and combine run as today. So sc.xd (main and rest rows), the down partials and
//! the combined rows are today's bytes (tf-dsv41-exl3-bench checks them on a rank's real weights).
//! Launches a chunk: group_count, group_place, work_list, (parity: rest_picks), gate / up's main columns, the rest block,
//! the gate / up epilogue (and the rest's); after the exchange: down, down_combine. Today: thirteen, with rot_in's
//! 2 x P x D fp16 rows written once and read twice.
const std = @import("std");
const cuda = @import("cuda");
const weights = @import("weights.zig");
const X = @import("exl3_experts.zig");
const X2 = @import("exl3_experts2d.zig");
const ops_mod = @import("ops.zig");
const exl3_prefill = @import("exl3_prefill.zig");

/// TF_DS_EXL3_PROMPT: "v2" takes these kernels; unset (or anything else but a refused value) today's.
pub fn enabled() !bool {
    const v = std.c.getenv("TF_DS_EXL3_PROMPT") orelse return false;
    const s = std.mem.span(v);
    if (s.len == 0 or std.mem.eql(u8, s, "v1")) return false;
    if (std.mem.eql(u8, s, "v2")) return true;
    return error.BadExl3Prompt;
}

/// TF_DS_EXL3_WQ_CACHE (with v2; default on, "0" off): each encoder layer's dense prompt GEMMs read W_q unpacked once
/// at load (exl3_prefill.matmul with weights.Linear.wq) instead of unpacking it on every chunk.
pub fn wqCache() bool {
    const v = std.c.getenv("TF_DS_EXL3_WQ_CACHE") orelse return true;
    return !std.mem.eql(u8, std.mem.span(v), "0");
}

/// The W_q cache: every linear the prompt GEMM takes (more than 128 rows) in layers [0, layers) unpacked once into a
/// buffer of its own (held by the weights, freed with them) and set as its .wq; a linear whose bits have no unpack
/// kernel keeps unpacking per call. Returns the bytes held.
pub fn cacheWq(d: *const cuda.Driver, w: *weights.Weights, pf: *const exl3_prefill.Kernels, s: cuda.Stream, layers: usize) !usize {
    var total: usize = 0;
    for (w.layers[0..@min(layers, w.layers.len)]) |*lay| {
        var ls: [16]*weights.Linear = undefined;
        var n: usize = 0;
        for ([_]*weights.Linear{ &lay.wq_a, &lay.wkv, &lay.wq_b, &lay.wo_b }) |l| {
            ls[n] = l;
            n += 1;
        }
        for (lay.wo_a[0..lay.groups]) |*l| {
            ls[n] = l;
            n += 1;
        }
        for ([_]*?weights.Linear{ &lay.comp_wkv, &lay.comp_wgate, &lay.idx_wq_b, &lay.idx_wk, &lay.engram_wkv }) |ol| {
            if (ol.*) |*l| {
                ls[n] = l;
                n += 1;
            }
        }
        for (ls[0..n]) |l| {
            if (l.wq != 0) continue;
            if (l.k2 > 16 or pf.unpack[l.k2][@import("exl3_linear.zig").codebook_mul1] == null) continue;
            const bytes = @as(usize, l.k) * l.n * 2;
            var b = try cuda.DeviceBuffer.alloc(d, bytes);
            errdefer b.free();
            try exl3_prefill.unpack(pf, s, l.*, b.ptr);
            try w.buffers.append(w.gpa, b);
            l.wq = b.ptr;
            total += bytes;
        }
    }
    try s.synchronize();
    return total;
}

pub const Kernels = struct {
    module: cuda.Module,
    gu: [4]cuda.Function, // by exl3_experts.range, then [3]: K2 2..6 (the ring pipeline)
    dn: [4]cuda.Function,
    rows: usize, // member rows a program (tf_ds_v2_rows): which work list the launches walk (32 or 64 rows)
    ring16: [2]cuda.Function, // the ring pipeline at 256 columns a program (16 warps), K2 2..10: gate / up, down
    ring_smem: [2]u32, // the ring entries' dynamic shared memory: 8 warps, 16 warps

    pub fn load(d: *const cuda.Driver, image: []const u8) !Kernels {
        var m = try cuda.Module.load(d, image);
        errdefer m.unload();
        const gl = try m.global("tf_ds_v2_rows");
        var rows_: c_int = 0;
        try d.check(d.api.cuMemcpyDtoH_v2(@ptrCast(&rows_), gl.ptr, @sizeOf(c_int)), "cuMemcpyDtoH");
        if (rows_ != X.rows_rows_d and rows_ != X.mma_rows_gu) return error.BadImage;
        const gs = try m.global("tf_ds_v2_ring_smem");
        var smem: [2]c_int = undefined;
        try d.check(d.api.cuMemcpyDtoH_v2(@ptrCast(&smem), gs.ptr, @sizeOf([2]c_int)), "cuMemcpyDtoH");
        const r16 = [2]cuda.Function{ try m.function("tf_ds_v2_gu16_2_10"), try m.function("tf_ds_v2_dn16_2_10") };
        const r8 = [2]cuda.Function{ try m.function("tf_ds_v2_gu_2_6"), try m.function("tf_ds_v2_dn_2_6") };
        for (r8) |f| try f.allowDynamicShared(@intCast(smem[0]));
        for (r16) |f| try f.allowDynamicShared(@intCast(smem[1]));
        return .{
            .ring16 = r16,
            .ring_smem = .{ @intCast(smem[0]), @intCast(smem[1]) },
            .rows = @intCast(rows_),
            .module = m,
            .gu = .{ try m.function("tf_ds_v2_gu_8_8"), try m.function("tf_ds_v2_gu_2_10"), try m.function("tf_ds_v2_gu_2_16"), try m.function("tf_ds_v2_gu_2_6") },
            .dn = .{ try m.function("tf_ds_v2_dn_8_8"), try m.function("tf_ds_v2_dn_2_10"), try m.function("tf_ds_v2_dn_2_16"), try m.function("tf_ds_v2_dn_2_6") },
        };
    }

    pub fn unload(k: *Kernels) void {
        k.module.unload();
        k.* = undefined;
    }
};

/// exl3_prompt.cu's Args, field for field (C layout).
pub const Args = extern struct {
    x: u64 = 0,
    x_stride: c_int = 0,
    cb0: c_int = 0, // TF_DS_PROMPT_OVL=dn: the launch's first column block (0: from the first)
    suh0: u64 = 0,
    suh1: u64 = 0,
    xa: u64 = 0,
    tp0: u64 = 0,
    tp1: u64 = 0,
    k2_0: u64 = 0,
    k2_1: u64 = 0,
    rtp0: u64 = 0,
    rtp1: u64 = 0,
    rk2_0: u64 = 0,
    rk2_1: u64 = 0,
    uids: u64 = 0,
    ucount: u64 = 0,
    members: u64 = 0,
    work: u64 = 0,
    maxm: c_int = 0,
    slots: c_int = 0,
    z: u64 = 0,
    zr: u64 = 0,
    K: c_int = 0,
    N: c_int = 0,
    NR: c_int = 0,
    P: c_int = 0,
    chain: c_int = 0,
    wk: c_int = 0,
    fold: c_int = 0,
    k2_lo: c_int = 0, // a program runs for a matrix of K2 in [k2_lo, the entry's top]
    // TF_DS_PROMPT_OVL=dn: the main Z as a compact part [mats, P, z_ld] of columns [z_c0, z_c0 + z_ld) (0: [mats, P, N])
    z_ld: c_int = 0,
    z_c0: c_int = 0,
};

comptime {
    std.debug.assert(@sizeOf(Args) == 200 and @offsetOf(Args, "cb0") == 12 and @offsetOf(Args, "suh0") == 16 and @offsetOf(Args, "maxm") == 136 and @offsetOf(Args, "z") == 144 and @offsetOf(Args, "fold") == 184 and @offsetOf(Args, "k2_lo") == 188 and @offsetOf(Args, "z_ld") == 192 and @offsetOf(Args, "z_c0") == 196);
}

fn launch(f: cuda.Function, s: cuda.Stream, grid: [3]usize, block: usize, args: *cuda.Args) !void {
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(grid[0]), .y = @intCast(grid[1]), .z = @intCast(grid[2]) }, .block = .{ .x = @intCast(block) } }, s, args);
}

/// One part's launches (gate / up's main columns, its rest block, or down), one per K2 window of [lo, hi]: K2 2..6 by the
/// ring pipeline (256 columns a program when `cols` allows, else 128), the rest by the register entries
/// (exl3_experts.range's), each program returning at once for a K2 outside its launch's window. `g` says which part
/// (N the main columns, NR the rest's; the other 0). The launches cover columns [c0, c0 + cols) of the part (c0 0:
/// all of it, today's launches; TF_DS_PROMPT_OVL=dn's down halves start at c0 = N / 2 with the same program width).
fn launchPart(k: *const Kernels, s: cuda.Stream, g0: Args, lo: u32, hi: u32, cols: usize, gate_up: bool, nw: usize, c0: usize) !void {
    const mats: usize = if (gate_up) 2 else 1;
    var g = g0;
    var top: u32 = 0; // the K2 the ring launches covered
    if (c0 % 128 != 0 or (c0 != 0 and cols % 256 == 0 and c0 % 256 != 0)) return error.UnsupportedShape;
    if (cols % 256 == 0 and lo <= 10) {
        // 256 columns a program, K2 2..10 in one launch (the shared expert's wider trellises beside the routed ones)
        g.k2_lo = 0;
        g.cb0 = @intCast(c0 / 256);
        var a: cuda.Args = .{};
        a.add(g);
        try cuda.launch.launch(k.ring16[if (gate_up) 0 else 1], .{ .grid = .{ .x = @intCast(mats * cols / 256), .y = @intCast(nw) }, .block = .{ .x = 512 }, .shared = k.ring_smem[1] }, s, &a);
        top = 10;
    } else if (lo <= 6) {
        g.k2_lo = 0;
        g.cb0 = @intCast(c0 / 128);
        var a: cuda.Args = .{};
        a.add(g);
        try cuda.launch.launch(if (gate_up) k.gu[3] else k.dn[3], .{ .grid = .{ .x = @intCast(mats * cols / 128), .y = @intCast(nw) }, .block = .{ .x = 256 }, .shared = k.ring_smem[0] }, s, &a);
        top = 6;
    }
    if (hi > top) {
        const l = @max(lo, top + 1);
        g.k2_lo = @intCast(l);
        g.cb0 = @intCast(c0 / 128);
        var a: cuda.Args = .{};
        a.add(g);
        const f = if (gate_up) k.gu[X.range(l, hi)] else k.dn[X.range(l, hi)];
        try launch(f, s, .{ mats * cols / 128, nw, 1 }, 256, &a);
    }
}

fn i32a(a: *cuda.Args, v: usize) void {
    a.add(@as(c_int, @intCast(v)));
}

/// The K chains of a served launch: k tiles a chain and chains a split (its warps), checked as the served launcher
/// checks them.
fn chains(k: usize, cfg: [4]usize) ![2]usize {
    const kt = k / 16;
    if (k % 128 != 0 or kt % (cfg[1] * cfg[2]) != 0) return error.UnsupportedShape;
    return .{ kt / (cfg[1] * cfg[2]), cfg[1] };
}

/// Which launches of a chunk's gate / up gateUpPart issues: .all today's (the grouping, the main columns, the rest
/// block, the main epilogue, the rest's); TF_DS_PROMPT_OVL=pack splits them at the pack's seam: .main (the grouping,
/// the main columns and their epilogue: sc.xd's main rows [P, ig] final) then .rest (the rest block and its epilogue:
/// the rest rows after them). The same launches with the same arguments; only the main epilogue moves before the rest
/// block's GEMM (they share no buffer: the main epilogue reads sc.z, the rest block writes rs.z).
pub const Part = enum { all, main, rest };

/// exl3_experts2d.gateUp (+ gateUpRest on a parity split) on a prompt chunk (R >= 64): sc.xd [P, ig] fp16 (then the rest's
/// rows [P, ir] after them, valid on the slots this pair computes), the same bytes. The grouping and its work lists stay
/// in sc for down.
pub fn gateUp(k: *const Kernels, xk: *const X.Kernels, o: *const ops_mod.Ops, s: cuda.Stream, ex: weights.Experts, sc: X.Scratch, rs: ?X2.RestScratch, x: u64, x_stride: usize, pick: u64, r: usize, limit: f32) !void {
    return gateUpPart(k, xk, o, s, ex, sc, rs, x, x_stride, pick, r, limit, .all);
}

pub fn gateUpPart(k: *const Kernels, xk: *const X.Kernels, o: *const ops_mod.Ops, s: cuda.Stream, ex: weights.Experts, sc: X.Scratch, rs: ?X2.RestScratch, x: u64, x_stride: usize, pick: u64, r: usize, limit: f32, part: Part) !void {
    const pl = try X2.Plan.of(ex);
    const e: usize = ex.count;
    const slots = sc.slots;
    if (r < X.exact_rows or r > sc.rows) return error.NotAPromptChunk;
    if (slots > 32 or pl.ig % 128 != 0) return error.UnsupportedShape;
    if (part != .all and ex.rest == null) return error.NoRest;
    const p = r * slots;
    const maxu = @min(p, e);
    const maxm = X.memberStride(r);
    const n_gu = X.listLen(p, maxu, X.mma_rows_gu);
    const n_d = X.listLen(p, maxu, X.rows_rows_d);
    const ch = try chains(pl.d, pl.cgu);

    var a: cuda.Args = .{};
    if (part != .rest) {
        // the served grouping and its work lists (v2 walks the 32-row one; the 64-row one is built as the served launch builds it)
        a.add(pick);
        a.add(sc.counts);
        i32a(&a, p);
        try launch(xk.group_count, s, .{ e, 1, 1 }, X.place_threads, &a);
        a = .{};
        for ([_]u64{ pick, sc.counts, sc.ids, sc.count, sc.members }) |v| a.add(v);
        for ([_]usize{ p, slots, e, maxm }) |v| i32a(&a, v);
        try launch(xk.group_place, s, .{ e, 1, 1 }, X.place_threads, &a);
        a = .{};
        for ([_]u64{ sc.counts, sc.ids, sc.count }) |v| a.add(v);
        i32a(&a, maxu);
        a.add(sc.work_gu);
        i32a(&a, X.mma_rows_gu);
        i32a(&a, n_gu);
        a.add(sc.work_d);
        i32a(&a, X.rows_rows_d);
        i32a(&a, n_d);
        a.add(@as(c_int, -1));
        try launch(xk.work_list, s, .{ 1, 1, 1 }, X.list_threads, &a);
    }

    var g: Args = .{ .x = x, .x_stride = @intCast(x_stride), .suh0 = ex.suh_g, .suh1 = ex.suh_u, .tp0 = ex.gate_ptr, .tp1 = ex.up_ptr, .k2_0 = ex.gate_k2, .k2_1 = ex.up_k2, .uids = sc.ids, .ucount = sc.count, .members = sc.members, .work = if (k.rows == X.mma_rows_gu) sc.work_gu else sc.work_d, .maxm = @intCast(maxm), .slots = @intCast(slots), .z = sc.z, .K = @intCast(pl.d), .N = @intCast(pl.ig), .P = @intCast(p), .chain = @intCast(ch[0]), .wk = @intCast(ch[1]), .fold = 1 };
    var nr: usize = 0;
    if (ex.rest) |rest| {
        const rsc = rs orelse return error.NoRestScratch;
        nr = rest.width;
        if (nr % 128 != 0) return error.UnsupportedShape;
        if (part != .main) try o.remapPicks(s, pick, rest.remap, rsc.rpick, p); // the rest epilogue's picks (another pair's experts -> E)
        g.rtp0 = rest.gate_ptr;
        g.rtp1 = rest.up_ptr;
        g.rk2_0 = rest.gate_k2;
        g.rk2_1 = rest.up_k2;
        g.zr = rsc.z;
        g.NR = @intCast(nr);
    }
    const nw = X.listLen(p, maxu, k.rows);
    // the main columns (256 a program), then the rest block (128): measured faster than the rest block as a half-idle
    // program of the main launch
    if (nw > 0 and part != .rest) {
        var gm = g;
        gm.NR = 0;
        try launchPart(k, s, gm, ex.k2_gu[0], ex.k2_gu[1], pl.ig, true, nw, 0);
    }
    if (part == .main) return mainEpilogue(xk, s, ex, sc, pick, p, pl.ig, limit);
    if (nw > 0) if (ex.rest) |rb| {
        var gr = g;
        gr.N = 0;
        try launchPart(k, s, gr, rb.k2_gu[0], rb.k2_gu[1], nr, true, nw, 0);
    };
    if (part == .all) try mainEpilogue(xk, s, ex, sc, pick, p, pl.ig, limit);
    if (ex.rest) |rest| {
        const rsc = rs.?;
        a = .{};
        for ([_]u64{ rsc.z, rsc.rpick, rest.svh_g, rest.svh_u, rest.suh_d, sc.xd + p * pl.ig * 2 }) |v| a.add(v);
        for ([_]usize{ p, nr, 1, e }) |v| i32a(&a, v);
        a.add(limit);
        a.add(@as(c_int, X.act_f32));
        try launch(xk.gateup_epilogue, s, .{ p, nr / 128, 1 }, 32, &a);
    }
}

/// gate / up's epilogue on the main columns: sc.xd [P, ig].
fn mainEpilogue(xk: *const X.Kernels, s: cuda.Stream, ex: weights.Experts, sc: X.Scratch, pick: u64, p: usize, ig: usize, limit: f32) !void {
    var a: cuda.Args = .{};
    for ([_]u64{ sc.z, pick, ex.svh_g, ex.svh_u, ex.suh_d, sc.xd }) |v| a.add(v);
    for ([_]usize{ p, ig, 1, @as(usize, ex.count) }) |v| i32a(&a, v);
    a.add(limit);
    a.add(@as(c_int, X.act_f32));
    try launch(xk.gateup_epilogue, s, .{ p, ig / 128, 1 }, 32, &a);
}

/// exl3_experts2d.down on a prompt chunk: down over the TP2 rank's whole intermediate xd_full [R * slots, I] fp16 for this
/// node's output columns, then the served combine into out fp32 [R, dn]. Uses gateUp's grouping and 64-row work list in sc.
pub fn down(k: *const Kernels, xk: *const X.Kernels, s: cuda.Stream, ex: weights.Experts, sc: X.Scratch, xd_full: u64, pick: u64, wts: u64, out: u64, r: usize) !void {
    const pl = try X2.Plan.of(ex);
    try downGemm(k, s, ex, sc, xd_full, r, .{ .c0 = 0, .cols = pl.dn, .z = sc.z, .compact = false });
    try combineRows(xk, s, ex.svh_d, sc, pick, wts, out, r, pl.dn, @as(usize, ex.count), sc.z, 0, r);
}

/// The columns a down launch covers and where their partials land: [c0, c0 + cols) into z ([P, cols] when compact,
/// else today's [P, N] with every column at its place).
pub const DownCols = struct { c0: usize, cols: usize, z: u64, compact: bool };

/// down's GEMM (no combine) for columns [dc.c0, + dc.cols) of this node's output: each partial is the one today's launch
/// makes for that (member row, column): the same program arithmetic on the same rows and k tiles; only the launch's
/// column blocks (cb0) and, compact, the partials' row stride and first column (z_ld, z_c0) change.
pub fn downGemm(k: *const Kernels, s: cuda.Stream, ex: weights.Experts, sc: X.Scratch, xd_full: u64, r: usize, dc: DownCols) !void {
    const pl = try X2.Plan.of(ex);
    const e: usize = ex.count;
    const slots = sc.slots;
    if (r < X.exact_rows or r > sc.rows) return error.NotAPromptChunk;
    if (dc.cols == 0 or dc.c0 + dc.cols > pl.dn or dc.c0 % 128 != 0 or dc.cols % 128 != 0) return error.UnsupportedShape;
    const p = r * slots;
    const maxu = @min(p, e);
    const maxm = X.memberStride(r);
    const nw = X.listLen(p, maxu, k.rows);
    const ch = try chains(pl.i, pl.cd);
    if (pl.cd[2] != 1) return error.UnsupportedShape;
    var g: Args = .{ .xa = xd_full, .tp0 = ex.down_ptr, .tp1 = ex.down_ptr, .k2_0 = ex.down_k2, .k2_1 = ex.down_k2, .uids = sc.ids, .ucount = sc.count, .members = sc.members, .work = if (k.rows == X.mma_rows_gu) sc.work_gu else sc.work_d, .maxm = @intCast(maxm), .slots = @intCast(slots), .z = dc.z, .K = @intCast(pl.i), .N = @intCast(pl.dn), .P = @intCast(p), .chain = @intCast(ch[0]), .wk = @intCast(ch[1]), .fold = 0 };
    if (dc.compact) {
        g.z_ld = @intCast(dc.cols);
        g.z_c0 = @intCast(dc.c0);
    }
    // (64 rows a program for the routed experts' down measured no faster at 2,048 rows: 2.85 + 0.54 ms against 3.31)
    if (nw > 0) try launchPart(k, s, g, ex.k2_d[0], ex.k2_d[1], dc.cols, false, nw, dc.c0);
}

/// The served combine (down_combine_kernel: one block a row, per row its slots' partials, the Hadamard of each 128-column
/// block, the weights in slot order) for rows [r0, r0 + rows) of the chunk: z [P, d] fp32 partials (d columns), svh
/// [E, d], out [R, d] fp32. Each block reads only its own row's slots and writes only its own row, so a row's bytes do not
/// depend on which rows a launch covers: the launch takes its rows' pick / wts / z / out at their offsets (P only indexes
/// the K splits, of which down has one).
pub fn combineRows(xk: *const X.Kernels, s: cuda.Stream, svh: u64, sc: X.Scratch, pick: u64, wts: u64, out: u64, r: usize, d: usize, e: usize, z: u64, r0: usize, rows: usize) !void {
    if (rows == 0) return;
    if (r0 + rows > r or d % 128 != 0) return error.UnsupportedShape;
    const slots = sc.slots;
    const p = r * slots;
    var a: cuda.Args = .{};
    for ([_]u64{ z + r0 * slots * d * 4, pick + r0 * slots * 4, svh, sc.no_y, wts + r0 * slots * 4, 0, out + r0 * d * 4 }) |v| a.add(v);
    for ([_]usize{ p, d, 1, e, slots, 0 }) |v| i32a(&a, v);
    try launch(xk.down_combine, s, .{ rows, d / 128, 1 }, 32 * slots, &a);
}

test "TF_DS_EXL3_PROMPT: unset is today's path" {
    // (the test process's environment does not set it)
    if (std.c.getenv("TF_DS_EXL3_PROMPT") == null) try std.testing.expect(!try enabled());
    if (std.c.getenv("TF_DS_EXL3_WQ_CACHE") == null) try std.testing.expect(wqCache());
}

test "v2's chains are the served launches'" {
    // gate / up: K 5120 at GLM's (8, 4, 4, 1): 20 k tiles a chain, 4 a split; down: K 1152 at (8, 4, 1, 1): 18, 4
    try std.testing.expectEqual([2]usize{ 20, 4 }, try chains(5120, try X.config(5120, 1152, true)));
    try std.testing.expectEqual([2]usize{ 18, 4 }, try chains(1152, try X.config(1152, 5120, false)));
    try std.testing.expectError(error.UnsupportedShape, chains(1088, .{ 8, 4, 1, 1 }));
}
