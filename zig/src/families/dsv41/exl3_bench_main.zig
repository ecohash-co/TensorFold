//! tf-dsv41-exl3-bench MODEL_DIR CACHE_DIR RANK WORLD KIT_DIR [--layers 3,20,40] [--rows 2048,4096] [--iters 5]
//!   [--x gauss|embed|edge] [--route gate|uniform] [--seed S] [--v2 0|1] [--v2-image FATBIN] [--dense 0|1]:
//! one rank's prompt-chunk EXL3 GEMMs on one GPU with the rank's real weights (the lane's rank cache): a layer's routed
//! experts as a 2D node of the parity split runs them (prompt2d.Two.experts: gate / up and the epilogue, the rest
//! block's, the pack assembled, down and the combine; the column partner's pack stands in as a copy of this node's), and
//! with --dense 1 the layer's dense EXL3 prompt GEMMs (exl3_prefill.matmul's rot_in + unpack; the Triton GEMM is not
//! run here). Each stage is timed with events (min and median over --iters), and with --v2 1 the TF_DS_EXL3_PROMPT=v2
//! kernels (exl3_prompt.zig) run on the same inputs and every output byte is compared with today's (xd's main and rest
//! rows, the assembled intermediate, the combined partial). One JSON line a (layer, rows).
//! Inputs: x bf16 [rows, D] (gauss: N(0, 1) per element; embed: random token embeddings scaled to unit RMS), the
//! routing from the layer's real gate (fp32 logits on the host, top-k of logit + bias; the shared expert in the last
//! slot) or uniform picks. No NCCL, no other node.
const std = @import("std");
const cuda = @import("cuda");
const dsv41 = @import("dsv41");
const X = dsv41.exl3_experts;
const X2 = dsv41.exl3_experts2d;
const P = dsv41.exl3_prompt;

const usage = "usage: tf-dsv41-exl3-bench MODEL_DIR CACHE_DIR RANK WORLD KIT_DIR [--layers 3,20,40] [--rows 2048,4096] [--iters 5] [--x gauss|embed|edge] [--route gate|uniform] [--seed S] [--v2 0|1] [--v2-image FATBIN] [--dense 0|1]\n";

fn readFile(a: std.mem.Allocator, io: std.Io, dir: []const u8, name: []const u8) ![]const u8 {
    const path = try std.fs.path.join(a, &.{ dir, name });
    return std.Io.Dir.cwd().readFileAllocOptions(io, path, a, .limited(1 << 30), .@"16", null); // (module images 16-byte aligned)
}

fn bf16(f: f32) u16 {
    const b: u32 = @bitCast(f);
    const r = b + 0x7fff + ((b >> 16) & 1);
    return @intCast(r >> 16);
}

fn fromBf16(h: u16) f32 {
    return @bitCast(@as(u32, h) << 16);
}

fn fromF16(h: u16) f32 {
    return @floatCast(@as(f16, @bitCast(h)));
}

fn gauss(r: std.Random) f32 {
    // Box-Muller
    const u1_ = @max(r.float(f32), 1e-7);
    const u2_ = r.float(f32);
    return @sqrt(-2.0 * @log(u1_)) * @cos(2.0 * std.math.pi * u2_);
}

const Timer = struct {
    ev: [24]cuda.Event,
    n: usize = 0,
    fn init(d: *const cuda.Driver) !Timer {
        var t: Timer = .{ .ev = undefined };
        for (&t.ev) |*e| e.* = try cuda.Event.init(d, true);
        return t;
    }
    fn mark(t: *Timer, s: cuda.Stream) !void {
        try t.ev[t.n].record(s);
        t.n += 1;
    }
    fn ms(t: *const Timer, i: usize) !f32 {
        return cuda.Event.elapsedMs(t.ev[i], t.ev[i + 1]);
    }
};

fn median(v: []f32) f32 {
    std.mem.sort(f32, v, {}, std.sort.asc(f32));
    return v[v.len / 2];
}

fn alloc(d: *const cuda.Driver, list: *std.ArrayList(cuda.DeviceBuffer), a: std.mem.Allocator, n: usize, fill: ?u8) !u64 {
    var b = try cuda.DeviceBuffer.alloc(d, @max(n, 16));
    if (fill) |v| try b.fill8(v, null);
    try list.append(a, b);
    return b.ptr;
}

fn download(d: *const cuda.Driver, a: std.mem.Allocator, ptr: u64, n: usize) ![]u8 {
    const h = try a.alloc(u8, n);
    try d.check(d.api.cuMemcpyDtoH_v2(h.ptr, ptr, n), "cuMemcpyDtoH");
    return h;
}

fn upload(d: *const cuda.Driver, ptr: u64, bytes: []const u8) !void {
    try d.check(d.api.cuMemcpyHtoD_v2(ptr, bytes.ptr, bytes.len), "cuMemcpyHtoD");
}

/// Bytes that differ between two device buffers (and the first differing byte's index, or null).
fn compare(d: *const cuda.Driver, a: std.mem.Allocator, x: u64, y: u64, n: usize) !struct { diff: usize, first: ?usize } {
    const hx = try download(d, a, x, n);
    defer a.free(hx);
    const hy = try download(d, a, y, n);
    defer a.free(hy);
    var diff: usize = 0;
    var first: ?usize = null;
    for (hx, hy, 0..) |p, q, i| if (p != q) {
        diff += 1;
        if (first == null) first = i;
    };
    return .{ .diff = diff, .first = first };
}

fn launch(f: cuda.Function, s: cuda.Stream, grid: [3]usize, block: usize, args: *cuda.Args) !void {
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(grid[0]), .y = @intCast(grid[1]), .z = @intCast(grid[2]) }, .block = .{ .x = @intCast(block) } }, s, args);
}

fn i32a(a: *cuda.Args, v: usize) void {
    a.add(@as(c_int, @intCast(v)));
}

/// exl3_experts2d.gateUp + gateUpRest + down with a timer mark after each kernel (the same launches): marks
/// [group(3), rot_in, mma, epilogue, rest group(4), rest mma, rest epilogue, assemble, down, combine].
const detail_names = [_][]const u8{ "group", "rot_in", "gu_mma", "gu_epi", "rest_group", "rest_mma", "rest_epi", "assemble", "down_rows", "combine" };
fn oldDetail(k: *const X.Kernels, o: *const dsv41.ops.Ops, s: cuda.Stream, tm: *Timer, ex: dsv41.weights.Experts, sc: X.Scratch, rs: X2.RestScratch, x: u64, x_stride: usize, pick: u64, wts: u64, xd_full: u64, out: u64, r: usize, limit: f32) !void {
    const pl = try X2.Plan.of(ex);
    const e: usize = ex.count;
    const slots = sc.slots;
    const p = r * slots;
    const maxu = @min(p, e);
    const maxm = X.memberStride(r);
    const n_gu = X.listLen(p, maxu, X.mma_rows_gu);
    const n_d = X.listLen(p, maxu, X.rows_rows_d);
    const mma = k.mma[X.range(ex.k2_gu[0], ex.k2_gu[1])] orelse return error.MissingKernel;
    try tm.mark(s);
    var a: cuda.Args = .{};
    a.add(pick);
    a.add(sc.counts);
    i32a(&a, p);
    try launch(k.group_count, s, .{ e, 1, 1 }, X.place_threads, &a);
    a = .{};
    for ([_]u64{ pick, sc.counts, sc.ids, sc.count, sc.members }) |v| a.add(v);
    for ([_]usize{ p, slots, e, maxm }) |v| i32a(&a, v);
    try launch(k.group_place, s, .{ e, 1, 1 }, X.place_threads, &a);
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
    try launch(k.work_list, s, .{ 1, 1, 1 }, X.list_threads, &a);
    try tm.mark(s);
    a = .{};
    a.add(x);
    i32a(&a, x_stride);
    for ([_]u64{ pick, ex.suh_g, ex.suh_u, sc.xg, sc.xu }) |v| a.add(v);
    for ([_]usize{ pl.d, slots, e }) |v| i32a(&a, v);
    try launch(k.rot_in, s, .{ p, pl.d / 128, 2 }, 32, &a);
    try tm.mark(s);
    a = .{};
    for ([_]u64{ sc.xg, sc.xu, ex.gate_ptr, ex.up_ptr, ex.gate_k2, ex.up_k2, sc.ids, sc.count, sc.members, sc.z }) |v| a.add(v);
    for ([_]usize{ pl.d, pl.ig, p, pl.cgu[2], pl.cgu[1], maxm, slots, n_gu }) |v| i32a(&a, v);
    a.add(sc.work_gu);
    if (n_gu > 0) try launch(mma, s, .{ 1, pl.ig / X.mma_cols, n_gu * 2 }, 256, &a);
    try tm.mark(s);
    a = .{};
    for ([_]u64{ sc.z, pick, ex.svh_g, ex.svh_u, ex.suh_d, sc.xd }) |v| a.add(v);
    for ([_]usize{ p, pl.ig, 1, e }) |v| i32a(&a, v);
    a.add(limit);
    a.add(@as(c_int, X.act_f32));
    try launch(k.gateup_epilogue, s, .{ p, pl.ig / 128, 1 }, 32, &a);
    try tm.mark(s);
    // the rest (exl3_experts2d.gateUpRest)
    const rest = ex.rest.?;
    const irr: usize = rest.width;
    const rmma = k.mma[X.range(rest.k2_gu[0], rest.k2_gu[1])] orelse return error.MissingKernel;
    try o.remapPicks(s, pick, rest.remap, rs.rpick, p);
    a = .{};
    a.add(rs.rpick);
    a.add(rs.counts);
    i32a(&a, p);
    try launch(k.group_count, s, .{ e, 1, 1 }, X.place_threads, &a);
    a = .{};
    for ([_]u64{ rs.rpick, rs.counts, rs.ids, rs.count, rs.members }) |v| a.add(v);
    for ([_]usize{ p, slots, e, maxm }) |v| i32a(&a, v);
    try launch(k.group_place, s, .{ e, 1, 1 }, X.place_threads, &a);
    a = .{};
    for ([_]u64{ rs.counts, rs.ids, rs.count }) |v| a.add(v);
    i32a(&a, maxu);
    a.add(rs.work_gu);
    i32a(&a, X.mma_rows_gu);
    i32a(&a, n_gu);
    a.add(rs.work_d);
    i32a(&a, X.rows_rows_d);
    i32a(&a, n_d);
    a.add(@as(c_int, -1));
    try launch(k.work_list, s, .{ 1, 1, 1 }, X.list_threads, &a);
    try tm.mark(s);
    a = .{};
    for ([_]u64{ sc.xg, sc.xu, rest.gate_ptr, rest.up_ptr, rest.gate_k2, rest.up_k2, rs.ids, rs.count, rs.members, rs.z }) |v| a.add(v);
    for ([_]usize{ pl.d, irr, p, pl.cgu[2], pl.cgu[1], maxm, slots, n_gu }) |v| i32a(&a, v);
    a.add(rs.work_gu);
    if (n_gu > 0) try launch(rmma, s, .{ 1, irr / X.mma_cols, n_gu * 2 }, 256, &a);
    try tm.mark(s);
    a = .{};
    for ([_]u64{ rs.z, rs.rpick, rest.svh_g, rest.svh_u, rest.suh_d, sc.xd + p * pl.ig * 2 }) |v| a.add(v);
    for ([_]usize{ p, irr, 1, e }) |v| i32a(&a, v);
    a.add(limit);
    a.add(@as(c_int, X.act_f32));
    try launch(k.gateup_epilogue, s, .{ p, irr / 128, 1 }, 32, &a);
    try tm.mark(s);
    try o.assembleRest(s, sc.xd, sc.xd, pick, rest.owner, xd_full, p, pl.ig, irr);
    try tm.mark(s);
    const dfn = k.rows[X.range(ex.k2_d[0], ex.k2_d[1])] orelse return error.MissingKernel;
    a = .{};
    for ([_]u64{ xd_full, xd_full, ex.down_ptr, ex.down_ptr, ex.down_k2, ex.down_k2, sc.ids, sc.count, sc.members, sc.z }) |v| a.add(v);
    for ([_]usize{ pl.i, pl.dn, p, 1, maxm, slots, n_d }) |v| i32a(&a, v);
    a.add(sc.work_d);
    if (n_d > 0) try launch(dfn, s, .{ 1, pl.dn / 128, n_d }, 4 * 32, &a);
    try tm.mark(s);
    a = .{};
    for ([_]u64{ sc.z, pick, ex.svh_d, sc.no_y, wts, 0, out }) |v| a.add(v);
    for ([_]usize{ p, pl.dn, 1, e, slots, 0 }) |v| i32a(&a, v);
    try launch(k.down_combine, s, .{ r, pl.dn / 128, 1 }, 32 * slots, &a);
    try tm.mark(s);
}

/// --dense 1: a layer's dense EXL3 prompt GEMMs (exl3_prefill.matmul: rot_in, unpack, the Triton GEMM) at `n` rows,
/// each stage timed, and the GEMM again on a W_q unpacked once beforehand (TF_DS_EXL3_PROMPT=v2's cache) with its
/// output compared byte for byte.
fn denseLayer(d: *const cuda.Driver, a: std.mem.Allocator, out: anytype, s: cuda.Stream, tm: *Timer, pf: *const dsv41.exl3_prefill.Kernels, t: dsv41.tri.Tri, ws: dsv41.exl3_prefill.Workspace, wcache: u64, xin: u64, o1: u64, o2: u64, lay: dsv41.weights.Layer, li: usize, n: usize, iters: usize) !void {
    const L = dsv41.weights.Linear;
    const OT = dsv41.tri_markov.OutType;
    var ls: [16]L = undefined;
    var ots: [16]OT = undefined;
    var nl: usize = 0;
    const add = struct {
        fn f(lsp: *[16]L, otp: *[16]OT, k: *usize, l: L, o: OT) void {
            lsp[k.*] = l;
            otp[k.*] = o;
            k.* += 1;
        }
    }.f;
    add(&ls, &ots, &nl, lay.wq_a, .bf16);
    add(&ls, &ots, &nl, lay.wkv, .bf16);
    add(&ls, &ots, &nl, lay.wq_b, .bf16);
    for (lay.wo_a[0..lay.groups]) |wo| add(&ls, &ots, &nl, wo, .bf16);
    add(&ls, &ots, &nl, lay.wo_b, .fp32);
    if (lay.comp_wkv) |l| add(&ls, &ots, &nl, l, .fp32);
    if (lay.comp_wgate) |l| add(&ls, &ots, &nl, l, .fp32);
    if (lay.idx_wq_b) |l| add(&ls, &ots, &nl, l, .bf16);
    if (lay.idx_wk) |l| add(&ls, &ots, &nl, l, .bf16);
    var t_rot: f32 = 0;
    var t_unp: f32 = 0;
    var t_gemm: f32 = 0;
    var t_gemm2: f32 = 0;
    var wq_bytes: usize = 0;
    var bad: usize = 0;
    var skipped: usize = 0;
    for (ls[0..nl], ots[0..nl]) |l, ot| {
        const k: usize = l.k;
        const nn: usize = l.n;
        wq_bytes += k * nn * 2;
        // the cache: this linear's W_q unpacked once (its own launch, before the timed ones)
        dsv41.exl3_prefill.unpack(pf, s, l, wcache) catch |err| {
            std.debug.print("layer {d} unpack k {d} n {d} k2 {d}: {s}\n", .{ li, k, nn, l.k2, @errorName(err) });
            return err;
        };
        var rr = try a.alloc(f32, iters);
        var uu = try a.alloc(f32, iters);
        var gg = try a.alloc(f32, iters);
        var g2 = try a.alloc(f32, iters);
        var ok = true;
        for (0..iters + 1) |it| {
            tm.n = 0;
            try tm.mark(s);
            dsv41.exl3_prefill.rotIn(pf, s, xin, .bf16, k, l.suh, ws.xh, n, k) catch |err| {
                std.debug.print("layer {d} rot_in k {d}: {s}\n", .{ li, k, @errorName(err) });
                return err;
            };
            try tm.mark(s);
            try dsv41.exl3_prefill.unpack(pf, s, l, ws.w);
            try tm.mark(s);
            dsv41.tri_markov.matmul(t, ws.xh, ws.w, ws.h, l.svh, null, o1, ot, nn, n, k, nn) catch |err| {
                std.debug.print("layer {d} linear k {d} n {d}: {s}\n", .{ li, k, nn, @errorName(err) });
                ok = false;
                break;
            };
            try tm.mark(s);
            try dsv41.tri_markov.matmul(t, ws.xh, wcache, ws.h, l.svh, null, o2, ot, nn, n, k, nn);
            try tm.mark(s);
            try s.synchronize();
            if (it > 0) {
                rr[it - 1] = try tm.ms(0);
                uu[it - 1] = try tm.ms(1);
                gg[it - 1] = try tm.ms(2);
                g2[it - 1] = try tm.ms(3);
            }
        }
        if (!ok) {
            skipped += 1;
            try s.synchronize();
            continue;
        }
        const ob: usize = if (ot == .fp32) 4 else 2;
        const c = try compare(d, a, o1, o2, n * nn * ob);
        const cw = try compare(d, a, ws.w, wcache, k * nn * 2);
        bad += c.diff + cw.diff;
        t_rot += median(rr);
        t_unp += median(uu);
        t_gemm += median(gg);
        t_gemm2 += median(g2);
    }
    const fnn: f32 = @floatFromInt(n);
    try out.interface.print("{{\"layer\": {d}, \"rows\": {d}, \"path\": \"dense\", \"linears\": {d}, \"skipped\": {d}, \"wq_MB\": {d:.1}, \"ms\": {{\"rot_in\": {d:.3}, \"unpack\": {d:.3}, \"gemm\": {d:.3}, \"gemm_cached\": {d:.3}}}, \"us_row\": {{\"today\": {d:.3}, \"cached\": {d:.3}}}, \"diff_bytes\": {d}}}\n", .{ li, n, nl, skipped, @as(f64, @floatFromInt(wq_bytes)) / 1e6, t_rot, t_unp, t_gemm, t_gemm2, (t_rot + t_unp + t_gemm) * 1000 / fnn, (t_rot + t_gemm2) * 1000 / fnn, bad });
    try out.interface.flush();
}

/// TF_BENCH_XOVL: the expert-side TF_DS_PROMPT_OVL plans (pack, dnc, dn) against v2's whole launches on the same inputs
/// (xd_full2 / pm2 hold v2's results on entry), and their launch cost (no exchange: one GPU). One JSON line.
fn xovlCheck(d: *const cuda.Driver, a: std.mem.Allocator, out: anytype, s: cuda.Stream, tm: *Timer, k: *const P.Kernels, xk: *const X.Kernels, ops: *const dsv41.ops.Ops, ex: dsv41.weights.Experts, sc2: X.Scratch, rs2: ?X2.RestScratch, xin: u64, dd: usize, pick: u64, wts: u64, xd_full2: u64, pm2: u64, ref_xd: u64, save_xd: u64, pm3: u64, pmh: u64, svh_h: u64, n: usize, sl: usize, ig: usize, ir: usize, dn: usize, ne: usize, rank: u32, iters: usize, limit: f32, li: usize) !void {
    const ovl = dsv41.ovl;
    const p = n * sl;
    const half = dn / 2;
    // v2's whole gate / up -> ref_xd, then pack's split order into sc2.xd, each from a zeroed sc2.xd (the rest rows
    // of the other pair's slots are written by neither); sc2.xd is put back as it was at the end (the v2-vs-today
    // compare of later cases relies on both scratches' never-written rows having the same history)
    const xd_bytes = p * (ig + ir) * 2;
    try d.check(d.api.cuMemcpyDtoDAsync_v2(save_xd, sc2.xd, xd_bytes, s.handle), "cuMemcpyDtoDAsync");
    try d.check(d.api.cuMemsetD8Async(sc2.xd, 0, xd_bytes, s.handle), "cuMemsetD8Async");
    try P.gateUp(k, xk, ops, s, ex, sc2, rs2, xin, dd, pick, n, limit);
    try d.check(d.api.cuMemcpyDtoDAsync_v2(ref_xd, sc2.xd, p * (ig + ir) * 2, s.handle), "cuMemcpyDtoDAsync");
    // (zero: the rest rows of the other pair's slots are never written, so they stay zero as in ref_xd)
    try d.check(d.api.cuMemsetD8Async(sc2.xd, 0, p * (ig + ir) * 2, s.handle), "cuMemsetD8Async");
    const rest_ok = ex.rest != null;
    if (rest_ok) {
        try P.gateUpPart(k, xk, ops, s, ex, sc2, rs2, xin, dd, pick, n, limit, .main);
        try P.gateUpPart(k, xk, ops, s, ex, sc2, rs2, xin, dd, pick, n, limit, .rest);
    } else try P.gateUp(k, xk, ops, s, ex, sc2, rs2, xin, dd, pick, n, limit);
    try s.synchronize();
    const c_pack = try compare(d, a, ref_xd, sc2.xd, p * (ig + ir) * 2);
    // the down GEMM below reads xd_full2 (v2's assembled intermediate) and sc2's grouping (just rebuilt identically)
    const o = dsv41.prompt2d.Owners.of(n);
    const rp = ovl.othersFirst(o, rank) orelse return error.BadOwnerRows;
    // dnc: today's down GEMM, the combine in three row parts
    try d.check(d.api.cuMemsetD8Async(pm3, 0xa5, n * dn * 4, s.handle), "cuMemsetD8Async");
    try P.downGemm(k, s, ex, sc2, xd_full2, n, .{ .c0 = 0, .cols = dn, .z = sc2.z, .compact = false });
    for (rp) |r| try P.combineRows(xk, s, ex.svh_d, sc2, pick, wts, pm3, n, dn, ne, sc2.z, r.r0, r.rows);
    try s.synchronize();
    const c_dnc = try compare(d, a, pm2, pm3, n * dn * 4);
    // dn: svh_d's halves (model.dnTables), each half's GEMM into a compact part of sc2.z and its combine (others, own)
    for (0..2) |h| try ops.copyRows(s, ex.svh_d + h * half * 2, dn * 2, svh_h + h * ne * half * 2, half * 2, half * 2, ne);
    try d.check(d.api.cuMemsetD8Async(pmh, 0xa5, n * dn * 4, s.handle), "cuMemsetD8Async");
    try d.check(d.api.cuMemsetD8Async(pm3, 0x5a, n * dn * 4, s.handle), "cuMemsetD8Async");
    try d.check(d.api.cuMemsetD8Async(sc2.z, 0x5a, p * dn * 4, s.handle), "cuMemsetD8Async");
    for (0..2) |h| {
        const z = sc2.z + h * p * half * 4;
        try P.downGemm(k, s, ex, sc2, xd_full2, n, .{ .c0 = h * half, .cols = half, .z = z, .compact = true });
        for (rp[0..2]) |r| try P.combineRows(xk, s, svh_h + h * ne * half * 2, sc2, pick, wts, pmh + h * n * half * 4, n, half, ne, z, r.r0, r.rows);
    }
    for (0..2) |h| try P.combineRows(xk, s, svh_h + h * ne * half * 2, sc2, pick, wts, pmh + h * n * half * 4, n, half, ne, sc2.z + h * p * half * 4, rp[2].r0, rp[2].rows);
    // half-major -> row-major (what gather's copies do across the nodes)
    for (0..2) |h| try ops.copyRows(s, pmh + h * n * half * 4, half * 4, pm3 + h * half * 4, dn * 4, half * 4, n);
    try s.synchronize();
    const c_dn = try compare(d, a, pm2, pm3, n * dn * 4);
    // launch cost on one GPU: v2's gate / up whole vs pack's two parts; v2's down + combine vs dnc's vs dn's
    var t: [5][]f32 = undefined;
    for (&t) |*x| x.* = try a.alloc(f32, iters);
    for (0..iters + 1) |it| {
        tm.n = 0;
        try tm.mark(s);
        try P.gateUp(k, xk, ops, s, ex, sc2, rs2, xin, dd, pick, n, limit);
        try tm.mark(s);
        if (rest_ok) {
            try P.gateUpPart(k, xk, ops, s, ex, sc2, rs2, xin, dd, pick, n, limit, .main);
            try P.gateUpPart(k, xk, ops, s, ex, sc2, rs2, xin, dd, pick, n, limit, .rest);
        }
        try tm.mark(s);
        try P.down(k, xk, s, ex, sc2, xd_full2, pick, wts, pm3, n);
        try tm.mark(s);
        try P.downGemm(k, s, ex, sc2, xd_full2, n, .{ .c0 = 0, .cols = dn, .z = sc2.z, .compact = false });
        for (rp) |r| try P.combineRows(xk, s, ex.svh_d, sc2, pick, wts, pm3, n, dn, ne, sc2.z, r.r0, r.rows);
        try tm.mark(s);
        for (0..2) |h| {
            const z = sc2.z + h * p * half * 4;
            try P.downGemm(k, s, ex, sc2, xd_full2, n, .{ .c0 = h * half, .cols = half, .z = z, .compact = true });
            for (rp) |r| try P.combineRows(xk, s, svh_h + h * ne * half * 2, sc2, pick, wts, pmh + h * n * half * 4, n, half, ne, z, r.r0, r.rows);
        }
        try tm.mark(s);
        try s.synchronize();
        if (it > 0) for (0..5) |j| {
            t[j][it - 1] = try tm.ms(j);
        };
    }
    var m: [5]f32 = undefined;
    for (0..5) |j| m[j] = median(t[j]);
    // the rest block's share of gate / up (pack's hiding window) on its own
    var tr = try a.alloc(f32, iters);
    if (rest_ok) for (0..iters + 1) |it| {
        try P.gateUpPart(k, xk, ops, s, ex, sc2, rs2, xin, dd, pick, n, limit, .main);
        tm.n = 0;
        try tm.mark(s);
        try P.gateUpPart(k, xk, ops, s, ex, sc2, rs2, xin, dd, pick, n, limit, .rest);
        try tm.mark(s);
        try s.synchronize();
        if (it > 0) tr[it - 1] = try tm.ms(0);
    };
    const m_rest = if (rest_ok) median(tr) else 0;
    const equal = c_pack.diff == 0 and c_dnc.diff == 0 and c_dn.diff == 0;
    const fnn: f32 = @floatFromInt(n);
    try d.check(d.api.cuMemcpyDtoDAsync_v2(sc2.xd, save_xd, xd_bytes, s.handle), "cuMemcpyDtoDAsync");
    try s.synchronize();
    try out.interface.print("{{\"layer\": {d}, \"rows\": {d}, \"path\": \"xovl\", \"equal\": {}, \"diff_bytes\": {{\"pack_xd\": {d}, \"dnc_out\": {d}, \"dn_out\": {d}}}, \"ms\": {{\"gateup\": {d:.3}, \"gateup_pack\": {d:.3}, \"rest_part\": {d:.3}, \"down\": {d:.3}, \"down_dnc\": {d:.3}, \"down_dn\": {d:.3}}}, \"us_row\": {{\"pack_extra\": {d:.3}, \"rest_window\": {d:.3}, \"dnc_extra\": {d:.3}, \"dn_extra\": {d:.3}, \"down\": {d:.3}}}}}\n", .{ li, n, equal, c_pack.diff, c_dnc.diff, c_dn.diff, m[0], m[1], m_rest, m[2], m[3], m[4], (m[1] - m[0]) * 1000 / fnn, m_rest * 1000 / fnn, (m[3] - m[2]) * 1000 / fnn, (m[4] - m[2]) * 1000 / fnn, m[2] * 1000 / fnn });
    try out.interface.flush();
}

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    if (args.len < 6) {
        std.debug.print("{s}", .{usage});
        return 2;
    }
    const rank = try std.fmt.parseInt(u32, args[3], 10);
    const world = try std.fmt.parseInt(u32, args[4], 10);
    const kit = args[5];
    var layers_text: []const u8 = "3,20,40";
    var rows_text: []const u8 = "2048,4096";
    var iters: usize = 5;
    var x_mode: []const u8 = "gauss";
    var route_mode: []const u8 = "gate";
    var seed: u64 = 1;
    var v2 = false;
    var v2_image: ?[]const u8 = null;
    var dense = false;
    var i: usize = 6;
    while (i + 1 < args.len) : (i += 2) {
        const k = args[i];
        const v = args[i + 1];
        if (std.mem.eql(u8, k, "--layers")) layers_text = v else if (std.mem.eql(u8, k, "--rows")) rows_text = v else if (std.mem.eql(u8, k, "--iters")) iters = try std.fmt.parseInt(usize, v, 10) else if (std.mem.eql(u8, k, "--x")) x_mode = v else if (std.mem.eql(u8, k, "--route")) route_mode = v else if (std.mem.eql(u8, k, "--seed")) seed = try std.fmt.parseInt(u64, v, 10) else if (std.mem.eql(u8, k, "--v2")) v2 = !std.mem.eql(u8, v, "0") else if (std.mem.eql(u8, k, "--v2-image")) v2_image = v else if (std.mem.eql(u8, k, "--dense")) dense = !std.mem.eql(u8, v, "0") else {
            std.debug.print("{s}", .{usage});
            return 2;
        }
    }
    var layer_ids: std.ArrayList(usize) = .empty;
    var lit = std.mem.splitScalar(u8, layers_text, ',');
    while (lit.next()) |t| try layer_ids.append(a, try std.fmt.parseInt(usize, t, 10));
    var rows_list: std.ArrayList(usize) = .empty;
    var rit = std.mem.splitScalar(u8, rows_text, ',');
    while (rit.next()) |t| try rows_list.append(a, try std.fmt.parseInt(usize, t, 10));
    var cap: usize = 64;
    for (rows_list.items) |n| cap = @max(cap, n);
    var top: usize = 0;
    for (layer_ids.items) |l| top = @max(top, l);

    var cfg = try dsv41.Config.read(a, io, args[1]);
    if (top >= cfg.layers) return error.NoSuchLayer;
    cfg.layers = top + 1; // load layers 0 .. top only
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    const d = &driver;

    var sp = try dsv41.prompt2d.split(rank, world);
    sp.parity = sp.pair != null; // the served TF_DS_2D_GU=parity
    const t0 = std.Io.Timestamp.now(io, .awake);
    var cdir = try std.Io.Dir.cwd().openDir(io, args[2], .{ .iterate = true });
    defer cdir.close(io);
    const cname = try dsv41.rank_cache.find(a, io, cdir, rank, world);
    var ix = try dsv41.rank_cache.open(init.gpa, io, cdir, cname);
    defer ix.deinit();
    var file = try cdir.openFile(io, cname, .{});
    defer file.close(io);
    var w = try dsv41.weights.load(init.gpa, d, .{ .cache = .{ .file = file, .index = &ix, .io = io } }, cfg, sp, false);
    defer w.deinit();
    const load_s = @as(f64, @floatFromInt(t0.durationTo(std.Io.Timestamp.now(io, .awake)).nanoseconds)) / 1e9;

    var out_buf: [1 << 14]u8 = undefined;
    var out = std.Io.File.stdout().writer(io, &out_buf);
    try out.interface.print("{{\"load_s\": {d:.1}, \"layers\": {d}, \"rank\": {d}, \"world\": {d}}}\n", .{ load_s, cfg.layers, rank, world });
    try out.interface.flush();

    var xk = try X.Kernels.load(d, try readFile(a, io, kit, "cubins/experts.cubin"), try readFile(a, io, kit, "cubins/experts_cb.cubin"), dsv41.exl3_linear.codebook_mul1);
    defer xk.unload();
    var ops = try dsv41.ops.Ops.load(d, cuda.kernels.torch_pointwise, cuda.kernels.torch_movement, cuda.kernels.dsv41_ops);
    defer ops.unload();
    var pk: ?P.Kernels = null;
    if (v2) pk = try P.Kernels.load(d, if (v2_image) |path| try std.Io.Dir.cwd().readFileAllocOptions(io, path, a, .limited(1 << 28), .@"16", null) else cuda.kernels.dsv41_exl3_prompt);
    defer if (pk) |*k| k.unload();

    var s = try cuda.Stream.init(d, true);
    defer s.deinit();
    var aot_set: ?cuda.aot.Set = null;
    var pfk: ?dsv41.exl3_prefill.Kernels = null;
    if (dense) {
        aot_set = cuda.aot.Set.load(init.gpa, io, d, ctx.device, try std.fs.path.join(a, &.{ kit, "aot" })) catch |err| {
            std.debug.print("aot set: {s}\n", .{@errorName(err)});
            return err;
        };
        pfk = dsv41.exl3_prefill.Kernels.load(d, try readFile(a, io, kit, "cubins/linear.cubin")) catch |err| {
            std.debug.print("linear cubin: {s}\n", .{@errorName(err)});
            return err;
        };
    }
    var tm = try Timer.init(d);

    const ex0 = w.layers[layer_ids.items[0]].experts;
    const sl = cfg.top_k + 1;
    const dd: usize = cfg.hidden;
    const e_count: usize = ex0.count;
    const ig: usize = ex0.width;
    const ir: usize = if (ex0.rest) |r| r.width else 0;
    const ifull: usize = ex0.down_k;
    const dn: usize = ex0.down_n;

    var bufs: std.ArrayList(cuda.DeviceBuffer) = .empty;
    defer for (bufs.items) |*b| b.free();
    // today's scratch (prompt.zig: exl3_experts.Scratch at the pack's width; prompt2d: the rest's, xd_full)
    const sz = X.Scratch.sizes(cap, sl, dd, ig + ir, e_count);
    var sc: X.Scratch = undefined;
    sc.rows = cap;
    sc.slots = sl;
    inline for (.{ "xg", "xu", "xd", "z", "no_y", "ids", "count", "counts", "members", "work_gu", "work_d" }, 0..) |f, j| {
        @field(sc, f) = try alloc(d, &bufs, a, sz[j], if (j == 8) 0xff else 0);
    }
    var rs: ?X2.RestScratch = null;
    if (ex0.rest != null) {
        const rsz = try X2.RestScratch.sizes(cap, sl, ex0, true);
        var r: X2.RestScratch = undefined;
        inline for (.{ "ids", "count", "counts", "members", "cnt_gu", "z", "work_gu", "work_d", "rpick" }, 0..) |f, j| @field(r, f) = try alloc(d, &bufs, a, rsz[j], if (j == 3) 0xff else 0);
        rs = r;
    }
    const xd_full = try alloc(d, &bufs, a, cap * sl * ifull * 2, 0);
    const pm = try alloc(d, &bufs, a, cap * dn * 4, 0);
    // inputs
    const xin = try alloc(d, &bufs, a, cap * dd * 2, 0);
    const pick = try alloc(d, &bufs, a, cap * sl * 4, 0);
    const wts = try alloc(d, &bufs, a, cap * sl * 4, 0);
    // v2's own copies of today's scratch (the same sizes and initial contents) and outputs
    var sc2: X.Scratch = undefined;
    var rs2: ?X2.RestScratch = null;
    var xd_full2: u64 = 0;
    var pm2: u64 = 0;
    if (pk != null) {
        sc2.rows = cap;
        sc2.slots = sl;
        inline for (.{ "xg", "xu", "xd", "z", "no_y", "ids", "count", "counts", "members", "work_gu", "work_d" }, 0..) |f, j| {
            @field(sc2, f) = try alloc(d, &bufs, a, if (j < 2) 16 else sz[j], if (j == 8) 0xff else 0); // no xg / xu
        }
        if (ex0.rest != null) {
            const rsz = try X2.RestScratch.sizes(cap, sl, ex0, true);
            var r: X2.RestScratch = undefined;
            inline for (.{ "ids", "count", "counts", "members", "cnt_gu", "z", "work_gu", "work_d", "rpick" }, 0..) |f, j| @field(r, f) = try alloc(d, &bufs, a, rsz[j], if (j == 3) 0xff else 0);
            rs2 = r;
        }
        xd_full2 = try alloc(d, &bufs, a, cap * sl * ifull * 2, 0);
        pm2 = try alloc(d, &bufs, a, cap * dn * 4, 0);
    }
    // TF_BENCH_XOVL=1 (with --v2 1; stage ZX1): TF_DS_PROMPT_OVL's expert-side launch plans on the same inputs, every
    // byte against v2's whole launches: pack's split gate / up (gateUpPart .main then .rest), dnc's row-split combine,
    // dn's column-half down GEMM (compact partials) and combine (svh_d halves, half-major out, put back row-major here)
    const xovl = pk != null and std.c.getenv("TF_BENCH_XOVL") != null;
    var ref_xd: u64 = 0;
    var save_xd: u64 = 0;
    var pm3: u64 = 0;
    var pmh: u64 = 0;
    var svh_h: u64 = 0;
    if (xovl) {
        ref_xd = try alloc(d, &bufs, a, cap * sl * (ig + ir) * 2, 0);
        save_xd = try alloc(d, &bufs, a, cap * sl * (ig + ir) * 2, 0);
        pm3 = try alloc(d, &bufs, a, cap * dn * 4, 0);
        pmh = try alloc(d, &bufs, a, cap * dn * 4, 0);
        svh_h = try alloc(d, &bufs, a, 2 * e_count * (dn / 2) * 2, 0);
    }

    var prng = std.Random.DefaultPrng.init(seed);
    const rnd = prng.random();
    const xh = try a.alloc(u16, cap * dd);
    const xf = try a.alloc(f32, cap * dd);
    if (std.mem.eql(u8, x_mode, "embed")) {
        const row = try a.alloc(u16, dd);
        for (0..cap) |r| {
            const tok = rnd.uintLessThan(usize, 120000);
            try d.check(d.api.cuMemcpyDtoH_v2(@ptrCast(row.ptr), w.embed + tok * dd * 2, dd * 2), "cuMemcpyDtoH");
            var ss: f32 = 0;
            for (row) |h| ss += fromBf16(h) * fromBf16(h);
            const sc_ = 1.0 / @sqrt(ss / @as(f32, @floatFromInt(dd)) + 1e-12);
            for (0..dd) |c| xh[r * dd + c] = bf16(fromBf16(row[c]) * sc_);
        }
    } else if (std.mem.eql(u8, x_mode, "edge")) {
        // adversarial rows: a third x64 (fp16 overflow after the rotation), zeros and -0 (and runs of them inside rows),
        // tiny values (bf16 subnormals), the rest N(0, 1)
        for (0..cap) |r| {
            for (0..dd) |c| {
                const g0 = gauss(rnd);
                xh[r * dd + c] = switch ((r + c / 128) % 6) {
                    0, 1 => bf16(g0 * 4096.0),
                    2 => if (c % 2 == 0) 0x8000 else 0x0000,
                    3 => bf16(g0 * 1e-39),
                    else => bf16(g0),
                };
            }
        }
    } else {
        for (xh) |*h| h.* = bf16(gauss(rnd));
    }
    for (xh, xf) |h, *f| f.* = fromBf16(h);
    try upload(d, xin, std.mem.sliceAsBytes(xh));

    const pick_h = try a.alloc(i32, cap * sl);
    const wts_h = try a.alloc(f32, cap * sl);
    const gw = try a.alloc(f32, cfg.experts * dd);
    const gb = try a.alloc(f32, cfg.experts);
    const logit = try a.alloc(f32, cfg.experts);

    for (layer_ids.items) |li| {
        const lay = w.layers[li];
        const ex = lay.experts;
        // the routing for this layer (every row count takes a prefix of the cap rows)
        if (std.mem.eql(u8, route_mode, "gate")) {
            const gwh = try download(d, a, lay.gate_w, cfg.experts * dd * 2);
            defer a.free(gwh);
            const g16 = std.mem.bytesAsSlice(u16, gwh);
            for (gw, 0..) |*f, j| f.* = fromF16(g16[j]);
            const gbh = try download(d, a, lay.gate_b, cfg.experts * 4);
            defer a.free(gbh);
            @memcpy(std.mem.sliceAsBytes(gb), gbh);
        }
        var counts = try a.alloc(u32, e_count);
        defer a.free(counts);
        @memset(counts, 0);
        for (0..cap) |r| {
            var chosen: [32]usize = undefined;
            const kk = cfg.top_k;
            if (std.mem.eql(u8, route_mode, "gate")) {
                const xr = xf[r * dd ..][0..dd];
                const V = @Vector(16, f32);
                for (0..cfg.experts) |e| {
                    const g = gw[e * dd ..][0..dd];
                    var acc: V = @splat(0);
                    var c: usize = 0;
                    while (c < dd) : (c += 16) acc += @as(V, xr[c..][0..16].*) * @as(V, g[c..][0..16].*);
                    logit[e] = @reduce(.Add, acc);
                }
                for (0..kk) |q| {
                    var best: usize = 0;
                    var bv: f32 = -std.math.inf(f32);
                    for (0..cfg.experts) |e| {
                        var taken = false;
                        for (chosen[0..q]) |c| taken = taken or c == e;
                        const sv = logit[e] + gb[e];
                        if (!taken and sv > bv) {
                            bv = sv;
                            best = e;
                        }
                    }
                    chosen[q] = best;
                }
            } else {
                for (0..kk) |q| {
                    while (true) {
                        const e = rnd.uintLessThan(usize, cfg.experts);
                        var taken = false;
                        for (chosen[0..q]) |c| taken = taken or c == e;
                        if (!taken) {
                            chosen[q] = e;
                            break;
                        }
                    }
                }
            }
            var tot: f32 = 0;
            for (0..kk) |q| tot += 1.0 / (1.0 + @exp(-(if (std.mem.eql(u8, route_mode, "gate")) logit[chosen[q]] else 0)));
            for (0..kk) |q| {
                pick_h[r * sl + q] = @intCast(chosen[q]);
                const sg = 1.0 / (1.0 + @exp(-(if (std.mem.eql(u8, route_mode, "gate")) logit[chosen[q]] else 0)));
                wts_h[r * sl + q] = sg / tot * cfg.routed_scaling;
                counts[chosen[q]] += 1;
            }
            pick_h[r * sl + sl - 1] = @intCast(e_count - 1);
            wts_h[r * sl + sl - 1] = 1.0;
        }
        try upload(d, pick, std.mem.sliceAsBytes(pick_h));
        try upload(d, wts, std.mem.sliceAsBytes(wts_h));
        var busiest: u32 = 0;
        var used: usize = 0;
        for (counts[0 .. e_count - 1]) |c| {
            busiest = @max(busiest, c);
            used += @intFromBool(c > 0);
        }
        // the layer's trellis bytes (DRAM floor): gate / up / down of every expert, and the rest block's
        var tbytes: usize = 0;
        var rbytes: usize = 0;
        var nb: [128]u8 = undefined;
        const tname = try std.fmt.bufPrint(&nb, "layers.{d}.ffn.experts.trellis", .{li});
        for (w.named.items) |nm| if (std.mem.eql(u8, nm.name, tname)) {
            tbytes = nm.len;
        };
        const rname = try std.fmt.bufPrint(&nb, "layers.{d}.ffn.experts.rest_trellis", .{li});
        for (w.named.items) |nm| if (std.mem.eql(u8, nm.name, rname)) {
            rbytes = nm.len;
        };

        if (dense) {
            var maxk: usize = 0;
            var maxkn: usize = 0;
            var maxn: usize = 0;
            for ([_]?dsv41.weights.Linear{ lay.wq_a, lay.wkv, lay.wq_b, lay.wo_a[0], lay.wo_b, lay.comp_wkv, lay.comp_wgate, lay.idx_wq_b, lay.idx_wk }) |ol| if (ol) |l| {
                maxk = @max(maxk, l.k);
                maxn = @max(maxn, l.n);
                maxkn = @max(maxkn, @as(usize, l.k) * l.n);
            };
            var dbufs: std.ArrayList(cuda.DeviceBuffer) = .empty;
            defer for (dbufs.items) |*b| b.free();
            const ws: dsv41.exl3_prefill.Workspace = .{ .xh = try alloc(d, &dbufs, a, cap * maxk * 2, 0), .w = try alloc(d, &dbufs, a, maxkn * 2, 0), .h = try alloc(d, &dbufs, a, 128 * 128 * 2, 0) };
            var had: [128 * 128]u16 = undefined;
            dsv41.exl3_prefill.hadamard(&had);
            try upload(d, ws.h, std.mem.sliceAsBytes(&had));
            const wcache = try alloc(d, &dbufs, a, maxkn * 2, 0);
            const xd_in = try alloc(d, &dbufs, a, cap * maxk * 2, 0);
            const hx = try a.alloc(u16, cap * maxk);
            for (hx) |*h| h.* = bf16(gauss(rnd));
            try upload(d, xd_in, std.mem.sliceAsBytes(hx));
            const o1 = try alloc(d, &dbufs, a, cap * maxn * 4, 0);
            const o2 = try alloc(d, &dbufs, a, cap * maxn * 4, 0);
            const tri_t: dsv41.tri.Tri = .{ .set = &aot_set.?, .stream = s };
            for (rows_list.items) |n| try denseLayer(d, a, &out, s, &tm, &pfk.?, tri_t, ws, wcache, xd_in, o1, o2, lay, li, n, iters);
            if (std.c.getenv("TF_BENCH_DENSE_ONLY") != null) continue;
        }
        for (rows_list.items) |n| {
            const p = n * sl;
            // today's launches, as prompt2d.Two.experts makes them (the pack exchange replaced by this node's pack)
            var t_old: [4][]f32 = undefined;
            for (&t_old) |*t| t.* = try a.alloc(f32, iters);
            for (0..iters + 1) |it| {
                tm.n = 0;
                try tm.mark(s);
                try X2.gateUp(&xk, s, ex, sc, xin, dd, pick, n, cfg.swiglu_limit);
                try tm.mark(s);
                if (rs) |r| try X2.gateUpRest(&xk, &ops, s, ex, sc, r, pick, n, cfg.swiglu_limit);
                try tm.mark(s);
                if (ex.rest) |rest| {
                    const pair = sp.pair.?;
                    _ = pair;
                    try ops.assembleRest(s, sc.xd, sc.xd, pick, rest.owner, xd_full, p, ig, ir);
                } else {
                    try d.check(d.api.cuMemcpyDtoDAsync_v2(xd_full, sc.xd, p * ig * 2, s.handle), "cuMemcpyDtoDAsync");
                }
                try tm.mark(s);
                try X2.down(&xk, s, ex, sc, xd_full, pick, wts, pm, n);
                try tm.mark(s);
                try s.synchronize();
                if (it > 0) for (0..4) |j| {
                    t_old[j][it - 1] = try tm.ms(j);
                };
            }
            var med_old: [4]f32 = undefined;
            var tot_old: f32 = 0;
            for (0..4) |j| {
                med_old[j] = median(t_old[j]);
                tot_old += med_old[j];
            }
            try out.interface.print("{{\"layer\": {d}, \"rows\": {d}, \"path\": \"today\", \"k2\": [{d}, {d}, {d}, {d}, {d}, {d}], \"used\": {d}, \"busiest\": {d}, \"trellis_MB\": {d:.1}, \"rest_trellis_MB\": {d:.1}, \"ms\": {{\"gateup\": {d:.3}, \"rest\": {d:.3}, \"assemble\": {d:.3}, \"down\": {d:.3}}}, \"total_ms\": {d:.3}, \"us_row\": {d:.3}, \"floor_us_row\": {d:.3}}}\n", .{ li, n, ex.k2_gu[0], ex.k2_gu[1], ex.k2_d[0], ex.k2_d[1], if (ex.rest) |r| r.k2_gu[0] else 0, if (ex.rest) |r| r.k2_gu[1] else 0, used, busiest, @as(f64, @floatFromInt(tbytes)) / 1e6, @as(f64, @floatFromInt(rbytes)) / 1e6, med_old[0], med_old[1], med_old[2], med_old[3], tot_old, tot_old * 1000 / @as(f32, @floatFromInt(n)), @as(f64, @floatFromInt(tbytes + rbytes)) / 250e9 * 1e6 / @as(f64, @floatFromInt(n)) });
            try out.interface.flush();

            if (rs) |r| {
                var t_d: [detail_names.len][]f32 = undefined;
                for (&t_d) |*t| t.* = try a.alloc(f32, iters);
                for (0..iters + 1) |it| {
                    tm.n = 0;
                    try oldDetail(&xk, &ops, s, &tm, ex, sc, r, xin, dd, pick, wts, xd_full, pm, n, cfg.swiglu_limit);
                    try s.synchronize();
                    if (it > 0) for (0..detail_names.len) |j| {
                        t_d[j][it - 1] = try tm.ms(j);
                    };
                }
                try out.interface.print("{{\"layer\": {d}, \"rows\": {d}, \"path\": \"today-detail\", \"ms\": {{", .{ li, n });
                for (detail_names, 0..) |nm, j| try out.interface.print("{s}\"{s}\": {d:.3}", .{ if (j == 0) "" else ", ", nm, median(t_d[j]) });
                try out.interface.print("}}}}\n", .{});
                try out.interface.flush();
            }
            if (pk) |*k| {
                // equality first: gate / up (Z main, Z rest, the pack), then down on today's intermediate (Z, the rows)
                try X2.gateUp(&xk, s, ex, sc, xin, dd, pick, n, cfg.swiglu_limit);
                if (rs) |r| try X2.gateUpRest(&xk, &ops, s, ex, sc, r, pick, n, cfg.swiglu_limit);
                try P.gateUp(k, &xk, &ops, s, ex, sc2, rs2, xin, dd, pick, n, cfg.swiglu_limit);
                try s.synchronize();
                const c_zg = try compare(d, a, sc.z, sc2.z, 2 * p * ig * 4);
                const c_zr = if (rs) |r| try compare(d, a, r.z, rs2.?.z, 2 * p * ir * 4) else @TypeOf(c_zg){ .diff = 0, .first = null };
                const c_xd = try compare(d, a, sc.xd, sc2.xd, p * (ig + ir) * 2);
                if (ex.rest) |rest| {
                    try ops.assembleRest(s, sc.xd, sc.xd, pick, rest.owner, xd_full, p, ig, ir);
                    try ops.assembleRest(s, sc2.xd, sc2.xd, pick, rest.owner, xd_full2, p, ig, ir);
                }
                try X2.down(&xk, s, ex, sc, xd_full, pick, wts, pm, n);
                try P.down(k, &xk, s, ex, sc2, xd_full, pick, wts, pm2, n);
                try s.synchronize();
                const c_full = try compare(d, a, xd_full, xd_full2, p * ifull * 2);
                const c_zd = try compare(d, a, sc.z, sc2.z, p * dn * 4);
                const c_pm = try compare(d, a, pm, pm2, n * dn * 4);
                const same = c_zg.diff == 0 and c_zr.diff == 0 and c_xd.diff == 0 and c_full.diff == 0 and c_zd.diff == 0 and c_pm.diff == 0;
                // then the timing (v2's whole path: its gate / up, the assembly, its down)
                var t_new: [4][]f32 = undefined;
                for (&t_new) |*t| t.* = try a.alloc(f32, iters);
                for (0..iters + 1) |it| {
                    tm.n = 0;
                    try tm.mark(s);
                    try P.gateUp(k, &xk, &ops, s, ex, sc2, rs2, xin, dd, pick, n, cfg.swiglu_limit);
                    try tm.mark(s);
                    try tm.mark(s);
                    if (ex.rest) |rest| try ops.assembleRest(s, sc2.xd, sc2.xd, pick, rest.owner, xd_full2, p, ig, ir);
                    try tm.mark(s);
                    try P.down(k, &xk, s, ex, sc2, xd_full2, pick, wts, pm2, n);
                    try tm.mark(s);
                    try s.synchronize();
                    if (it > 0) for (0..4) |j| {
                        t_new[j][it - 1] = if (j == 1) 0 else try tm.ms(j);
                    };
                }
                var med_new: [4]f32 = undefined;
                var tot_new: f32 = 0;
                for (0..4) |j| {
                    med_new[j] = median(t_new[j]);
                    tot_new += med_new[j];
                }
                if (xovl) try xovlCheck(d, a, &out, s, &tm, k, &xk, &ops, ex, sc2, rs2, xin, dd, pick, wts, xd_full2, pm2, ref_xd, save_xd, pm3, pmh, svh_h, n, sl, ig, ir, dn, e_count, rank, iters, cfg.swiglu_limit, li);
                try out.interface.print("{{\"layer\": {d}, \"rows\": {d}, \"path\": \"v2\", \"ms\": {{\"gateup+rest\": {d:.3}, \"assemble\": {d:.3}, \"down\": {d:.3}}}, \"total_ms\": {d:.3}, \"us_row\": {d:.3}, \"speedup\": {d:.3}, \"equal\": {}, \"diff_bytes\": {{\"z_gateup\": {d}, \"z_rest\": {d}, \"xd\": {d}, \"xd_full\": {d}, \"z_down\": {d}, \"out\": {d}}}, \"first\": {{\"z_gateup\": {?d}, \"z_down\": {?d}}}}}\n", .{ li, n, med_new[0], med_new[2], med_new[3], tot_new, tot_new * 1000 / @as(f32, @floatFromInt(n)), tot_old / tot_new, same, c_zg.diff, c_zr.diff, c_xd.diff, c_full.diff, c_zd.diff, c_pm.diff, c_zg.first, c_zd.first });
                try out.interface.flush();
            }
        }
    }
    return 0;
}
