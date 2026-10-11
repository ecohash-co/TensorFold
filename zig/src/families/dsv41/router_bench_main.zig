//! tf-dsv41-router-bench MODEL_DIR CACHE_DIR RANK WORLD KIT_DIR [--layers 0-19|a,b,..] [--iters K] [--seed S]
//!   [--x embed|gauss] [--windows W]:
//! TF_DS_PROMPT_ROUTER=own on one GPU (stage ZR1), with each layer's real gate (gate_w fp16, gate_b) from the rank
//! cache and the served route kernel (the kit's Triton set). For every layer and every chunk the lever acts on (a
//! 4,096-row pass of two 2,048-row runs; a 2,048-row chunk as one run and as a pass's run): today's router (x to fp32,
//! cuBLAS on the runs as prompt.xwTRows issues it, the route kernel on every row) against the lever's (each owner's
//! rows, router_own.cu from x bf16 and gate_w fp16 with the calibrated split-K, the route kernel on those rows), every
//! byte of the logits, picks and weights compared; then W random row windows inside a 2,048-row run (any offset, any
//! length) the same way, and cuBLAS itself on the owners' rows (the negative control: its M changes its split-K).
//! x: --x embed (random tokens' embedding rows, RMS-normed, times the layer's ffn_norm, to bf16: the FFN's input
//! scale) or gauss (N(0, 1) bf16). Timing (median of K): one node's router per pass today (x to fp32 of n rows, gate_w
//! to fp32, the runs, route n rows) and with the lever (the own kernel on q rows, route q rows, the four
//! staging copies; the NCCL gather is not on one GPU), and each kernel tiling at 512 / 1,024 rows. One JSON line a
//! case; the last line sums them.
const std = @import("std");
const cuda = @import("cuda");
const dsv41 = @import("dsv41");
const router = dsv41.router;
const Owners = dsv41.prompt2d.Owners;

const usage = "usage: tf-dsv41-router-bench MODEL_DIR CACHE_DIR RANK WORLD KIT_DIR [--layers 0-19|a,b,..] [--iters K] [--seed S] [--x embed|gauss] [--windows W]\n";

fn bf16(f: f32) u16 {
    const b: u32 = @bitCast(f);
    const r = b +% (0x7fff + ((b >> 16) & 1));
    return @intCast(r >> 16);
}

fn fromBf16(h: u16) f32 {
    return @bitCast(@as(u32, h) << 16);
}

fn gauss(r: std.Random) f32 {
    const u1_ = @max(r.float(f32), 1e-7);
    const u2_ = r.float(f32);
    return @sqrt(-2.0 * @log(u1_)) * @cos(2.0 * std.math.pi * u2_);
}

const Bufs = struct {
    list: std.ArrayList(cuda.DeviceBuffer) = .empty,
    fn take(b: *Bufs, d: *const cuda.Driver, a: std.mem.Allocator, n: usize) !u64 {
        const x = try cuda.DeviceBuffer.alloc(d, @max(n, 16));
        try b.list.append(a, x);
        return x.ptr;
    }
};

fn download(d: *const cuda.Driver, ptr: u64, out: []u8) !void {
    try d.check(d.api.cuMemcpyDtoH_v2(out.ptr, ptr, out.len), "cuMemcpyDtoH");
}

fn upload(d: *const cuda.Driver, ptr: u64, bytes: []const u8) !void {
    try d.check(d.api.cuMemcpyHtoD_v2(ptr, bytes.ptr, bytes.len), "cuMemcpyHtoD");
}

/// A poison fill on the bench's stream (ordered with its launches; a NULL-stream memset is not, against a
/// non-blocking stream).
fn fill(d: *const cuda.Driver, s: cuda.Stream, ptr: u64, v: u8, n: usize) !void {
    try d.check(d.api.cuMemsetD8Async(ptr, v, n, s.handle), "cuMemsetD8Async");
}

/// 4-byte words that differ in [0, n) words of two host slices.
fn diffWords(x: []const u8, y: []const u8) usize {
    var dn: usize = 0;
    var i: usize = 0;
    while (i + 4 <= x.len) : (i += 4) dn += @intFromBool(!std.mem.eql(u8, x[i..][0..4], y[i..][0..4]));
    return dn;
}

/// A rank-cache tensor's bytes.
fn entry(a: std.mem.Allocator, io: std.Io, file: std.Io.File, ix: *const dsv41.rank_cache.Index, key: []const u8) ![]u8 {
    const e = ix.get(key) orelse {
        std.debug.print("no {s} in the rank cache\n", .{key});
        return error.MissingTensor;
    };
    const buf = try a.alloc(u8, e.bytes);
    if (try file.readPositionalAll(io, buf, e.offset) != buf.len) return error.ShortRead;
    return buf;
}

const Ev = struct {
    ev: [2]cuda.Event,
    fn init(d: *const cuda.Driver) !Ev {
        return .{ .ev = .{ try cuda.Event.init(d, true), try cuda.Event.init(d, true) } };
    }
};

fn median(v: []f32) f32 {
    std.mem.sort(f32, v, {}, std.sort.asc(f32));
    return v[v.len / 2];
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
    var layers_text: []const u8 = "";
    var iters: usize = 9;
    var seed: u64 = 7;
    var x_mode: []const u8 = "embed";
    var n_windows: usize = 24;
    var i: usize = 6;
    while (i + 1 < args.len) : (i += 2) {
        const k = args[i];
        const v = args[i + 1];
        if (std.mem.eql(u8, k, "--layers")) layers_text = v else if (std.mem.eql(u8, k, "--iters")) iters = try std.fmt.parseInt(usize, v, 10) else if (std.mem.eql(u8, k, "--seed")) seed = try std.fmt.parseInt(u64, v, 10) else if (std.mem.eql(u8, k, "--x")) x_mode = v else if (std.mem.eql(u8, k, "--windows")) n_windows = try std.fmt.parseInt(usize, v, 10) else {
            std.debug.print("{s}", .{usage});
            return 2;
        }
    }
    const cfg = try dsv41.Config.read(a, io, args[1]);
    const enc = cfg.layers / 2;
    var layer_ids: std.ArrayList(usize) = .empty;
    if (layers_text.len == 0) {
        for (0..enc) |l| try layer_ids.append(a, l);
    } else if (std.mem.indexOfScalar(u8, layers_text, '-')) |dash| {
        const lo = try std.fmt.parseInt(usize, layers_text[0..dash], 10);
        const hi = try std.fmt.parseInt(usize, layers_text[dash + 1 ..], 10);
        for (lo..hi + 1) |l| try layer_ids.append(a, l);
    } else {
        var it = std.mem.splitScalar(u8, layers_text, ',');
        while (it.next()) |t| try layer_ids.append(a, try std.fmt.parseInt(usize, t, 10));
    }
    for (layer_ids.items) |l| if (l >= cfg.layers) return error.NoSuchLayer;
    const embed_x = std.mem.eql(u8, x_mode, "embed");
    if (!embed_x and !std.mem.eql(u8, x_mode, "gauss")) return error.BadX;

    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    const d = &driver;

    var cdir = try std.Io.Dir.cwd().openDir(io, args[2], .{ .iterate = true });
    defer cdir.close(io);
    const cname = try dsv41.rank_cache.find(a, io, cdir, rank, world);
    var ix = try dsv41.rank_cache.open(init.gpa, io, cdir, cname);
    defer ix.deinit();
    var file = try cdir.openFile(io, cname, .{});
    defer file.close(io);

    var s = try cuda.Stream.init(d, true);
    defer s.deinit();
    var set = try cuda.aot.Set.load(init.gpa, io, d, ctx.device, try std.fs.path.join(a, &.{ kit, "aot" }));
    defer set.deinit();
    const t: dsv41.tri.Tri = .{ .set = &set, .stream = s };
    var ops = try dsv41.ops.Ops.load(d, cuda.kernels.torch_pointwise, cuda.kernels.torch_movement, cuda.kernels.dsv41_ops);
    defer ops.unload();
    var rk = try router.Kernels.load(d, cuda.kernels.dsv41_router);
    defer rk.unload();

    const dd = cfg.hidden;
    const ne = cfg.experts;
    const sl = cfg.top_k + 1;
    const shared_id = ne; // the shared expert's slot id (the same constant on both paths)
    const cap: usize = 4096;
    var bufs: Bufs = .{};
    defer for (bufs.list.items) |*b| b.free();
    const blas_ws = try bufs.take(d, a, dsv41.cublas.Blas.workspace_bytes);
    var blas = try dsv41.cublas.Blas.open(s, blas_ws);
    defer blas.close();
    const x = try bufs.take(d, a, cap * dd * 2);
    const xf = try bufs.take(d, a, cap * dd * 4);
    const gate_f = try bufs.take(d, a, ne * dd * 4);
    const gate_h = try bufs.take(d, a, ne * dd * 2);
    const gate_b = try bufs.take(d, a, ne * 4);
    const lg_ref = try bufs.take(d, a, cap * ne * 4);
    const lg_own = try bufs.take(d, a, cap * ne * 4);
    const lg_cb = try bufs.take(d, a, cap * ne * 4);
    const pk_ref = try bufs.take(d, a, cap * sl * 4);
    const wt_ref = try bufs.take(d, a, cap * sl * 4);
    const pk_own = try bufs.take(d, a, cap * sl * 4);
    const wt_own = try bufs.take(d, a, cap * sl * 4);
    const rpw = try bufs.take(d, a, cap * 2 * sl * 4);
    const h_ref = try a.alloc(u8, cap * ne * 4);
    const h_own = try a.alloc(u8, cap * ne * 4);
    const hp_ref = try a.alloc(u8, cap * sl * 4 * 2);
    const hp_own = try a.alloc(u8, cap * sl * 4 * 2);

    var out_buf: [1 << 14]u8 = undefined;
    var out = std.Io.File.stdout().writer(io, &out_buf);

    // the calibration the engine runs at start (layer 0's gate)
    {
        const gw = try entry(a, io, file, &ix, "layers.0.ffn.gate.weight|torch.float16");
        try upload(d, gate_h, gw);
        try d.check(d.api.cuCtxSynchronize(), "cuCtxSynchronize");
        try ops.f16ToF32(s, gate_h, gate_f, ne * dd);
        try s.synchronize();
    }
    const t_cal0 = std.Io.Timestamp.now(io, .awake);
    const cal = try router.calibrate(init.gpa, d, &rk, &blas, s, gate_h, gate_f, dd, ne);
    const cal_ms = @as(f64, @floatFromInt(t_cal0.durationTo(std.Io.Timestamp.now(io, .awake)).nanoseconds)) / 1e6;
    const p = cal.parts;
    {
        try out.interface.print("{{\"calibrate\": {{\"parts\": {d}, \"bounds\": [", .{p.n});
        const pn: usize = @intCast(p.n);
        for (p.b[0 .. pn + 1], 0..) |b, j| try out.interface.print("{s}{d}", .{ if (j == 0) "" else ", ", b });
        try out.interface.print("], \"tried\": {d}, \"ms\": {d:.1}}}, \"rank\": {d}, \"x\": \"{s}\", \"seed\": {d}}}\n", .{ cal.tried, cal_ms, rank, x_mode, seed });
        try out.interface.flush();
    }

    var prng = std.Random.DefaultPrng.init(seed);
    const rnd = prng.random();
    var ev = try Ev.init(d);
    var ts = try a.alloc(f32, iters);
    var cases: usize = 0;
    var equal_cases: usize = 0;
    var win_cases: usize = 0;
    var win_equal: usize = 0;
    var ctl_cases: usize = 0;
    var ctl_differ: usize = 0;
    var sum_today: f64 = 0;
    var sum_own: f64 = 0;
    var n_timed: usize = 0;

    var embed_rows: []const u8 = &.{};
    var vocab: usize = 0;
    if (embed_x) {
        const e = ix.get("embed.weight|None") orelse return error.MissingTensor;
        if (e.dtype != .bf16 or e.dim(1) != dd) return error.UnexpectedTensor;
        vocab = e.dim(0);
        // cap random tokens' rows, read row by row
        const rows_h = try a.alloc(u8, cap * dd * 2);
        for (0..cap) |r| {
            const tok = rnd.uintLessThan(usize, vocab);
            if (try file.readPositionalAll(io, rows_h[r * dd * 2 ..][0 .. dd * 2], e.offset + tok * dd * 2) != dd * 2) return error.ShortRead;
        }
        embed_rows = rows_h;
    }
    const xh = try a.alloc(u16, cap * dd);

    for (layer_ids.items) |li| {
        var kb: [128]u8 = undefined;
        try upload(d, gate_h, try entry(a, io, file, &ix, try std.fmt.bufPrint(&kb, "layers.{d}.ffn.gate.weight|torch.float16", .{li})));
        try upload(d, gate_b, try entry(a, io, file, &ix, try std.fmt.bufPrint(&kb, "layers.{d}.ffn.gate.bias|torch.float32", .{li})));
        // x
        if (embed_x) {
            const nk = try std.fmt.bufPrint(&kb, "layers.{d}.ffn_norm.weight|None", .{li});
            const ne_ = ix.get(nk) orelse return error.MissingTensor;
            const nw = try entry(a, io, file, &ix, nk);
            const norm = try a.alloc(f32, dd);
            for (0..dd) |j| norm[j] = switch (ne_.dtype) {
                .bf16 => fromBf16(std.mem.readInt(u16, nw[j * 2 ..][0..2], .little)),
                .f32 => @bitCast(std.mem.readInt(u32, nw[j * 4 ..][0..4], .little)),
                else => return error.UnexpectedTensor,
            };
            for (0..cap) |r| {
                var ss: f32 = 0;
                for (0..dd) |j| {
                    const v = fromBf16(std.mem.readInt(u16, embed_rows[(r * dd + j) * 2 ..][0..2], .little));
                    ss += v * v;
                }
                const inv = 1.0 / @sqrt(ss / @as(f32, @floatFromInt(dd)) + 1e-6);
                for (0..dd) |j| {
                    const v = fromBf16(std.mem.readInt(u16, embed_rows[(r * dd + j) * 2 ..][0..2], .little));
                    xh[r * dd + j] = bf16(v * inv * norm[j]);
                }
            }
        } else {
            for (xh) |*v| v.* = bf16(gauss(rnd));
        }
        try upload(d, x, std.mem.sliceAsBytes(xh));
        try d.check(d.api.cuCtxSynchronize(), "cuCtxSynchronize"); // (the NULL-stream copies done before s reads them)
        try ops.f16ToF32(s, gate_h, gate_f, ne * dd);

        // the chunks the lever acts on: (n, sub)
        for ([_][2]usize{ .{ 4096, 2048 }, .{ 2048, 2048 }, .{ 2048, 0 } }) |cs| {
            const n = cs[0];
            const sub = cs[1];
            const o = Owners.of(n);
            if (!router.fits(o, sub)) return error.BenchCaseDoesNotFit;
            // today (prompt.moe: toF32, f16ToF32, xwTRows, route)
            try ops.toF32(s, x, xf, n * dd);
            var it = dsv41.prompt.subRows(sub, n);
            while (it.next()) |r| try blas.xwT(xf + r[0] * dd * 4, gate_f, lg_ref + r[0] * ne * 4, r[1], dd, ne);
            try dsv41.tri_norm.route(t, lg_ref, 0, gate_b, cfg.top_k, cfg.routed_scaling, shared_id, pk_ref, wt_ref, n, ne, sl);
            // the lever, every owner's rows one after another (prompt.routeOwn without the gather): x bf16 and gate_w
            // fp16 as the engine holds them
            try fill(d, s, lg_own, 0xa5, n * ne * 4);
            try fill(d, s, pk_own, 0xa5, n * sl * 4);
            try fill(d, s, wt_own, 0xa5, n * sl * 4);
            for (0..4) |g| {
                const r0 = o.r0(g);
                const rows = o.rows(g);
                try router.own(&rk, s, .h, router.variantFor(rows, ne), x + r0 * dd * 2, dd, gate_h, lg_own + r0 * ne * 4, rows, dd, ne, p);
                try dsv41.tri_norm.route(t, lg_own + r0 * ne * 4, 0, gate_b, cfg.top_k, cfg.routed_scaling, shared_id, pk_own + r0 * sl * 4, wt_own + r0 * sl * 4, rows, ne, sl);
            }
            try s.synchronize();
            try download(d, lg_ref, h_ref[0 .. n * ne * 4]);
            try download(d, lg_own, h_own[0 .. n * ne * 4]);
            const d_lg = diffWords(h_ref[0 .. n * ne * 4], h_own[0 .. n * ne * 4]);
            try download(d, pk_ref, hp_ref[0 .. n * sl * 4]);
            try download(d, pk_own, hp_own[0 .. n * sl * 4]);
            const d_pk = diffWords(hp_ref[0 .. n * sl * 4], hp_own[0 .. n * sl * 4]);
            try download(d, wt_ref, hp_ref[0 .. n * sl * 4]);
            try download(d, wt_own, hp_own[0 .. n * sl * 4]);
            const d_wt = diffWords(hp_ref[0 .. n * sl * 4], hp_own[0 .. n * sl * 4]);
            // the negative control: cuBLAS itself on each owner's rows (its M sets its split-K)
            for (0..4) |g| try blas.xwT(xf + o.r0(g) * dd * 4, gate_f, lg_cb + o.r0(g) * ne * 4, o.rows(g), dd, ne);
            try s.synchronize();
            try download(d, lg_cb, h_own[0 .. n * ne * 4]);
            const d_cb = diffWords(h_ref[0 .. n * ne * 4], h_own[0 .. n * ne * 4]);
            const eq = d_lg == 0 and d_pk == 0 and d_wt == 0;
            cases += 1;
            equal_cases += @intFromBool(eq);
            ctl_cases += 1;
            ctl_differ += @intFromBool(d_cb != 0);
            try out.interface.print("{{\"layer\": {d}, \"n\": {d}, \"sub\": {d}, \"owners\": [{d}, {d}, {d}, {d}], \"equal\": {}, \"diff_words\": {{\"logits\": {d}, \"pick\": {d}, \"wts\": {d}}}, \"cublas_own_rows_diff_logits\": {d}}}\n", .{ li, n, sub, o.rows(0), o.rows(1), o.rows(2), o.rows(3), eq, d_lg, d_pk, d_wt, d_cb });
            try out.interface.flush();
        }

        // random row windows inside a 2,048-row run of the 4,096-row reference (run 0 or 1)
        {
            try ops.toF32(s, x, xf, cap * dd);
            for (0..2) |run| try blas.xwT(xf + run * 2048 * dd * 4, gate_f, lg_ref + run * 2048 * ne * 4, 2048, dd, ne);
            try dsv41.tri_norm.route(t, lg_ref, 0, gate_b, cfg.top_k, cfg.routed_scaling, shared_id, pk_ref, wt_ref, cap, ne, sl);
            try s.synchronize();
            try download(d, lg_ref, h_ref[0 .. cap * ne * 4]);
            try download(d, pk_ref, hp_ref[0 .. cap * sl * 4]);
            try download(d, wt_ref, hp_ref[cap * sl * 4 ..][0 .. cap * sl * 4]);
            var bad: usize = 0;
            for (0..n_windows) |wi| {
                const run = rnd.uintLessThan(usize, 2);
                const off = rnd.uintLessThan(usize, 2048);
                const len = 1 + rnd.uintLessThan(usize, 2048 - off);
                const r0 = run * 2048 + off;
                const v: router.Variant = std.meta.tags(router.Variant)[wi % 4];
                const f: router.Form = if ((wi / 4) % 2 == 0) .h else .f32;
                try fill(d, s, lg_own, 0x5a, len * ne * 4);
                if (f == .h) try router.own(&rk, s, .h, v, x + r0 * dd * 2, dd, gate_h, lg_own, len, dd, ne, p) else try router.own(&rk, s, .f32, v, xf + r0 * dd * 4, dd, gate_f, lg_own, len, dd, ne, p);
                try dsv41.tri_norm.route(t, lg_own, 0, gate_b, cfg.top_k, cfg.routed_scaling, shared_id, pk_own, wt_own, len, ne, sl);
                try s.synchronize();
                try download(d, lg_own, h_own[0 .. len * ne * 4]);
                try download(d, pk_own, hp_own[0 .. len * sl * 4]);
                try download(d, wt_own, hp_own[len * sl * 4 ..][0 .. len * sl * 4]);
                const dl = diffWords(h_ref[r0 * ne * 4 ..][0 .. len * ne * 4], h_own[0 .. len * ne * 4]);
                const dp = diffWords(hp_ref[r0 * sl * 4 ..][0 .. len * sl * 4], hp_own[0 .. len * sl * 4]);
                const dw = diffWords(hp_ref[cap * sl * 4 + r0 * sl * 4 ..][0 .. len * sl * 4], hp_own[len * sl * 4 ..][0 .. len * sl * 4]);
                win_cases += 1;
                if (dl == 0 and dp == 0 and dw == 0) win_equal += 1 else {
                    bad += 1;
                    try out.interface.print("{{\"layer\": {d}, \"window_differs\": {{\"r0\": {d}, \"rows\": {d}, \"variant\": \"{s}\", \"form\": \"{s}\", \"logits\": {d}, \"pick\": {d}, \"wts\": {d}}}}}\n", .{ li, r0, len, @tagName(v), @tagName(f), dl, dp, dw });
                }
            }
            try out.interface.print("{{\"layer\": {d}, \"windows\": {d}, \"windows_equal\": {d}}}\n", .{ li, n_windows, n_windows - bad });
            try out.interface.flush();
        }

        // timing on the 4,096-row pass, this rank's rows (rank g = the node's own quarter)
        {
            const n: usize = 4096;
            const o = Owners.of(n);
            const g: usize = rank % 4;
            const r0 = o.r0(g);
            const q = o.rows(g);
            const row = sl * 4;
            var today: f32 = 0;
            var ownt: f32 = 0;
            for (0..2) |which| {
                for (0..iters + 2) |k| {
                    try ev.ev[0].record(s);
                    if (which == 0) {
                        try ops.toF32(s, x, xf, n * dd);
                        try ops.f16ToF32(s, gate_h, gate_f, ne * dd);
                        for (0..2) |run| try blas.xwT(xf + run * 2048 * dd * 4, gate_f, lg_ref + run * 2048 * ne * 4, 2048, dd, ne);
                        try dsv41.tri_norm.route(t, lg_ref, 0, gate_b, cfg.top_k, cfg.routed_scaling, shared_id, pk_ref, wt_ref, n, ne, sl);
                    } else {
                        try router.own(&rk, s, .h, router.variantFor(q, ne), x + r0 * dd * 2, dd, gate_h, lg_own + r0 * ne * 4, q, dd, ne, p);
                        try dsv41.tri_norm.route(t, lg_own + r0 * ne * 4, 0, gate_b, cfg.top_k, cfg.routed_scaling, shared_id, pk_own + r0 * row, wt_own + r0 * row, q, ne, sl);
                        try ops.copyRows(s, pk_own + r0 * row, row, rpw + r0 * 2 * row, 2 * row, row, q);
                        try ops.copyRows(s, wt_own + r0 * row, row, rpw + r0 * 2 * row + row, 2 * row, row, q);
                        try ops.copyRows(s, rpw, 2 * row, pk_own, row, row, n);
                        try ops.copyRows(s, rpw + row, 2 * row, wt_own, row, row, n);
                    }
                    try ev.ev[1].record(s);
                    try ev.ev[1].synchronize();
                    if (k >= 2) ts[k - 2] = try cuda.Event.elapsedMs(ev.ev[0], ev.ev[1]);
                }
                if (which == 0) today = median(ts) else ownt = median(ts);
            }
            // each tiling alone at 1,024 and 512 rows, and cuBLAS at 2,048 / 1,024
            var vt: [4][2]f32 = undefined;
            for (std.meta.tags(router.Variant)) |v| for ([_]usize{ 1024, 512 }, 0..) |rows, ri| {
                for (0..iters + 2) |k| {
                    try ev.ev[0].record(s);
                    try router.own(&rk, s, .h, v, x, dd, gate_h, lg_own, rows, dd, ne, p);
                    try ev.ev[1].record(s);
                    try ev.ev[1].synchronize();
                    if (k >= 2) ts[k - 2] = try cuda.Event.elapsedMs(ev.ev[0], ev.ev[1]);
                }
                vt[@intFromEnum(v)][ri] = median(ts);
            };
            var cb: [2]f32 = undefined;
            for ([_]usize{ 2048, 1024 }, 0..) |rows, ri| {
                for (0..iters + 2) |k| {
                    try ev.ev[0].record(s);
                    try blas.xwT(xf, gate_f, lg_cb, rows, dd, ne);
                    try ev.ev[1].record(s);
                    try ev.ev[1].synchronize();
                    if (k >= 2) ts[k - 2] = try cuda.Event.elapsedMs(ev.ev[0], ev.ev[1]);
                }
                cb[ri] = median(ts);
            }
            sum_today += today;
            sum_own += ownt;
            n_timed += 1;
            const fn_: f32 = @floatFromInt(n);
            try out.interface.print("{{\"layer\": {d}, \"timing_pass\": 4096, \"rank_rows\": {d}, \"ms\": {{\"today\": {d:.4}, \"own\": {d:.4}, \"cublas_2048\": {d:.4}, \"cublas_1024\": {d:.4}, \"own_64x64\": [{d:.4}, {d:.4}], \"own_32x64\": [{d:.4}, {d:.4}], \"own_128x64\": [{d:.4}, {d:.4}], \"own_64x128\": [{d:.4}, {d:.4}]}}, \"us_row\": {{\"today\": {d:.4}, \"own\": {d:.4}, \"saved\": {d:.4}}}}}\n", .{ li, q, today, ownt, cb[0], cb[1], vt[0][0], vt[0][1], vt[1][0], vt[1][1], vt[2][0], vt[2][1], vt[3][0], vt[3][1], today * 1000 / fn_, ownt * 1000 / fn_, (today - ownt) * 1000 / fn_ });
            try out.interface.flush();
        }
    }
    const nl: f64 = @floatFromInt(@max(n_timed, 1));
    try out.interface.print("{{\"summary\": {{\"cases\": {d}, \"equal\": {d}, \"windows\": {d}, \"windows_equal\": {d}, \"cublas_own_rows_cases\": {d}, \"cublas_own_rows_differ\": {d}, \"layers_timed\": {d}, \"router_us_row_20_layers\": {{\"today\": {d:.3}, \"own_compute\": {d:.3}}}}}}}\n", .{ cases, equal_cases, win_cases, win_equal, ctl_cases, ctl_differ, n_timed, sum_today / nl * 20 * 1000 / 4096, sum_own / nl * 20 * 1000 / 4096 });
    try out.interface.flush();
    return if (equal_cases == cases and win_equal == win_cases) 0 else 1;
}
