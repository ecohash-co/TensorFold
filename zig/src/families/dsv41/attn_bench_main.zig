//! tf-dsv41-attn-bench MODEL_DIR CACHE_DIR RANK WORLD KIT_DIR [--start S] [--rows N] [--iters K] [--seed X]
//!   [--sub 2048|0] [--levers LIST] [--cmp 0|1] [--layers N]:
//! one 2D node's prompt-chunk attention (prompt.attentionCore: the attention front, RoPE, window keys, a kv source's
//! compressor and caches, the indexer and its selection, the sparse attention, wo_a) and wo_b, layer after layer
//! (0 .. N-1, default the 20 encoder layers), on one GPU with the rank's real weights (the lane's rank cache) and the
//! served kernels (the kit's Triton set and cubins, the engine's images, the v2 W_q cache). No NCCL, no other node:
//! wo_b reads Two.u_full as it stands (the column partner's half not exchanged).
//! The compressed caches of positions [0, S) are made first by the real compressor (each kv-source layer's
//! kvSourceUpdate on 4,096-row chunks of random tokens), the window rings filled with N(0, 1) keys; the chunk is then S ..
//! S + N of random tokens, each layer's x its hc_pre of the token streams (attnMixes, not timed).
//! Each run is timed per layer with events (attentionCore, wo_b) and by phase (pprof marks: attn, dense, mix); one JSON
//! line a run. --levers: a comma list of TF_DS_PROMPT_ATTN levers (attn.zig) for the runs after the first; --cmp 1:
//! the first run with no lever, then every layer's outputs of each later run compared byte for byte with it (q, the
//! window keys, o, u, pa, the selection and the caches the layer writes, the ring).
const std = @import("std");
const cuda = @import("cuda");
const dsv41 = @import("dsv41");
const prompt = dsv41.prompt;

const usage = "usage: tf-dsv41-attn-bench MODEL_DIR CACHE_DIR RANK WORLD KIT_DIR [--start S] [--rows N] [--iters K] [--seed X] [--sub 2048|0] [--levers LIST] [--cmp 0|1] [--layers N]\n";

fn readFile(a: std.mem.Allocator, io: std.Io, dir: []const u8, name: []const u8) ![]const u8 {
    const path = try std.fs.path.join(a, &.{ dir, name });
    return std.Io.Dir.cwd().readFileAllocOptions(io, path, a, .limited(1 << 30), .@"16", null);
}

fn bf16(f: f32) u16 {
    const b: u32 = @bitCast(f);
    const r = b + 0x7fff + ((b >> 16) & 1);
    return @intCast(r >> 16);
}

fn gauss(r: std.Random) f32 {
    const u1_ = @max(r.float(f32), 1e-7);
    const u2_ = r.float(f32);
    return @sqrt(-2.0 * @log(u1_)) * @cos(2.0 * std.math.pi * u2_);
}

fn download(d: *const cuda.Driver, a: std.mem.Allocator, ptr: u64, n: usize) ![]u8 {
    const h = try a.alloc(u8, n);
    try d.check(d.api.cuMemcpyDtoH_v2(h.ptr, ptr, n), "cuMemcpyDtoH");
    return h;
}

/// sha256 of a device range (the first 8 bytes as a u64).
fn digest(d: *const cuda.Driver, a: std.mem.Allocator, ptr: u64, n: usize) !u64 {
    if (n == 0) return 0;
    const h = try download(d, a, ptr, n);
    defer a.free(h);
    var out: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(h, &out, .{});
    return std.mem.readInt(u64, out[0..8], .little);
}

const n_points = 9;
const point_names = [n_points][]const u8{ "q", "wsrc", "o", "u", "pa", "cidx", "comp", "idxk", "ring" };

/// One layer's output digests.
const Digests = struct { v: [n_points]u64 = @splat(0), kk: usize = 0 };

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
    var start: usize = 28672;
    var rows: usize = 4096;
    var iters: usize = 3;
    var seed: u64 = 7;
    var sub: usize = 2048;
    var levers_text: []const u8 = "";
    var cmp = false;
    var n_layers: usize = 20;
    var i: usize = 6;
    while (i + 1 < args.len) : (i += 2) {
        const k = args[i];
        const v = args[i + 1];
        if (std.mem.eql(u8, k, "--start")) start = try std.fmt.parseInt(usize, v, 10) else if (std.mem.eql(u8, k, "--rows")) rows = try std.fmt.parseInt(usize, v, 10) else if (std.mem.eql(u8, k, "--iters")) iters = try std.fmt.parseInt(usize, v, 10) else if (std.mem.eql(u8, k, "--seed")) seed = try std.fmt.parseInt(u64, v, 10) else if (std.mem.eql(u8, k, "--sub")) sub = try std.fmt.parseInt(usize, v, 10) else if (std.mem.eql(u8, k, "--levers")) levers_text = v else if (std.mem.eql(u8, k, "--cmp")) cmp = !std.mem.eql(u8, v, "0") else if (std.mem.eql(u8, k, "--layers")) n_layers = try std.fmt.parseInt(usize, v, 10) else {
            std.debug.print("{s}", .{usage});
            return 2;
        }
    }
    const lv = try dsv41.attn.parse(if (levers_text.len == 0) null else levers_text);
    const cap: usize = 4096;
    if (rows > cap or rows < 256 or start % cap != 0) return error.BadShape;
    const ctx_cap = start + cap;

    var cfg = try dsv41.Config.read(a, io, args[1]);
    if (n_layers > cfg.layers / 2) return error.NotAnEncoderLayer;
    const full_layers = cfg.layers;
    cfg.layers = n_layers; // (loads layers 0 .. n_layers - 1)
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
    cfg.layers = full_layers; // (the encoder's layers are layers / 2: the levers' and the selection's rules)
    const load_s = @as(f64, @floatFromInt(t0.durationTo(std.Io.Timestamp.now(io, .awake)).nanoseconds)) / 1e9;

    var out_buf: [1 << 14]u8 = undefined;
    var out = std.Io.File.stdout().writer(io, &out_buf);
    try out.interface.print("{{\"load_s\": {d:.1}, \"layers\": {d}, \"rank\": {d}, \"world\": {d}, \"start\": {d}, \"rows\": {d}, \"sub\": {d}, \"levers\": \"{s}\"}}\n", .{ load_s, n_layers, rank, world, start, rows, sub, levers_text });
    try out.interface.flush();

    var s = try cuda.Stream.init(d, true);
    defer s.deinit();
    var set = try cuda.aot.Set.load(init.gpa, io, d, ctx.device, try std.fs.path.join(a, &.{ kit, "aot" }));
    defer set.deinit();
    var pf = try dsv41.exl3_prefill.Kernels.load(d, try readFile(a, io, kit, "cubins/linear.cubin"));
    defer pf.unload();
    var lg = try dsv41.exl3_linear.Kernels.load(d, try readFile(a, io, kit, "cubins/linear_grouped.cubin"));
    defer lg.unload();
    var exk = try dsv41.exl3_experts.Kernels.load(d, try readFile(a, io, kit, "cubins/experts.cubin"), try readFile(a, io, kit, "cubins/experts_cb.cubin"), dsv41.exl3_linear.codebook_mul1);
    defer exk.unload();
    var ops = try dsv41.ops.Ops.load(d, cuda.kernels.torch_pointwise, cuda.kernels.torch_movement, cuda.kernels.dsv41_ops);
    defer ops.unload();
    var ex = try dsv41.exact.Exact.load(d, cuda.kernels.dsv41_torch);
    defer ex.unload();

    // the RoPE tables, as model.zig loads them
    const half = cfg.rope_dim / 2;
    const rope_rows = ctx_cap + 64 + 1;
    var rope_buf = try cuda.DeviceBuffer.alloc(d, 4 * rope_rows * half * 4);
    defer rope_buf.free();
    {
        const part = try a.alloc(u8, rope_rows * half * 4);
        for ([_][]const u8{ "rope-plain-cos.f32", "rope-plain-sin.f32", "rope-compressed-cos.f32", "rope-compressed-sin.f32" }, 0..) |name, j| {
            var f = try std.Io.Dir.cwd().openFile(io, try std.fs.path.join(a, &.{ kit, name }), .{});
            defer f.close(io);
            if (try f.readPositionalAll(io, part, 0) != part.len) return error.ShortRopeTable;
            try rope_buf.upload(j * part.len, part);
        }
    }
    const tbl = rope_rows * half * 4;

    var arena = try prompt.Arena.init(d, 256 << 20);
    defer arena.deinit();
    var two = try dsv41.prompt2d.Two.init(&cfg, &w, &arena, sp, cap);
    var x2 = try dsv41.exl3_prompt.Kernels.load(d, cuda.kernels.dsv41_exl3_prompt);
    defer x2.unload();
    var e: prompt.Engine = .{ .d = d, .s = s, .t = .{ .set = &set, .stream = s }, .blas = undefined, .comm = undefined, .pf = &pf, .lin = &lg, .ex = &exk, .ops = &ops, .exact = &ex, .c = &cfg, .w = &w, .world = sp.world, .plain = .{ .cos = rope_buf.ptr, .sin = rope_buf.ptr + tbl }, .compressed = .{ .cos = rope_buf.ptr + 2 * tbl, .sin = rope_buf.ptr + 3 * tbl }, .two = &two };
    e.x2 = &x2;
    const wq_bytes = try dsv41.exl3_prompt.cacheWq(d, &w, &pf, s, n_layers);
    var ch = try prompt.Chunk.init(&e, &arena, cap, ctx_cap);
    var cs = try prompt.Caches.init(&e, &arena, ctx_cap, 1);
    var blas = try dsv41.cublas.Blas.open(s, ch.blas_ws);
    defer blas.close();
    e.blas = &blas;
    // the window rings (one stream), N(0, 1) keys, and a pristine copy each run starts from
    const ring_bytes = e.ringBytes();
    const rings = try a.alloc(u64, n_layers);
    const ring0 = try arena.take(n_layers * ring_bytes);
    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();
    {
        const host = try a.alloc(u16, ring_bytes / 2);
        for (rings, 0..) |*x, li| {
            x.* = try arena.take(ring_bytes);
            for (host) |*v| v.* = bf16(gauss(r));
            try prompt.upload(&e, x.*, host.ptr, ring_bytes);
            try prompt.upload(&e, ring0 + li * ring_bytes, host.ptr, ring_bytes);
        }
    }
    try s.synchronize();
    try out.interface.print("{{\"wq_cache_mib\": {d}, \"arena_mib\": {d}}}\n", .{ wq_bytes >> 20, arena.used >> 20 });
    try out.interface.flush();

    // the caches of positions [0, start): the compressor of every kv-source layer on 4,096-row chunks of random tokens
    const ids = try a.alloc(i64, cap);
    const host_pos = try a.alloc(i64, cap);
    const vocab: u64 = 128000;
    {
        var sh: prompt.Shared = .{};
        var p0: usize = 0;
        const tc = std.Io.Timestamp.now(io, .awake);
        while (p0 < start) : (p0 += cap) {
            for (ids) |*x| x.* = @intCast(r.uintLessThan(u64, vocab) + 3);
            try prompt.begin(&e, &ch, ids, p0, host_pos);
            ch.sub_rows = sub;
            for (0..n_layers) |li| {
                if (w.layers[li].comp_wkv == null) continue;
                try prompt.attnMixes(&e, &ch, li);
                try prompt.kvSourceUpdate(&e, &ch, &cs, &sh, li);
            }
        }
        try s.synchronize();
        const cs_s = @as(f64, @floatFromInt(tc.durationTo(std.Io.Timestamp.now(io, .awake)).nanoseconds)) / 1e9;
        try out.interface.print("{{\"caches_s\": {d:.1}, \"positions\": {d}}}\n", .{ cs_s, start });
        try out.interface.flush();
    }
    // the chunk's tokens (the same every run)
    for (ids) |*x| x.* = @intCast(r.uintLessThan(u64, vocab) + 3);

    var prof = try dsv41.pprof.Prof.init(init.gpa, d, rank);
    defer prof.deinit();
    const nl = n_layers;
    // TF_DS_PROMPT_ATTN: the state the engine reads (run 0 and the runs without a lever: none on); this node's SP
    // rows (rows2's owners of the chunk) around the attention
    var ast: dsv41.attn.State = .{ .emulate = cmp, .no_gather = !cmp };
    var ak: ?dsv41.attn.Kernels = null;
    defer if (ak) |*x| x.unload();
    if (lv.tsel) {
        ak = try dsv41.attn.Kernels.load(d, cuda.kernels.dsv41_attn);
        ast.k = &ak.?;
    }
    if (lv.any()) e.attn = &ast;
    const node = two.r + 2 * two.p;
    var ev: [2 * 64 + 1]cuda.Event = undefined;
    for (&ev) |*x| x.* = try cuda.Event.init(d, true);
    const base = try a.alloc(Digests, nl);
    const cur = try a.alloc(Digests, nl);
    const runs = if (cmp) iters + 1 else iters;
    for (0..runs) |run| {
        const lever = lv.any() and run > 0 and (cmp or run > iters / 2);
        ast.lv = if (lever) lv else .{};
        // every run from the same state: the rings, then the chunk
        for (rings, 0..) |x, li| try ops.copyRows(s, ring0 + li * ring_bytes, ring_bytes, x, ring_bytes, ring_bytes, 1);
        try prompt.begin(&e, &ch, ids[0..rows], start, host_pos);
        ch.sub_rows = if (rows > 2048) sub else 0;
        var sh: prompt.Shared = .{};
        var ms_core = try a.alloc(f32, nl);
        var ms_wob = try a.alloc(f32, nl);
        e.prof = &prof;
        try prof.begin(s, start, rows, 1 << 30);
        for (0..nl) |li| {
            const lay = w.layers[li];
            try prompt.attnMixes(&e, &ch, li);
            try ev[2 * li].record(s);
            ch.sp = if (lv.any()) prompt.Sp.of(node, rows, false) else null;
            try prompt.attentionCore(&e, &ch, &cs, &sh, li, rings[li], 0, false);
            ch.sp = null;
            try ev[2 * li + 1].record(s);
            // wo_b of u_full (this node's half as wo_a left it in u: copied in, the partner's columns as they stand)
            const k = two.uw[0] + two.uw[1];
            try prompt.mm(&e, &ch, lay.wo_b, two.u_full, k, ch.pa, .fp32, lay.wo_b.n);
            try prompt.pm(&e, .dense);
            try ev[2 * nl + 0].record(s);
            try s.synchronize();
            ms_core[li] = try cuda.Event.elapsedMs(ev[2 * li], ev[2 * li + 1]);
            ms_wob[li] = try cuda.Event.elapsedMs(ev[2 * li + 1], ev[2 * nl]);
            if (cmp) {
                const dg = if (run == 0) &base[li] else &cur[li];
                dg.* = try layerDigests(&e, &ch, &cs, &sh, li, rings[li], a);
            }
        }
        e.prof = null;
        try prof.finish();
        var tot: f64 = 0;
        var totw: f64 = 0;
        for (ms_core, ms_wob) |x, y| {
            tot += x;
            totw += y;
        }
        const fr: f64 = @floatFromInt(rows);
        try out.interface.print("{{\"run\": {d}, \"lever\": {}, \"core_ms\": {d:.3}, \"wob_ms\": {d:.3}, \"core_us_row\": {d:.2}, \"wob_us_row\": {d:.2}, \"per_layer_core_ms\": [", .{ run, lever, tot, totw, tot * 1000 / fr, totw * 1000 / fr });
        for (ms_core, 0..) |x, li| try out.interface.print("{s}{d:.3}", .{ if (li == 0) "" else ", ", x });
        try out.interface.print("]", .{});
        if (cmp and run > 0) {
            var diff: usize = 0;
            try out.interface.print(", \"cmp\": [", .{});
            var first = true;
            for (0..nl) |li| for (0..n_points) |j| {
                if (base[li].v[j] != cur[li].v[j] or base[li].kk != cur[li].kk) {
                    diff += 1;
                    try out.interface.print("{s}\"L{d}:{s}\"", .{ if (first) "" else ", ", li, point_names[j] });
                    first = false;
                }
            };
            try out.interface.print("], \"equal\": {}", .{diff == 0});
        }
        try out.interface.print("}}\n", .{});
        try out.interface.flush();
    }
    return 0;
}

/// A layer's outputs after attentionCore and wo_b: q, the window keys (wsrc), o, u, pa, the selection (cidx, kk a
/// row) when the layer has one, the caches its kv source writes (all rows up to the chunk's end), its ring.
fn layerDigests(e: *const prompt.Engine, ch: *const prompt.Chunk, cs: *const prompt.Caches, sh: *const prompt.Shared, li: usize, ring: u64, a: std.mem.Allocator) !Digests {
    const c = e.c;
    const d = e.d;
    const lay = e.w.layers[li];
    const n = ch.n;
    const hl = e.heads();
    const hd = c.head_dim;
    var g: Digests = .{};
    g.v[0] = try digest(d, a, ch.q, n * hl * hd * 2);
    const lo = ch.start -| (c.window - 1);
    g.v[1] = try digest(d, a, ch.wsrc, (ch.start - lo + n) * hd * 2);
    g.v[2] = try digest(d, a, ch.o, n * hl * hd * 2);
    g.v[3] = try digest(d, a, ch.u, n * lay.groups * lay.wo_a[0].n * 2);
    g.v[4] = try digest(d, a, ch.pa, n * lay.wo_b.n * 4);
    if (lay.ratio != 0) {
        const kk = sh.kk orelse 0;
        g.kk = kk;
        g.v[5] = try digest(d, a, ch.cidx, n * kk * 8);
        if (lay.comp_wkv != null) {
            const ents = (ch.start + n) / lay.ratio;
            g.v[6] = (try digest(d, a, cs.comp_codes[li], ents * hd / 2)) ^ (try digest(d, a, cs.comp_scales[li], ents * hd / 16));
            if (lay.idx_wk != null) g.v[7] = (try digest(d, a, cs.idx_codes[li], ents * c.index_head_dim / 2)) ^ (try digest(d, a, cs.idx_scales[li], ents * c.index_head_dim / 32));
        }
    }
    g.v[8] = try digest(d, a, ring, e.ringBytes());
    return g;
}
