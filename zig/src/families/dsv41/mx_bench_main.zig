//! tf-dsv41-mx-bench MODEL_DIR CKPT_DIR SLICES_DIR NODE [--layers 0,1] [--rows 2048,4096] [--iters 10] [--warmup 3]
//!   [--seed 7] [--check 1] [--pairs 8] [--geos 2,2,4] [--tune 0|1] [--dump-check DIR] [--min-avail-gib 4]
//! Stage MX3 (milestone 1): a 2D node's MXFP4 prompt-chunk experts (mx_prompt.zig) on ONE GPU with that node's real
//! slices (convert_mxfp4_slices.py), as bench_mx_zigsplit.py (stage MX2) runs the Python kernel:
//!   - routing: the layer's real router from the original checkpoint (CKPT_DIR) on N(0, 1) bf16 rows (host fp32 logits,
//!     sqrt(softplus) scores, top-6 of score + bias, normalized x 1.5; the shared expert in slot 7 with weight 1);
//!   - the chain timed with events, median over --iters after --warmup, a 256 MiB L2 flush before each rep: plan (both
//!     groupings), gate_main, gate_rest, shared_gu, assemble (assembleRest of this node's pack with itself: the column
//!     partner's main part and rest rows stand in as this node's, as tf-dsv41-exl3-bench does), down, shared_down,
//!     combine (7 slots). routed_ms = plan + gate_main + gate_rest + down + combine (the Python bench's mxfp4 total);
//!   - --check 1: --pairs routed pairs against a float64 reference from the CHECKPOINT bytes (the Python reference():
//!     act over the main columns, the rest columns where this pair computes them, y = W2 xd), two rows' shared slot
//!     against float64 from the slice's exact bf16 cuts, four rows' combine against a float64 weighted sum, and every
//!     y / pack value finite. PASS: act and rest rel < 2e-2, y rel < 3e-2, shared act < 2e-2 and y < 3e-2, combine
//!     rel < 1e-5, no non-finite value (the Python bench's thresholds);
//!   - --dump-check DIR: the Python bench's sampled pairs (dump_to_bin.py files check-L<layer>-R<rows>.bin of this
//!     node) through this binary's kernels: act (main), the rest block (parity experts) and y compared BIT FOR BIT
//!     with what the Python kernel wrote, and the float64 reference recomputed against the recorded act_rel / y_rel;
//!   - --tune 1: every geometry for gate_main / gate_rest / down at the largest row count.
//! One JSON line a record; the last line {"pass": ...}; exit 0 only when every check passed.
//! MODEL_DIR is the EXL3 model directory (its config.json: the dimensions and the split). No NCCL, no other node.
const std = @import("std");
const cuda = @import("cuda");
const dsv41 = @import("dsv41");
const MX = dsv41.mx_prompt;
const R = dsv41.mx_ref;

const usage = "usage: tf-dsv41-mx-bench MODEL_DIR CKPT_DIR SLICES_DIR NODE [--layers 0,1] [--rows 2048,4096] [--iters 10] [--warmup 3] [--seed 7] [--check 1] [--pairs 8] [--geos 2,2,4] [--tune 0|1] [--dump-check DIR] [--min-avail-gib 4]\n";

const stage_names = [_][]const u8{ "plan", "gate_main", "gate_rest", "shared_gu", "assemble", "down", "shared_down", "combine" };

const Timer = struct {
    ev: [16]cuda.Event,
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

fn gauss(r: std.Random) f32 {
    const u1_ = @max(r.float(f32), 1e-7);
    const u2_ = r.float(f32);
    return @sqrt(-2.0 * @log(u1_)) * @cos(2.0 * std.math.pi * u2_);
}

fn alloc(d: *const cuda.Driver, list: *std.ArrayList(cuda.DeviceBuffer), a: std.mem.Allocator, n: usize) !u64 {
    var b = try cuda.DeviceBuffer.alloc(d, @max(n, 256));
    try b.fill8(0, null);
    try list.append(a, b);
    return b.ptr;
}

fn download(d: *const cuda.Driver, out: []u8, ptr: u64) !void {
    try d.check(d.api.cuMemcpyDtoH_v2(out.ptr, ptr, out.len), "cuMemcpyDtoH");
}

fn upload(d: *const cuda.Driver, ptr: u64, bytes: []const u8) !void {
    try d.check(d.api.cuMemcpyHtoD_v2(ptr, bytes.ptr, bytes.len), "cuMemcpyHtoD");
}

/// A JSON number, or a string for a non-finite value.
fn num(v: f64) Num {
    return .{ .v = v };
}
const Num = struct {
    v: f64,
    pub fn format(n: Num, w: *std.Io.Writer) std.Io.Writer.Error!void {
        if (std.math.isFinite(n.v)) return w.print("{e}", .{n.v});
        return w.print("\"{d}\"", .{n.v});
    }
};

fn parseList(a: std.mem.Allocator, text: []const u8) ![]usize {
    var out: std.ArrayList(usize) = .empty;
    var it = std.mem.splitScalar(u8, text, ',');
    while (it.next()) |t| try out.append(a, try std.fmt.parseInt(usize, t, 10));
    return out.items;
}

/// Everything the cases share.
const Bench = struct {
    a: std.mem.Allocator,
    io: std.Io,
    d: *const cuda.Driver,
    s: cuda.Stream,
    k: *const MX.Kernels,
    ops: *const dsv41.ops.Ops,
    blas: *const dsv41.cublas.Blas,
    tm: *Timer,
    out: *std.Io.File.Writer,
    cfg: dsv41.Config,
    slots: usize,
    cap: usize,
    // device buffers (cap rows)
    x: u64,
    pick: u64,
    wts: u64,
    pack: u64,
    xd_full: u64,
    y: u64,
    outp: u64,
    flush: u64,
    flush_bytes: usize,
    sc: MX.Scratch,
    iters: usize,
    warmup: usize,
    seed: u64,
    pairs: usize,
    all_pass: bool = true,

    fn flushL2(b: *Bench) !void {
        try b.d.check(b.d.api.cuMemsetD8Async(b.flush, 0, b.flush_bytes, b.s.handle), "cuMemsetD8Async");
    }

    /// One pass of the chain with a timer mark around each stage.
    fn once(b: *Bench, lay: *const MX.Layer, n: usize, g: MX.Geos) ![stage_names.len]f32 {
        const X = lay.x;
        const p = n * b.slots;
        const lim = b.cfg.swiglu_limit;
        const t = b.tm;
        t.n = 0;
        try t.mark(b.s);
        try MX.plan(b.k, b.ops, b.s, lay, b.sc, b.pick, n);
        try t.mark(b.s);
        try MX.gateMain(b.k, b.s, lay, b.sc, b.x, X.hidden, n, b.pack, lim, g.main);
        try t.mark(b.s);
        try MX.gateRest(b.k, b.s, lay, b.sc, b.x, X.hidden, n, b.pack, lim, g.rest);
        try t.mark(b.s);
        try MX.sharedGateUp(b.k, b.blas, b.s, lay, b.sc, b.x, X.hidden, n, b.pack, lim);
        try t.mark(b.s);
        try b.ops.assembleRest(b.s, b.pack, b.pack, b.pick, lay.owner, b.xd_full, p, X.mainW(), X.restW());
        try t.mark(b.s);
        try MX.downRouted(b.k, b.s, lay, b.sc, b.xd_full, n, b.y, g.down);
        try t.mark(b.s);
        try MX.sharedDown(b.blas, lay, b.xd_full, n, b.slots, b.y);
        try t.mark(b.s);
        try MX.combine(b.k, b.s, b.y, b.wts, b.outp, X.downN(), n, b.slots, X.downN());
        try t.mark(b.s);
        try b.s.synchronize();
        var r: [stage_names.len]f32 = undefined;
        for (0..stage_names.len) |i| r[i] = try t.ms(i);
        return r;
    }

    fn timed(b: *Bench, lay: *const MX.Layer, n: usize, g: MX.Geos, warm: usize, reps: usize) ![stage_names.len]f32 {
        var samples = try b.a.alloc([stage_names.len]f32, reps);
        defer b.a.free(samples);
        for (0..warm + reps) |i| {
            try b.flushL2();
            const t = try b.once(lay, n, g);
            if (i >= warm) samples[i - warm] = t;
        }
        var med: [stage_names.len]f32 = undefined;
        const col = try b.a.alloc(f32, reps);
        defer b.a.free(col);
        for (0..stage_names.len) |j| {
            for (samples, 0..) |sm, i| col[i] = sm[j];
            med[j] = median(col);
        }
        return med;
    }
};

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const gpa = init.gpa;
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    if (args.len < 5) {
        std.debug.print("{s}", .{usage});
        return 2;
    }
    const node = try std.fmt.parseInt(u32, args[4], 10);
    if (node > 3) {
        std.debug.print("NODE is 0..3 (castor, pollux, clytemnestra, helen)\n", .{});
        return 2;
    }
    var layers_text: []const u8 = "0,1";
    var rows_text: []const u8 = "2048,4096";
    var iters: usize = 10;
    var warmup: usize = 3;
    var seed: u64 = 7;
    var check = true;
    var pairs: usize = 8;
    var geos_text: []const u8 = "2,2,4";
    var tune = false;
    var dump_dir: ?[]const u8 = null;
    var min_avail_gib: f64 = 4;
    var i: usize = 5;
    while (i + 1 < args.len) : (i += 2) {
        const key = args[i];
        const v = args[i + 1];
        if (std.mem.eql(u8, key, "--layers")) layers_text = v else if (std.mem.eql(u8, key, "--rows")) rows_text = v else if (std.mem.eql(u8, key, "--iters")) iters = try std.fmt.parseInt(usize, v, 10) else if (std.mem.eql(u8, key, "--warmup")) warmup = try std.fmt.parseInt(usize, v, 10) else if (std.mem.eql(u8, key, "--seed")) seed = try std.fmt.parseInt(u64, v, 10) else if (std.mem.eql(u8, key, "--check")) check = !std.mem.eql(u8, v, "0") else if (std.mem.eql(u8, key, "--pairs")) pairs = try std.fmt.parseInt(usize, v, 10) else if (std.mem.eql(u8, key, "--geos")) geos_text = v else if (std.mem.eql(u8, key, "--tune")) tune = !std.mem.eql(u8, v, "0") else if (std.mem.eql(u8, key, "--dump-check")) dump_dir = v else if (std.mem.eql(u8, key, "--min-avail-gib")) min_avail_gib = try std.fmt.parseFloat(f64, v) else {
            std.debug.print("{s}", .{usage});
            return 2;
        }
    }
    if (i != args.len or iters == 0) {
        std.debug.print("{s}", .{usage});
        return 2;
    }
    const layer_ids = try parseList(a, layers_text);
    const rows_list = try parseList(a, rows_text);
    const gl = try parseList(a, geos_text);
    if (gl.len != 3) return error.BadGeos;
    const geo: MX.Geos = .{ .main = @intCast(gl[0]), .rest = @intCast(gl[1]), .down = @intCast(gl[2]) };
    for (gl) |g| _ = try MX.geoIndex(@intCast(g));
    var cap: usize = 64;
    for (rows_list) |n| cap = @max(cap, n);

    const cfg = try dsv41.Config.read(gpa, io, args[1]);
    var sp = try dsv41.prompt2d.split(node, 4);
    sp.parity = true; // the served TF_DS_2D_GU=parity
    const slots = cfg.top_k + 1;

    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    const d = &driver;
    var s = try cuda.Stream.init(d, true);
    defer s.deinit();
    var k = try MX.Kernels.load(d, cuda.kernels.dsv41_mx_prompt);
    defer k.unload();
    var ops = try dsv41.ops.Ops.load(d, cuda.kernels.torch_pointwise, cuda.kernels.torch_movement, cuda.kernels.dsv41_ops);
    defer ops.unload();
    var bufs: std.ArrayList(cuda.DeviceBuffer) = .empty;
    defer for (bufs.items) |*b| b.free();
    const ws = try alloc(d, &bufs, a, dsv41.cublas.Blas.workspace_bytes);
    var blas = try dsv41.cublas.Blas.open(s, ws);
    defer blas.close();
    var tm = try Timer.init(d);
    var ck = try R.Ckpt.open(gpa, io, args[2]);
    defer ck.close();

    var out_buf: [1 << 14]u8 = undefined;
    var out = std.Io.File.stdout().writer(io, &out_buf);

    const X0 = try MX.Expect.of(cfg, sp, @intCast(layer_ids[0]));
    const p_cap = cap * slots;
    const ssz = MX.Scratch.sizes(cap, slots, X0);
    var sptr: [12]u64 = undefined;
    for (ssz, &sptr) |n, *pp| pp.* = try alloc(d, &bufs, a, n);
    const flush_bytes: usize = 256 << 20;
    var b: Bench = .{
        .a = a,
        .io = io,
        .d = d,
        .s = s,
        .k = &k,
        .ops = &ops,
        .blas = &blas,
        .tm = &tm,
        .out = &out,
        .cfg = cfg,
        .slots = slots,
        .cap = cap,
        .x = try alloc(d, &bufs, a, cap * X0.hidden * 2),
        .pick = try alloc(d, &bufs, a, p_cap * 4),
        .wts = try alloc(d, &bufs, a, p_cap * 4),
        .pack = try alloc(d, &bufs, a, p_cap * (X0.mainW() + X0.restW()) * 2),
        .xd_full = try alloc(d, &bufs, a, p_cap * X0.halfW() * 2),
        .y = try alloc(d, &bufs, a, p_cap * X0.downN() * 4),
        .outp = try alloc(d, &bufs, a, cap * X0.downN() * 4),
        .flush = try alloc(d, &bufs, a, flush_bytes),
        .flush_bytes = flush_bytes,
        .sc = MX.Scratch.of(cap, slots, X0, sptr),
        .iters = iters,
        .warmup = warmup,
        .seed = seed,
        .pairs = pairs,
    };
    const scratch_bytes = blk: {
        var t: usize = 0;
        for (bufs.items) |bb| t += bb.len;
        break :blk t;
    };
    try out.interface.print("{{\"node\": {d}, \"pair\": {d}, \"tp2_rank\": {d}, \"slots\": {d}, \"cap_rows\": {d}, \"scratch_MB\": {d:.1}, \"geos\": [{d}, {d}, {d}], \"entries\": [", .{ node, X0.pair, X0.tp2_rank, slots, cap, @as(f64, @floatFromInt(scratch_bytes)) / 1e6, geo.main, geo.rest, geo.down });
    for (k.shape, 0..) |sh, j| try out.interface.print("{s}{{\"threads\": {d}, \"smem\": {d}, \"cb\": {d}}}", .{ if (j == 0) "" else ", ", sh.threads, sh.smem, sh.cb });
    try out.interface.print("]}}\n", .{});
    try out.interface.flush();

    const xh = try a.alloc(u16, cap * X0.hidden);
    const xf = try a.alloc(f32, cap * X0.hidden);
    const pick_h = try a.alloc(i32, p_cap);
    const wts_h = try a.alloc(f32, p_cap);

    for (layer_ids) |li_| {
        const li: u32 = @intCast(li_);
        const X = try MX.Expect.of(cfg, sp, li);
        const t0 = std.Io.Timestamp.now(io, .awake);
        var lay = try MX.loadLayer(gpa, d, args[3], X);
        defer lay.deinit(gpa);
        const load_s = secs(io, t0);
        const avail = memAvail(a, io);
        try out.interface.print("{{\"layer\": {d}, \"path\": \"load\", \"slice_GB\": {d:.3}, \"load_s\": {d:.1}, \"mem_available_GiB\": {d:.1}}}\n", .{ li, @as(f64, @floatFromInt(lay.bytes)) / 1e9, load_s, avail });
        try out.interface.flush();
        if (avail < min_avail_gib) {
            std.log.err("MemAvailable {d:.1} GiB after the load is under --min-avail-gib {d:.1}: refusing to go on", .{ avail, min_avail_gib });
            return 3;
        }

        if (dump_dir) |dir| try dumpCheck(&b, &lay, &ck, dir, li, node);

        // the rows (N(0, 1) as bf16) and the layer's real routing on the host
        var prng = std.Random.DefaultPrng.init(seed + li);
        const rnd = prng.random();
        for (xh, xf) |*h, *f| {
            h.* = R.f32ToBf16(gauss(rnd));
            f.* = R.bf16ToF32(h.*);
        }
        const t1 = std.Io.Timestamp.now(io, .awake);
        const g = try R.gate(a, &ck, li, cfg.experts, X.hidden);
        R.route(xf, g, X.hidden, cfg.experts, cfg.top_k, cfg.routed_scaling, 0, cap, pick_h, wts_h);
        var counts = try a.alloc(u32, cfg.experts);
        @memset(counts, 0);
        for (pick_h, 0..) |e, j| if (j % slots != slots - 1) {
            counts[@intCast(e)] += 1;
        };
        var used: usize = 0;
        var busiest: u32 = 0;
        for (counts) |cnt| {
            used += @intFromBool(cnt > 0);
            busiest = @max(busiest, cnt);
        }
        try upload(d, b.x, std.mem.sliceAsBytes(xh));
        try upload(d, b.pick, std.mem.sliceAsBytes(pick_h));
        try upload(d, b.wts, std.mem.sliceAsBytes(wts_h));
        try out.interface.print("{{\"layer\": {d}, \"path\": \"route\", \"rows\": {d}, \"host_s\": {d:.1}, \"experts_used\": {d}, \"busiest\": {d}}}\n", .{ li, cap, secs(io, t1), used, busiest });
        try out.interface.flush();

        for (rows_list) |n| {
            // a clean pack: the rest rows of experts another pair computes are never written (assemble reads this
            // node's rows for them, zeros, as the Python bench's act_r)
            try d.check(d.api.cuMemsetD8Async(b.pack, 0, n * slots * (X.mainW() + X.restW()) * 2, s.handle), "cuMemsetD8Async");
            const med = try b.timed(&lay, n, geo, warmup, iters);
            const routed = med[0] + med[1] + med[2] + med[5] + med[7];
            const shared = med[3] + med[6];
            const fnn: f32 = @floatFromInt(n);
            try out.interface.print("{{\"layer\": {d}, \"rows\": {d}, \"path\": \"mx\", \"ms\": {{", .{ li, n });
            for (stage_names, 0..) |nm, j| try out.interface.print("{s}\"{s}\": {d:.3}", .{ if (j == 0) "" else ", ", nm, med[j] });
            try out.interface.print("}}, \"routed_ms\": {d:.3}, \"shared_ms\": {d:.3}, \"total_ms\": {d:.3}, \"us_row\": {{\"routed\": {d:.3}, \"total\": {d:.3}}}, \"target_routed_ms_4096\": 13.0}}\n", .{ routed, shared, routed + shared, routed * 1000 / fnn, (routed + shared) * 1000 / fnn });
            try out.interface.flush();
            if (check) try checkCase(&b, &lay, &ck, li, n, pick_h, wts_h, xf);
            if (tune and n == cap) {
                try out.interface.print("{{\"layer\": {d}, \"rows\": {d}, \"path\": \"tune\"", .{ li, n });
                for (MX.geos) |gg| {
                    const t = try b.timed(&lay, n, .{ .main = gg, .rest = gg, .down = gg }, 2, 5);
                    const desc = try MX.geoDesc(gg);
                    try out.interface.print(", \"{d}\": {{\"desc\": [{d}, {d}, {d}, {d}, {d}, {d}], \"gate_main\": {d:.3}, \"gate_rest\": {d:.3}, \"down\": {d:.3}}}", .{ gg, desc[0], desc[1], desc[2], desc[3], desc[4], desc[5], t[1], t[2], t[5] });
                }
                try out.interface.print("}}\n", .{});
                try out.interface.flush();
            }
        }
    }
    try out.interface.print("{{\"pass\": {}}}\n", .{b.all_pass});
    try out.interface.flush();
    return if (b.all_pass) 0 else 1;
}

fn secs(io: std.Io, t0: std.Io.Timestamp) f64 {
    return @as(f64, @floatFromInt(t0.durationTo(std.Io.Timestamp.now(io, .awake)).nanoseconds)) / 1e9;
}

fn memAvail(a: std.mem.Allocator, io: std.Io) f64 {
    var file = std.Io.Dir.cwd().openFile(io, "/proc/meminfo", .{}) catch return std.math.inf(f64);
    defer file.close(io);
    var buf: [4096]u8 = undefined;
    var r = file.readerStreaming(io, &buf);
    const text = r.interface.allocRemaining(a, .limited(1 << 20)) catch return std.math.inf(f64);
    const v = MX.memAvailable(text) orelse return std.math.inf(f64);
    return @as(f64, @floatFromInt(v)) / (1 << 30);
}

fn rowF64(a: std.mem.Allocator, h: []const u16) ![]f64 {
    const o = try a.alloc(f64, h.len);
    for (o, h) |*x, v| x.* = R.bf16ToF32(v);
    return o;
}

fn f32ToF64(a: std.mem.Allocator, h: []const f32) ![]f64 {
    const o = try a.alloc(f64, h.len);
    for (o, h) |*x, v| x.* = v;
    return o;
}

/// The case's outputs against float64 (bench_mx_zigsplit --check, plus the rest block, the shared slot, the combine).
fn checkCase(b: *Bench, lay: *const MX.Layer, ck: *R.Ckpt, li: u32, n: usize, pick_h: []const i32, wts_h: []const f32, xf: []const f32) !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const X = lay.x;
    const d = b.d;
    const sl = b.slots;
    const p = n * sl;
    const mw = X.mainW();
    const rw = X.restW();
    const hw = X.halfW();
    const dn = X.downN();
    const lim: f64 = b.cfg.swiglu_limit;
    const y = try a.alloc(f32, p * dn);
    try download(d, std.mem.sliceAsBytes(y), b.y);
    const pack = try a.alloc(u16, p * (mw + rw));
    try download(d, std.mem.sliceAsBytes(pack), b.pack);
    const xd = try a.alloc(u16, p * hw);
    try download(d, std.mem.sliceAsBytes(xd), b.xd_full);
    const outh = try a.alloc(f32, n * dn);
    try download(d, std.mem.sliceAsBytes(outh), b.outp);
    var nf_y: usize = 0;
    for (y) |v| nf_y += @intFromBool(!std.math.isFinite(v));
    var nf_pack: usize = 0;
    for (pack) |v| nf_pack += @intFromBool((v & 0x7f80) == 0x7f80);
    var nf_out: usize = 0;
    for (outh) |v| nf_out += @intFromBool(!std.math.isFinite(v));
    const rest_rows = pack[p * mw ..];
    var prng = std.Random.DefaultPrng.init(b.seed * 7919 + li * 31 + n);
    const rnd = prng.random();
    var A: R.Rel = .{};
    var AR: R.Rel = .{};
    var Y: R.Rel = .{};
    var nrest: usize = 0;
    var chosen: std.ArrayList(usize) = .empty;
    while (chosen.items.len < @min(b.pairs, n * (sl - 1))) {
        const pr = rnd.uintLessThan(usize, p);
        if (pr % sl == sl - 1) continue;
        var dup = false;
        for (chosen.items) |c| dup = dup or c == pr;
        if (!dup) try chosen.append(a, pr);
    }
    for (chosen.items) |pr| {
        const r = pr / sl;
        const e: usize = @intCast(pick_h[pr]);
        const rest = e % 2 == X.pair;
        const xdr = try a.alloc(f32, hw);
        for (xdr, xd[pr * hw ..][0..hw]) |*o, v| o.* = R.bf16ToF32(v);
        const ref = try R.pairRef(a, ck, li, X, e, xf[r * X.hidden ..][0..X.hidden], xdr, rest, lim);
        A.add(try rowF64(a, pack[pr * mw ..][0..mw]), ref.act);
        if (rest) {
            AR.add(try rowF64(a, rest_rows[pr * rw ..][0..rw]), ref.act_rest);
            nrest += 1;
        }
        Y.add(try f32ToF64(a, y[pr * dn ..][0..dn]), ref.y);
    }
    // the shared slot of two rows, against float64 from the slice's bf16 cuts (downloaded back from the device)
    const sw = X.sharedW();
    const sgu = try a.alloc(u16, 2 * sw * X.hidden);
    try download(d, std.mem.sliceAsBytes(sgu), lay.shared_gu);
    const sdn = try a.alloc(u16, dn * hw);
    try download(d, std.mem.sliceAsBytes(sdn), lay.shared_down);
    var SA: R.Rel = .{};
    var SY: R.Rel = .{};
    for ([_]usize{ 0, n - 1 }) |r| {
        const pr = r * sl + sl - 1;
        const xdr = try a.alloc(f32, hw);
        for (xdr, xd[pr * hw ..][0..hw]) |*o, v| o.* = R.bf16ToF32(v);
        const act = try a.alloc(f64, sw);
        const yr = try a.alloc(f64, dn);
        R.sharedRef(sgu[0 .. sw * X.hidden], sgu[sw * X.hidden ..], sdn, X.hidden, hw, xf[r * X.hidden ..][0..X.hidden], xdr, act, yr, lim);
        const got = try a.alloc(f64, sw);
        for (0..mw) |j| got[j] = R.bf16ToF32(pack[pr * mw + j]);
        for (mw..sw) |j| got[j] = R.bf16ToF32(rest_rows[pr * rw + j - mw]);
        SA.add(got, act);
        SY.add(try f32ToF64(a, y[pr * dn ..][0..dn]), yr);
    }
    // the combine of four rows against the float64 weighted sum of the same y
    var C: R.Rel = .{};
    for ([_]usize{ 0, n / 3, (2 * n) / 3, n - 1 }) |r| {
        const ref = try a.alloc(f64, dn);
        @memset(ref, 0);
        for (0..sl) |sj| for (ref, 0..) |*o, c| {
            o.* += @as(f64, wts_h[r * sl + sj]) * y[(r * sl + sj) * dn + c];
        };
        C.add(try f32ToF64(a, outh[r * dn ..][0..dn]), ref);
    }
    const ok = A.nonfinite_got == 0 and Y.nonfinite_got == 0 and AR.nonfinite_got == 0 and nf_y == 0 and nf_pack == 0 and nf_out == 0 and
        A.rel < 2e-2 and AR.rel < 2e-2 and Y.rel < 3e-2 and SA.rel < 2e-2 and SY.rel < 3e-2 and C.rel < 1e-5 and
        SA.nonfinite_got == 0 and SY.nonfinite_got == 0 and C.nonfinite_got == 0 and A.nonfinite_ref + Y.nonfinite_ref == 0;
    b.all_pass = b.all_pass and ok;
    const w = &b.out.interface;
    try w.print("{{\"layer\": {d}, \"rows\": {d}, \"path\": \"check\", \"pairs\": {d}, \"rest_pairs\": {d}, \"act_rel\": {f}, \"act_max_abs\": {f}, \"rest_rel\": {f}, \"y_rel\": {f}, \"y_max_abs\": {f}, \"shared_act_rel\": {f}, \"shared_y_rel\": {f}, \"combine_rel\": {f}, \"nonfinite\": {{\"y\": {d}, \"pack\": {d}, \"out\": {d}, \"ref\": {d}}}, \"ckpt_read_MB\": {d:.1}, \"pass\": {}}}\n", .{ li, n, chosen.items.len, nrest, num(A.rel), num(A.max_abs), num(AR.rel), num(Y.rel), num(Y.max_abs), num(SA.rel), num(SY.rel), num(C.rel), nf_y, nf_pack, nf_out, A.nonfinite_ref + Y.nonfinite_ref, @as(f64, @floatFromInt(ck.read_bytes)) / 1e6, ok });
    try w.flush();
}

/// The Python bench's sampled pairs (its --dump, converted) through these kernels: bit for bit.
fn dumpCheck(b: *Bench, lay: *const MX.Layer, ck: *R.Ckpt, dir: []const u8, li: u32, node: u32) !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const X = lay.x;
    const d = b.d;
    const io = b.io;
    const mw = X.mainW();
    const rw = X.restW();
    const hw = X.halfW();
    const dn = X.downN();
    var dh = std.Io.Dir.cwd().openDir(io, dir, .{ .iterate = true }) catch |err| {
        std.log.err("--dump-check {s}: {s}", .{ dir, @errorName(err) });
        return err;
    };
    defer dh.close(io);
    var it = dh.iterate();
    var pre_b: [32]u8 = undefined;
    const pre = try std.fmt.bufPrint(&pre_b, "check-L{d}-R", .{li});
    var found: usize = 0;
    while (try it.next(io)) |ent| {
        if (!std.mem.startsWith(u8, ent.name, pre) or !std.mem.endsWith(u8, ent.name, ".bin")) continue;
        const bytes = try dh.readFileAlloc(io, ent.name, a, .limited(1 << 26));
        const dmp = try R.parseDump(a, bytes, .{ X.hidden, mw, hw, dn });
        if (dmp.node != node or dmp.layer != li) continue;
        found += 1;
        const m = dmp.items.len;
        // x rows and xd rows: bf16 exactly (fp32 copies of bf16 values)
        const xh = try a.alloc(u16, m * X.hidden);
        const xdh = try a.alloc(u16, m * hw);
        const picks = try a.alloc(i32, m);
        const rpicks = try a.alloc(i32, m);
        var inexact: usize = 0;
        for (dmp.items, 0..) |item, j| {
            for (item.x, 0..) |v, c| {
                const u: u32 = @bitCast(v);
                inexact += @intFromBool(u & 0xffff != 0);
                xh[j * X.hidden + c] = @intCast(u >> 16);
            }
            for (item.xd, 0..) |v, c| {
                const u: u32 = @bitCast(v);
                inexact += @intFromBool(u & 0xffff != 0);
                xdh[j * hw + c] = @intCast(u >> 16);
            }
            picks[j] = @intCast(item.e);
            rpicks[j] = lay.host_rest_map[item.e];
        }
        try upload(d, b.x, std.mem.sliceAsBytes(xh));
        try upload(d, b.xd_full, std.mem.sliceAsBytes(xdh));
        try upload(d, b.pick, std.mem.sliceAsBytes(picks));
        try upload(d, b.sc.rpick, std.mem.sliceAsBytes(rpicks));
        const act_d = b.pack;
        const rest_d = b.pack + m * mw * 2;
        try d.check(d.api.cuMemsetD8Async(b.pack, 0, m * (mw + rw) * 2, b.s.handle), "cuMemsetD8Async");
        // one slot a row (pair j reads x row j), the routed plan and the rest plan over these m pairs
        try MX.route(b.k, b.s, b.pick, m, b.sc.main);
        try MX.gemm(b.k, b.s, .{ .epi = .gate_up, .geo = 2, .x = b.x, .x_stride = X.hidden, .slots = 1, .w = lay.gu_main, .scale = lay.gu_main_scale, .kg = X.hidden / 32, .nb = mw / 32, .out = act_d, .limit = b.cfg.swiglu_limit, .skip = -1, .pairs = m }, b.sc.main);
        try MX.route(b.k, b.s, b.sc.rpick, m, b.sc.rest);
        try MX.gemm(b.k, b.s, .{ .epi = .gate_up, .geo = 2, .x = b.x, .x_stride = X.hidden, .slots = 1, .w = lay.gu_rest, .scale = lay.gu_rest_scale, .kg = X.hidden / 32, .nb = rw / 32, .out = rest_d, .limit = b.cfg.swiglu_limit, .skip = @intCast(X.n_rest), .pairs = m }, b.sc.rest);
        try MX.gemm(b.k, b.s, .{ .epi = .down, .geo = 4, .x = b.xd_full, .x_stride = hw, .slots = 0, .w = lay.down, .scale = lay.down_scale, .kg = hw / 32, .nb = dn / 32, .out = b.y, .limit = 0, .skip = -1, .pairs = m }, b.sc.main);
        try b.s.synchronize();
        const act = try a.alloc(u16, m * mw);
        try download(d, std.mem.sliceAsBytes(act), act_d);
        const ract = try a.alloc(u16, m * rw);
        try download(d, std.mem.sliceAsBytes(ract), rest_d);
        const y = try a.alloc(f32, m * dn);
        try download(d, std.mem.sliceAsBytes(y), b.y);
        var act_diff: usize = 0;
        var rest_diff: usize = 0;
        var rest_items: usize = 0;
        var y_diff: usize = 0;
        var A: R.Rel = .{};
        var Y: R.Rel = .{};
        for (dmp.items, 0..) |item, j| {
            for (0..mw) |c| act_diff += @intFromBool(@as(u32, act[j * mw + c]) << 16 != @as(u32, @bitCast(item.act[c])));
            if (item.e % 2 == X.pair) {
                rest_items += 1;
                for (0..rw) |c| rest_diff += @intFromBool(@as(u32, ract[j * rw + c]) << 16 != @as(u32, @bitCast(item.xd[2 * mw + c])));
            }
            for (0..dn) |c| y_diff += @intFromBool(@as(u32, @bitCast(y[j * dn + c])) != @as(u32, @bitCast(item.y[c])));
            const ref = try R.pairRef(a, ck, li, X, item.e, item.x, item.xd, false, b.cfg.swiglu_limit);
            A.add(try f32ToF64(a, item.act), ref.act);
            Y.add(try f32ToF64(a, item.y), ref.y);
        }
        const ra = R.checkNumber(a, dmp.check, "act_rel") orelse std.math.nan(f64);
        const ry = R.checkNumber(a, dmp.check, "y_rel") orelse std.math.nan(f64);
        const ref_ok = @abs(A.rel - ra) <= 1e-6 * @abs(ra) and @abs(Y.rel - ry) <= 1e-4 * @abs(ry);
        const equal = act_diff == 0 and rest_diff == 0 and y_diff == 0 and inexact == 0;
        b.all_pass = b.all_pass and equal and ref_ok;
        try b.out.interface.print("{{\"layer\": {d}, \"rows\": {d}, \"path\": \"dump\", \"file\": \"{s}\", \"items\": {d}, \"rest_items\": {d}, \"equal\": {}, \"diff\": {{\"act\": {d}, \"rest\": {d}, \"y\": {d}, \"inexact_inputs\": {d}}}, \"ref\": {{\"act_rel\": {f}, \"recorded_act_rel\": {f}, \"y_rel\": {f}, \"recorded_y_rel\": {f}, \"agree\": {}}}}}\n", .{ li, dmp.rows, ent.name, m, rest_items, equal, act_diff, rest_diff, y_diff, inexact, num(A.rel), num(ra), num(Y.rel), num(ry), ref_ok });
        try b.out.interface.flush();
    }
    if (found == 0) {
        std.log.err("--dump-check {s}: no check-L{d}-R*.bin of node {d}", .{ dir, li, node });
        b.all_pass = false;
    }
}
