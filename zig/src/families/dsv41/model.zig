//! One TP rank of DeepSeek-V4.1-Flash loaded for serving: the weights, kernels, caches and buffers the layer gate sets
//! up (layer_main.zig), for a pool of draft.max_streams streams (MultiDecoder's slots: each stream's window rings,
//! positional stores and drafter rings in its slot, its compressed rows in its extent of one shared plane), and the
//! primitives a lane round runs on them. Both ranks run the same primitives in the same order (lanes.zig sends them
//! to rank 1): a prompt's chunks into its stream's slot and extent (fill), one round over every stream's window and
//! the windows' taps absorbed (verify), and the drafter's batched pass (pass). Every primitive is the gates' own code
//! path (prompt.zig, round.zig, draft.zig) with no checks.
const std = @import("std");
const cuda = @import("cuda");
const Config = @import("config.zig").Config;
const rank_cache = @import("rank_cache.zig");
const weights = @import("weights.zig");
const prompt = @import("prompt.zig");
const prompt2d = @import("prompt2d.zig");
const round = @import("round.zig");
const draft = @import("draft.zig");
const engram = @import("engram.zig");
const engram_io = @import("engram_io.zig");
const engram_aio = @import("engram_aio.zig");
const engram_pf = @import("engram_pf.zig");
const exl3_prefill = @import("exl3_prefill.zig");
const exl3_linear = @import("exl3_linear.zig");
const exl3_experts = @import("exl3_experts.zig");
const exl3_experts2d = @import("exl3_experts2d.zig");
const exl3_prompt = @import("exl3_prompt.zig");
const ops_mod = @import("ops.zig");
const exact = @import("exact.zig");
const cublas = @import("cublas.zig");
const comm_mod = @import("comm.zig");
const link = @import("link.zig");
const ring2d = @import("ring2d.zig");
const rdma = @import("rdma.zig");
const pprof_mod = @import("pprof.zig");
const ovl = @import("ovl.zig");
const attn = @import("attn.zig");
const router = @import("router.zig");
const p16 = @import("p16.zig");

/// The served prompt chunk (TF_DS_PREFILL_CHUNK): a prompt runs in chunks that start at its multiples.
pub const chunk_rows = 2048;
/// A prompt pass's rows at most (TF_DS_PROMPT_CHUNK=4096: two chunks a pass, passRows).
pub const max_pass_rows = 2 * chunk_rows;

/// TF_DS_PROMPT_CHUNK: a prompt pass's rows at most, 2,048 (unset: the served chunk, every pass one chunk) or 4,096
/// (passRows: two chunks in one pass where both end at or before the replay).
pub fn promptChunk() !usize {
    const v = std.c.getenv("TF_DS_PROMPT_CHUNK") orelse return chunk_rows;
    const x = std.fmt.parseInt(usize, std.mem.span(v), 10) catch return error.BadPromptChunk;
    if (x != chunk_rows and x != max_pass_rows) return error.BadPromptChunk;
    return x;
}

/// TF_DS_PROMPT_SP=rows: SP rows on the four-node split's encoder-only prompt chunks (prompt.spFits): each node mixes
/// its quarter of the rows, the partials go to the rows' owners and x is all-gathered; catRank by send / recv.
/// TF_DS_PROMPT_SP=rows2: the same, and also the chunk that reaches the replay and any row count (prompt.spFits2).
pub fn spMode() prompt.SpMode {
    const v = std.c.getenv("TF_DS_PROMPT_SP") orelse return .off;
    const x = std.mem.span(v);
    if (std.mem.eql(u8, x, "rows")) return .rows;
    if (std.mem.eql(u8, x, "rows2")) return .rows2;
    return .off;
}

/// Whether TF_DS_PROMPT_SP asks for SP rows (rows or rows2).
pub fn spRows() bool {
    return spMode() != .off;
}

/// TF_DS_PROMPT_ENGRAM=sp (four-node split, with SP rows): on an SP chunk an Engram layer's sum is made for the rows
/// each node owns alone (Two.engramSumSp), the only rows its gate reads; the same adds of the same words. Unset / ""
/// / "all": every row (Two.engramSum), as served.
pub fn promptEngram() !bool {
    const v = std.c.getenv("TF_DS_PROMPT_ENGRAM") orelse return false;
    const s = std.mem.span(v);
    if (s.len == 0 or std.mem.eql(u8, s, "all")) return false;
    if (std.mem.eql(u8, s, "sp")) return true;
    return error.BadPromptEngram;
}

/// TF_DS_PROMPT_A2A=shift (four-node split, with SP rows): toOwners' all-to-all as three shift steps, one NCCL group
/// each (send to g + k, receive from g - k), so no node's port takes two senders at once; the same transfers. Unset /
/// "" / "group": one group of every send and receive, as served.
pub fn promptA2a() !bool {
    const v = std.c.getenv("TF_DS_PROMPT_A2A") orelse return false;
    const s = std.mem.span(v);
    if (s.len == 0 or std.mem.eql(u8, s, "group")) return false;
    if (std.mem.eql(u8, s, "shift")) return true;
    return error.BadPromptA2a;
}

/// TF_DS_PROMPT_OVL (four-node split, with SP rows; ovl.zig): some of an SP chunk's exchanges on a comm stream beside
/// the compute stream, under compute that does not read them yet: a comma list of engram (needs TF_DS_PROMPT_ENGRAM=sp),
/// xag, wob, prio, pack (needs TF_DS_EXL3_PROMPT=v2 and TF_DS_2D_GU=parity), dn or dnc (need v2), fxag, or all
/// (engram, xag, wob). Unset / "" / "off": none, as served. Any other value refuses to start.
pub fn promptOvl() !ovl.Levers {
    const v = std.c.getenv("TF_DS_PROMPT_OVL") orelse return .{};
    return ovl.parse(std.mem.span(v));
}

/// TF_DS_PROMPT_ATTN (four-node split, with SP rows; attn.zig): front, isplit, tsel, all. Unset / "" / "off": as served.
pub fn promptAttn() !attn.Levers {
    const v = std.c.getenv("TF_DS_PROMPT_ATTN") orelse return .{};
    return attn.parse(std.mem.span(v));
}

/// TF_DS_PROMPT_ROUTER (four-node split, with SP rows; router.zig): own, own,hide, all. Unset / "" / "off": as served.
pub fn promptRouter() !router.Levers {
    const v = std.c.getenv("TF_DS_PROMPT_ROUTER") orelse return .{};
    return router.parse(std.mem.span(v));
}

/// TF_DS_PROMPT_OVL=pack3: each encoder layer's gate / up epilogue tables (svh_g, svh_u, suh_d [E, width], read at
/// row stride width) copied once into column halves [3, 2, E, width / 2], so a half's epilogue reads the same values at
/// stride width / 2. Returns the bytes held.
fn packTables(d: *const cuda.Driver, w: *weights.Weights, o: *const ops_mod.Ops, s: cuda.Stream, layers: usize) !usize {
    var total: usize = 0;
    for (w.layers[0..@min(layers, w.layers.len)]) |*lay| {
        const ex = &lay.experts;
        const ig: usize = ex.width;
        const half = ovl.packHalf(ig) orelse return error.OvlPackShape;
        const ne: usize = ex.count;
        const bytes = 6 * ne * half * 2;
        var b = try cuda.DeviceBuffer.alloc(d, bytes);
        errdefer b.free();
        for ([_]u64{ ex.svh_g, ex.svh_u, ex.suh_d }, 0..) |src, t| for (0..2) |h| {
            try o.copyRows(s, src + h * half * 2, ig * 2, b.ptr + (2 * t + h) * ne * half * 2, half * 2, half * 2, ne);
        };
        try w.buffers.append(w.gpa, b);
        ex.gu_halves = b.ptr;
        total += bytes;
    }
    try s.synchronize();
    return total;
}

/// TF_DS_PROMPT_OVL=dn: each encoder layer's svh_d [E, dn] (down's output signs, which the served combine reads at row
/// stride dn) copied once into its column halves [2, E, dn / 2] (held by the weights), so a half's combine reads the
/// same values at stride dn / 2. Returns the bytes held.
fn dnTables(d: *const cuda.Driver, w: *weights.Weights, o: *const ops_mod.Ops, s: cuda.Stream, layers: usize) !usize {
    var total: usize = 0;
    for (w.layers[0..@min(layers, w.layers.len)]) |*lay| {
        const ex = &lay.experts;
        const dn: usize = ex.down_n;
        const half = ovl.dnHalf(dn) orelse return error.OvlDnShape;
        const ne: usize = ex.count;
        const bytes = 2 * ne * half * 2;
        var b = try cuda.DeviceBuffer.alloc(d, bytes);
        errdefer b.free();
        for (0..2) |h| try o.copyRows(s, ex.svh_d + h * half * 2, dn * 2, b.ptr + h * ne * half * 2, half * 2, half * 2, ne);
        try w.buffers.append(w.gpa, b);
        ex.svh_d_halves = b.ptr;
        total += bytes;
    }
    try s.synchronize();
    return total;
}

/// TF_DS_PROMPT_OVERLAP=1 (four-node split, with TF_DS_PROMPT_CHUNK=4096): an encoder-only pass's two chunks run one
/// layer apart, the second on a stream, buffer set and NCCL communicator of its own (Overlap), launched from this one
/// thread in a fixed order, so one chunk's exchanges run under the other's kernels. Each chunk runs the 2,048-row
/// chunk's own launches.
pub fn promptOverlap() bool {
    const v = std.c.getenv("TF_DS_PROMPT_OVERLAP") orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "1");
}

/// The second chunk's own state of an overlapped pass (promptOverlap): its stream, NCCL communicator (the same four
/// ranks), 2D exchange buffers, cuBLAS handle, engine and chunk buffers; an event at each of the first chunk's layer
/// ends (the second chunk's layer waits for the first's: its window keys, compressed rows and positional stores) and
/// one at the second chunk's end (the main stream waits for it before anything after the pass).
pub const Overlap = struct {
    stream: cuda.Stream,
    comm: comm_mod.Comm,
    two: prompt2d.Two,
    blas: cublas.Blas,
    eng: prompt.Engine,
    ch: prompt.Chunk,
    ev: []cuda.Event,
    done: cuda.Event,
};

/// The rows of the prompt pass from `start` (a chunk end) of a prompt of `len` rows with the bounded replay at
/// `replay`: two whole chunks when passes take 4,096 rows and both end at or before the replay (an encoder-only pass,
/// every decoder row in a chunk of its own as on 2,048-row passes), else the chunk from start.
pub fn passRows(pass: usize, start: usize, len: usize, replay: usize) usize {
    if (pass > chunk_rows and start % chunk_rows == 0 and start + pass <= replay and start + pass <= len) return pass;
    return @min(chunk_rows, len - start);
}
/// The pool's streams (--parallel 4).
pub const max_streams = draft.max_streams;
/// The pool's streams when unset (--parallel 4).
pub const max_streams_default = draft.default_streams;
/// The RDMA ring's slot (the served 4.25 MiB: a 16-row round's logits half fits).
pub const ring_bytes = 4456448;

pub const Options = struct {
    model_dir: []const u8, // config.json
    cache_dir: []const u8, // the lane's per-rank weight file (TF_DS_RANK_CACHE)
    kit_dir: []const u8, // a gate dir's layout: aot/, cubins/, rope-{plain,compressed}-{cos,sin}.f32, engram.json
    aot_dir: ?[]const u8 = null, // the recorded Triton set (aot.json and its cubins; null: kit_dir/aot)
    rank: u32,
    world: u32,
    master: [4]u8,
    port: u16,
    engram_dir: ?[]const u8 = null, // the Engram tables (none: a checkpoint without Engram layers)
    token_map: ?[]const u8 = null, // the compressed token map (the lane's JSON cache)
    pool: usize = 1 << 18, // positions of the streams' shared plane (their extents together)
    context: usize = 0, // a stream's extent at most (its prompt, budget and a round's rows; 0: the pool)
    drafts: bool = true, // load the DSpark drafter
    rdma_devices: ?[]const u8 = null, // the decode gathers over RDMA on these devices (comma separated), else NCCL: TP2's
    // ring (rdma.zig), or at world 4 the 2D split's decode-size exchanges over its rings (ring2d.zig)
    rdma_kernels: ?[]const u8 = null, // world 4: the rings' kernels' image (rdma_gather.cu's fatbin; null: the embedded one)
    graphs: bool = false, // the rounds' stretches as CUDA graphs (round.Graphs)
    side: bool = false, // the mixes' side work on a stream of its own (round.Round.useSide)
    prefetch: bool = false, // the paced L2 prefetch (round.Round.usePrefetch)
    engram_aio: bool = false, // a round's Engram reads by Linux AIO on O_DIRECT (else the reader pool)
    arena_bytes: usize = 0, // the device arena's first block (0: it grows from 256 MiB blocks as the buffers ask)
    round_rows: usize = round.default_rows, // a round's rows at most (TF_DS_ROUND_ROWS; the four-node lane's 48)
    streams: usize = draft.default_streams, // the pool's streams (--parallel; the four-node lane's 16)
    round_ms: ?[]const f64 = null, // the lane core's round costs by rows (TF_DS_ROUND_MS; null: the served ROUND_MS)
    vision_bias: ?[]const u8 = null, // gate_bias_vl.safetensors (null: the kit's vision/): the image-span routing bias
};

/// TF_DS_2D_GU=parity (the served switch): the 2D split's experts with half the gate / up blocks a pair and the ninth by
/// expert parity (plan.Split.parity); the rank cache must be one written for it.
pub fn parityGU() bool {
    const v = std.c.getenv("TF_DS_2D_GU") orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "parity");
}

/// TF_RDMA_GID: the RDMA rings' source GID index (the port's RoCE v2 IPv4 entry); 5 as served on the authors' nodes.
pub fn rdmaGid() u8 {
    const v = std.c.getenv("TF_RDMA_GID") orelse return 5;
    return std.fmt.parseInt(u8, std.mem.span(v), 10) catch 5;
}

/// The parity split's decode gate / up in one launch (experts_par.cu, the same bits); TF_DS_PAR_ONE=0: the served
/// build's launches (main, then the rest's grouping and gate / up, beside it on a stream of its own with --side).
pub fn parOne() bool {
    const v = std.c.getenv("TF_DS_PAR_ONE") orelse return true;
    return !std.mem.eql(u8, std.mem.span(v), "0");
}

pub const Model = struct {
    gpa: std.mem.Allocator,
    host: std.heap.ArenaAllocator, // allocations that live as long as the model
    io: std.Io,
    ctx: *const cuda.Context,
    stream: cuda.Stream,
    cfg: Config,
    rank: u32,
    world: u32,
    streams: usize, // the pool's streams (Options.streams)
    round_ms: []const f64, // the lane core's round costs by rows (Options.round_ms, else draft.round_ms)
    pool_cap: usize,
    ctx_cap: usize, // a stream's extent at most (Options.context, else the pool)
    cache_dir: std.Io.Dir,
    cache_file: std.Io.File,
    ix: rank_cache.Index,
    w: weights.Weights,
    fan: prompt2d.Fan,
    comm: comm_mod.Comm,
    rdma_mod: ?cuda.Module = null, // the RDMA kernels (TP2's ring or world 4's rings)
    ring: ?*rdma.Ring = null,
    set: cuda.aot.Set,
    pf: exl3_prefill.Kernels,
    lg: exl3_linear.Kernels,
    exk: exl3_experts.Kernels,
    ops: ops_mod.Ops,
    ex: exact.Exact,
    par: ?exl3_experts2d.ParKernels, // the parity split's one-launch gate / up (parOne)
    x2: ?exl3_prompt.Kernels = null, // TF_DS_EXL3_PROMPT=v2 (exl3_prompt.zig), a 2D node only
    rope_buf: cuda.DeviceBuffer,
    arena: prompt.Arena,
    eng: prompt.Engine,
    ch: prompt.Chunk,
    caches: prompt.Caches,
    blas: cublas.Blas,
    rings: []u64, // each layer's window rings, max_streams slots
    ring_view: []u64, // a filling stream's slot of them
    rd: round.Round,
    graphs: ?round.Graphs = null,
    pgraphs: ?draft.PassGraphs = null, // the drafter pass as CUDA graphs (with graphs)
    dpool: ?draft.Pool = null,
    dr: ?draft.Drafter = null,
    tables: ?engram_io.Tables = null,
    epool: ?*engram_io.Pool = null,
    aio: ?*engram_aio.Aio = null,
    paio: ?*engram_aio.Aio = null, // the prompt chunks' Engram reads, on a ring of their own
    eh: ?prompt.EngramHost = null,
    epf: ?*engram_pf.Prefetch = null, // TF_DS_ENGRAM_PREFETCH: a prompt's Engram rows read chunks ahead (engram_pf.zig)
    host_pos: []i64,
    ids64: []i64,
    vocab: usize,
    logits: []f32, // round.max_rows rows of logits on the host
    amax: u64, // u32 [round.max_rows]: a round's greedy tokens on the device (roundArgmax)
    amax_host: [round.max_rows]u32 = undefined,
    // the stream a fill is running: its slot and its view of the caches (its extent from base)
    fill_slot: usize = 0,
    // the prompt a fill's chunks come from (the caller's, set for the fill) and the thread reading the next chunk's
    // Engram rows during this one (fillChunk), joined before the prompt can change (joinAhead)
    fill_seq: []const i32 = &.{},
    ahead: ?std.Thread = null,
    view: prompt.Caches = undefined,
    // the filling prompt's image spans and their rows (setImages; cleared by fillBegin)
    spans: std.ArrayList(prompt.Span) = .empty,
    img_buf: ?cuda.DeviceBuffer = null,
    // profiling (lanes' --profile): the last round's forward and absorb, synchronized apart (ns)
    prof: bool = false,
    timer: ?*round.PhaseTimer = null, // a profile's GPU time by phase (eager rounds)
    ptimer: ?*round.PhaseTimer = null, // the same for prompt chunks (fills), a chunk a 'round'
    t_enqueue: u64 = 0, // the host's time to issue the forward (the GPU idle before it)
    // Model.useClock: the rounds' GPU clock gaps summed (ns; gap i = clock i + 1 - clock i: the forward's start to
    // the first stretch, then each stretch, then the wait before the next), and the rounds counted
    clk_ph: [256]u64 = @splat(0), // by the phase of the clock that ends it
    clk_cnt: [256]u64 = @splat(0),
    clk_rounds: u64 = 0,
    t_forward: u64 = 0,
    t_absorb: u64 = 0,
    // world 4: the 2D split's exchanges (prompt2d.zig, TP4's), with the RDMA rings when asked
    two: ?prompt2d.Two = null,
    pass_rows: usize = chunk_rows, // a prompt pass's rows at most (promptChunk)
    ov: ?*Overlap = null, // the overlapped passes' second chunk (promptOverlap)
    sp_rows: bool = false, // SP rows on encoder-only chunks (spRows; world 4)
    sp_mode: prompt.SpMode = .off, // which chunks take them (spMode)
    pprof: ?*pprof_mod.Prof = null, // TF_DS_PROMPT_PROF=1: the prompt chunks' GPU time by phase
    rdma_rings: ?ring2d.Rings = null,
    ovl_side: ?ovl.Side = null, // TF_DS_PROMPT_OVL: the comm stream, its events and the launch order (promptOvl)
    attn_st: attn.State = .{}, // TF_DS_PROMPT_ATTN: its levers (promptAttn; the engine points here when any is on)
    attn_k: ?attn.Kernels = null, // its tsel kernel (attn_prompt.cu)
    router_st: router.State = .{}, // TF_DS_PROMPT_ROUTER: its levers, kernels and calibrated split-K (promptRouter)
    router_k: ?router.Kernels = null, // router_own.cu
    p16_k: ?p16.Kernels = null, // TF_DS_PROMPT_P16=1: p16.cu (the SP chunks' owner exchange as bf16)

    /// Loads rank o.rank's weights from the lane's rank cache, links the ranks (rank 0 listens on o.port), opens NCCL
    /// and sets up every buffer. Needs `ctx` current on this thread.
    pub fn open(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, o: Options) !*Model {
        if (o.world != 2 and o.world != 4) return error.NotPortedYet; // TP2, or the four-node 2D split (prompt2d.zig)
        const m = try gpa.create(Model);
        errdefer gpa.destroy(m);
        m.* = undefined;
        m.gpa = gpa;
        if (std.c.getenv("TF_DS_REPLAY_FLOOR")) |v| prompt.replay_floor = std.mem.eql(u8, std.mem.span(v), "1"); // every rank alike
        m.spans = .empty;
        m.img_buf = null;
        m.host = std.heap.ArenaAllocator.init(gpa);
        errdefer m.host.deinit();
        const a = m.host.allocator();
        m.io = io;
        m.ctx = ctx;
        m.rank = o.rank;
        m.world = o.world;
        if (o.streams == 0 or o.streams > max_streams) return error.BadStreams;
        m.streams = o.streams;
        if (std.c.getenv("TF_DS_L2_SKIP")) |v| round.l2_skip = std.fmt.parseInt(u8, std.mem.span(v), 10) catch 0;
        for ([_][:0]const u8{ "TF_DS_L2_ATTN_MB", "TF_DS_L2_MOE_MB", "TF_DS_L2_NEXT_MB", "TF_DS_L2_STRETCH_END_MB" }, 0..) |name, i| {
            if (std.c.getenv(name)) |v| round.l2_mb[i] = std.fmt.parseInt(u64, std.mem.span(v), 10) catch round.l2_mb[i];
        }
        if (std.c.getenv("TF_EXL3_L2_DELAY_NS")) |v| ops_mod.l2_delay_ns = std.fmt.parseInt(c_longlong, std.mem.span(v), 10) catch ops_mod.l2_delay_ns;
        if (std.c.getenv("TF_EXL3_L2_RATE_GBPS")) |v| ops_mod.l2_rate_gbps = std.fmt.parseFloat(f64, std.mem.span(v)) catch ops_mod.l2_rate_gbps;
        m.round_ms = o.round_ms orelse &draft.round_ms;
        if (m.round_ms.len == 0) return error.BadRoundMs;
        m.pool_cap = o.pool;
        m.ctx_cap = if (o.context != 0) @min(o.context, o.pool) else o.pool;
        m.clearState();
        m.pass_rows = try promptChunk();
        m.sp_rows = spRows();
        m.sp_mode = spMode();
        const d = ctx.d;
        m.cfg = try Config.read(a, io, o.model_dir);
        const c = &m.cfg;
        m.stream = try cuda.Stream.init(d, true);
        errdefer m.stream.deinit();

        // the weights (and the DSpark blocks) from the lane's rank cache
        m.cache_dir = try std.Io.Dir.cwd().openDir(io, o.cache_dir, .{ .iterate = true });
        errdefer m.cache_dir.close(io);
        const cache_name = try rank_cache.find(a, io, m.cache_dir, o.rank, o.world);
        m.ix = try rank_cache.open(gpa, io, m.cache_dir, cache_name);
        errdefer m.ix.deinit();
        m.cache_file = try m.cache_dir.openFile(io, cache_name, .{});
        errdefer m.cache_file.close(io);
        var sp = try prompt2d.split(o.rank, o.world);
        sp.parity = sp.pair != null and parityGU(); // TF_DS_2D_GU=parity: the balanced experts split
        const ovl_lv = try promptOvl(); // (a bad value refuses to start before anything loads)
        if (ovl_lv.any()) {
            if (sp.pair == null) return error.OvlNeedsTheFourNodeSplit;
            if (promptOverlap()) return error.OvlWithOverlap;
        }
        const attn_lv = try promptAttn(); // (likewise)
        if (attn_lv.sp() and sp.pair == null) return error.AttnNeedsTheFourNodeSplit;
        if (attn_lv.any() and promptOverlap()) return error.AttnWithOverlap;
        const router_lv = try promptRouter(); // (likewise)
        // TF_DS_PROMPT_P16 (tier 2, p16.zig; likewise): a bad value, or 1 without the four-node split and SP rows, refuses
        const p16_on = try p16.fromEnv();
        try p16.check(p16_on, sp.pair != null, m.sp_rows);
        if (router_lv.any()) {
            if (sp.pair == null) return error.RouterNeedsTheFourNodeSplit;
            if (promptOverlap()) return error.RouterWithOverlap;
            if (router_lv.hide and !ovl_lv.any()) return error.RouterHideNeedsOvl;
        }
        // TF_DS_ENGRAM_PREFETCH (stage ZE2; likewise): a prompt's Engram rows by a reader of their own, chunks ahead,
        // on an AIO ring of its own (so it needs the served --aio 1); not with the overlapped passes (their read-ahead)
        const epf_set = try engram_pf.fromEnv();
        if (epf_set.on()) {
            if (promptOverlap()) return error.EngramPrefetchWithOverlap;
            if (!o.engram_aio) return error.EngramPrefetchNeedsAio;
        }
        // TF_DS_ENGRAM_SPLIT (stage ZE4; likewise): column partners split each chunk's reads in the prefetch's reader,
        // so it needs the prefetch and the four-node split (partners g and g ^ 2)
        const es_set = try engram_pf.splitFromEnv();
        if (es_set.on()) {
            if (!epf_set.on()) return error.EngramSplitNeedsPrefetch;
            if (sp.pair == null or o.world != 4) return error.EngramSplitNeedsTheFourNodeSplit;
            if (o.engram_dir == null) return error.NoEngramTables;
        }
        m.w = try weights.load(gpa, d, .{ .cache = .{ .file = m.cache_file, .index = &m.ix, .io = io } }, c.*, sp, o.drafts);
        errdefer m.w.deinit();
        if (o.drafts and m.w.dspark == null) return error.NoDrafter;

        // the ranks' link and NCCL (rank 0 listens: rank 1 retries its connect while rank 0 loads)
        m.fan = try prompt2d.Fan.open(io, o.master, o.port, o.rank, o.world);
        errdefer m.fan.close();
        m.comm = try m.fan.comm(o.rank, o.world);
        errdefer m.comm.deinit();
        // overlapped passes: their second NCCL communicator, right after the first on every rank (the unique id goes
        // down the same links in the same order)
        var comm2: ?comm_mod.Comm = null;
        if (promptOverlap()) {
            if (o.world != 4 or m.pass_rows != max_pass_rows) return error.OverlapNeedsFourNodesAndPasses;
            comm2 = try m.fan.comm(o.rank, o.world);
        }
        errdefer if (comm2) |*x| x.deinit();
        // TP2: the decode gathers' RDMA ring (both ranks connect before either returns)
        if (sp.pair == null and o.rdma_devices != null) {
            const list = o.rdma_devices.?;
            if (!cuda.kernels.available) return error.NoKernelImages;
            m.rdma_mod = try cuda.Module.load(d, cuda.kernels.dsv41_rdma);
            var devices: std.ArrayList([]const u8) = .empty;
            var it = std.mem.splitScalar(u8, list, ',');
            while (it.next()) |dev| try devices.append(a, dev);
            m.ring = try openRing(gpa, d, try rdma.Kernels.load(m.rdma_mod.?), devices.items, o.rank, &m.comm, m.stream);
        }
        errdefer if (m.ring) |r| r.destroy(gpa);
        errdefer if (m.rdma_mod) |*x| x.unload();

        // kernels: the recorded Triton set, the served extension cubins, the torch-op images
        m.set = try cuda.aot.Set.load(gpa, io, d, ctx.device, o.aot_dir orelse try std.fs.path.join(a, &.{ o.kit_dir, "aot" }));
        errdefer m.set.deinit();
        m.pf = try exl3_prefill.Kernels.load(d, try readKit(a, io, o.kit_dir, "linear.cubin"));
        errdefer m.pf.unload();
        m.lg = try exl3_linear.Kernels.load(d, try readKit(a, io, o.kit_dir, "linear_grouped.cubin"));
        errdefer m.lg.unload();
        m.exk = try exl3_experts.Kernels.load(d, try readKit(a, io, o.kit_dir, "experts.cubin"), try readKit(a, io, o.kit_dir, "experts_cb.cubin"), exl3_linear.codebook_mul1);
        errdefer m.exk.unload();
        if (!cuda.kernels.available) return error.NoKernelImages;
        m.ops = try ops_mod.Ops.load(d, cuda.kernels.torch_pointwise, cuda.kernels.torch_movement, cuda.kernels.dsv41_ops);
        errdefer m.ops.unload();
        m.ex = try exact.Exact.load(d, cuda.kernels.dsv41_torch);
        errdefer m.ex.unload();
        m.par = if (sp.parity and parOne()) try exl3_experts2d.ParKernels.load(d, cuda.kernels.dsv41_experts_par) else null;
        errdefer if (m.par) |*x| x.unload();
        m.x2 = null;
        if (try exl3_prompt.enabled()) {
            if (sp.pair == null) return error.Exl3PromptNeedsTheFourNodeSplit;
            m.x2 = try exl3_prompt.Kernels.load(d, cuda.kernels.dsv41_exl3_prompt);
        }
        errdefer if (m.x2) |*x| x.unload();

        // the RoPE tables (the served build's bits: torch CPU pow and polar), every position a stream reaches
        const half = c.rope_dim / 2;
        const rope_rows = m.ctx_cap + round.max_rows + 1;
        m.rope_buf = try cuda.DeviceBuffer.alloc(d, 4 * rope_rows * half * 4);
        errdefer m.rope_buf.free();
        {
            const part = try gpa.alloc(u8, rope_rows * half * 4);
            defer gpa.free(part);
            for ([_][]const u8{ "rope-plain-cos.f32", "rope-plain-sin.f32", "rope-compressed-cos.f32", "rope-compressed-sin.f32" }, 0..) |name, j| {
                var f = try std.Io.Dir.cwd().openFile(io, try std.fs.path.join(a, &.{ o.kit_dir, name }), .{});
                defer f.close(io);
                if (try f.readPositionalAll(io, part, 0) != part.len) return error.ShortRopeTable; // a table of fewer rows than a stream's context
                try m.rope_buf.upload(j * part.len, part);
            }
        }
        const tbl = rope_rows * half * 4;
        const rope: prompt.Rope = .{ .cos = m.rope_buf.ptr, .sin = m.rope_buf.ptr + tbl };
        const rope_c: prompt.Rope = .{ .cos = m.rope_buf.ptr + 2 * tbl, .sin = m.rope_buf.ptr + 3 * tbl };

        m.arena = try prompt.Arena.init(d, o.arena_bytes);
        errdefer m.arena.deinit();
        // world 4 (layer_main's setup): the 2D exchanges' buffers, and with rdma_devices their rings (whose infos go
        // round over NCCL world 4)
        errdefer if (m.rdma_rings) |*x| x.close(gpa);
        errdefer if (m.p16_k) |*x| x.unload();
        if (sp.pair != null) {
            m.two = try prompt2d.Two.init(c, &m.w, &m.arena, sp, m.pass_rows);
            m.two.?.p2p_cat = m.sp_rows;
            // TF_DS_PROMPT_ENGRAM / TF_DS_PROMPT_A2A: only SP chunks take them (nothing changes without SP rows)
            m.two.?.engram_sp = try promptEngram();
            m.two.?.a2a_shift = try promptA2a();
            if (p16_on) {
                if (!cuda.kernels.available) return error.NoKernelImages;
                m.p16_k = try p16.Kernels.load(d, cuda.kernels.dsv41_p16);
                try m.two.?.useP16(&m.arena, &m.p16_k.?, m.pass_rows);
                std.log.info(p16.start_line, .{});
            }
            if (m.two.?.engram_sp) std.log.info("TF_DS_PROMPT_ENGRAM=sp: Engram sums on owned rows ({s})", .{if (m.sp_rows) "active on SP chunks" else "inactive: TF_DS_PROMPT_SP unset"});
            if (m.two.?.a2a_shift) std.log.info("TF_DS_PROMPT_A2A=shift: toOwners in three shift steps ({s})", .{if (m.sp_rows) "active on SP chunks" else "inactive: TF_DS_PROMPT_SP unset"});
            if (ovl_lv.any()) {
                // TF_DS_PROMPT_OVL: the engram lever splits engramSumSp (its plan), so it needs TF_DS_PROMPT_ENGRAM=sp
                if (ovl_lv.engram and !m.two.?.engram_sp) return error.OvlEngramNeedsEngramSp;
                // the expert side (stage ZX1) splits v2's launches: pack the parity split's pack, dn / dnc v2's down
                if (ovl_lv.experts() and m.x2 == null) return error.OvlExpertsNeedV2;
                if ((ovl_lv.pack or ovl_lv.pack3) and m.two.?.ir == 0) return error.OvlPackNeedsParity;
                if (ovl_lv.pack3) for (m.two.?.gu) |w| if (ovl.packHalf(w) == null) return error.OvlPackShape;
                if (ovl_lv.dn) for (m.two.?.dw) |w| if (ovl.dnHalf(w) == null) return error.OvlDnShape;
                m.ovl_side = try ovl.Side.init(d, ovl_lv);
                var nb: [64]u8 = undefined;
                std.log.info("TF_DS_PROMPT_OVL={s}: exchanges on a comm stream beside compute ({s})", .{ ovl_lv.names(&nb), if (m.sp_rows) "active on SP chunks" else "inactive: TF_DS_PROMPT_SP unset" });
                if (ovl_lv.pack3) {
                    const tb = try packTables(d, &m.w, &m.ops, m.stream, c.layers / 2);
                    std.log.info("TF_DS_PROMPT_OVL=pack3: gate / up epilogue tables by halves {d} KiB (layers 0-{d})", .{ tb >> 10, c.layers / 2 - 1 });
                }
                if (ovl_lv.dn) {
                    const tb = try dnTables(d, &m.w, &m.ops, m.stream, c.layers / 2);
                    std.log.info("TF_DS_PROMPT_OVL=dn: down's svh_d halves {d} KiB (layers 0-{d})", .{ tb >> 10, c.layers / 2 - 1 });
                }
            }
            if (o.rdma_devices) |devs| {
                const image: []const u8 = if (o.rdma_kernels) |path|
                    try std.Io.Dir.cwd().readFileAllocOptions(io, path, a, .limited(1 << 26), .@"16", null)
                else if (cuda.kernels.available) cuda.kernels.dsv41_rdma else return error.NoRdmaKernels;
                m.rdma_mod = try cuda.Module.load(d, image);
                var devices: std.ArrayList([]const u8) = .empty;
                var dit = std.mem.splitScalar(u8, devs, ',');
                while (dit.next()) |x| try devices.append(a, x);
                m.rdma_rings = try ring2d.Rings.open(gpa, d, try rdma.Kernels.load(m.rdma_mod.?), devices.items, o.rank, .{ .max_bytes = ring_bytes, .gid_index = rdmaGid() }, &m.comm, m.stream);
                m.two.?.rings = &m.rdma_rings.?;
            }
        }
        m.eng = .{ .d = d, .s = m.stream, .t = .{ .set = &m.set, .stream = m.stream }, .blas = undefined, .comm = &m.comm, .pf = &m.pf, .lin = &m.lg, .ex = &m.exk, .ops = &m.ops, .exact = &m.ex, .c = c, .w = &m.w, .world = sp.world, .plain = rope, .compressed = rope_c, .ring = m.ring, .two = if (m.two) |*t| t else null };
        m.eng.round_rows = o.round_rows;
        if (attn_lv.any()) {
            m.attn_st = .{ .lv = attn_lv };
            if (attn_lv.tsel) {
                if (!cuda.kernels.available) return error.NoKernelImages;
                m.attn_k = try attn.Kernels.load(d, cuda.kernels.dsv41_attn);
                m.attn_st.k = &m.attn_k.?;
            }
            m.eng.attn = &m.attn_st;
            var nb: [64]u8 = undefined;
            std.log.info("TF_DS_PROMPT_ATTN={s}: the attention's replicated work on owned rows ({s})", .{ attn_lv.names(&nb), if (m.sp_rows) "active on SP chunks" else "inactive: TF_DS_PROMPT_SP unset" });
        }
        if (router_lv.any()) {
            // (Chunk.init sizes the picks' gather staging from it; the kernels and the split-K come after cuBLAS opens)
            m.router_st = .{ .lv = router_lv };
            m.eng.router = &m.router_st;
        }
        if (m.par) |*x| m.eng.par = x;
        if (m.x2) |*x| {
            m.eng.x2 = x;
            if (exl3_prompt.wqCache()) {
                const cb = try exl3_prompt.cacheWq(d, &m.w, &m.pf, m.stream, c.layers / 2);
                std.log.info("TF_DS_EXL3_PROMPT=v2: experts v2, W_q cache {d} MiB (layers 0-{d})", .{ cb >> 20, c.layers / 2 - 1 });
            } else std.log.info("TF_DS_EXL3_PROMPT=v2: experts v2, no W_q cache (TF_DS_EXL3_WQ_CACHE=0)", .{});
        }
        m.ch = try prompt.Chunk.init(&m.eng, &m.arena, m.pass_rows, m.ctx_cap); // a stream's keys at most
        m.caches = try prompt.Caches.init(&m.eng, &m.arena, o.pool, m.streams);
        m.blas = try cublas.Blas.open(m.stream, m.ch.blas_ws);
        errdefer m.blas.close();
        m.eng.blas = &m.blas;
        errdefer if (m.router_k) |*x| x.unload();
        if (router_lv.any()) {
            // TF_DS_PROMPT_ROUTER: router_own.cu, and today's 2,048-row cuBLAS run's split-K found on this GPU and
            // library against the library's own call (layer 0's gate; every own-row variant checked; no match: no start)
            if (!cuda.kernels.available) return error.NoKernelImages;
            m.router_k = try router.Kernels.load(d, cuda.kernels.dsv41_router);
            try m.ops.f16ToF32(m.stream, m.w.layers[0].gate_w, m.ch.gate_f, c.experts * c.hidden);
            const cal = try router.calibrate(gpa, d, &m.router_k.?, &m.blas, m.stream, m.w.layers[0].gate_w, m.ch.gate_f, c.hidden, c.experts);
            m.router_st.k = &m.router_k.?;
            m.router_st.parts = cal.parts;
            var nb: [32]u8 = undefined;
            var pb: [96]u8 = undefined;
            var pn: usize = 0;
            const np: usize = @intCast(cal.parts.n);
            for (cal.parts.b[1 .. np + 1], 0..) |b, i| pn += (std.fmt.bufPrint(pb[pn..], "{s}{d}", .{ if (i == 0) "" else ",", b }) catch break).len;
            std.log.info("TF_DS_PROMPT_ROUTER={s}: the MoE router on owned rows, picks gathered; today's 2,048-row cuBLAS run is split-K {d} at k {s} (candidate {d}, checked) ({s})", .{ router_lv.names(&nb), cal.parts.n, pb[0..pn], cal.tried, if (m.sp_rows) "active on SP chunks" else "inactive: TF_DS_PROMPT_SP unset" });
        }
        m.rings = try a.alloc(u64, c.layers);
        m.ring_view = try a.alloc(u64, c.layers);
        for (m.rings) |*r| {
            r.* = try m.arena.take(m.streams * m.eng.ringBytes());
            try d.check(d.api.cuMemsetD8Async(r.*, 0, m.streams * m.eng.ringBytes(), m.stream.handle), "cuMemsetD8Async");
        }
        m.rd = try round.Round.init(&m.eng, &m.arena, a, m.ctx_cap); // a round's deepest bucket: a stream's context
        m.amax = try m.arena.take(round.max_rows * 4);
        if (o.graphs) m.graphs = round.Graphs.init(gpa);
        if (o.side) try m.rd.useSide(&m.eng);
        if (o.prefetch) try m.rd.usePrefetch(&m.eng);
        if (m.two) |t| if (t.ir != 0 and o.side and m.par == null) try m.rd.usePar(&m.eng); // parity in two launches: the rest beside the main gate / up
        if (o.drafts) {
            m.dpool = try draft.Pool.init(&m.eng, &m.arena, m.streams);
            m.dr = try draft.Drafter.init(&m.eng, &m.arena, sp);
            if (o.graphs) {
                m.pgraphs = draft.PassGraphs.init(gpa);
                m.dr.?.graphs = &m.pgraphs.?;
            }
        }
        if (comm2) |cm| {
            // the overlapped passes' second chunk: the same engine on its own stream, communicator, exchange buffers,
            // cuBLAS handle and 2,048-row chunk buffers
            const ov = try gpa.create(Overlap);
            errdefer gpa.destroy(ov);
            ov.stream = try cuda.Stream.init(d, true);
            ov.comm = cm;
            comm2 = null;
            ov.two = try prompt2d.Two.init(c, &m.w, &m.arena, sp, chunk_rows);
            ov.two.p2p_cat = m.sp_rows;
            ov.two.engram_sp = m.two.?.engram_sp;
            ov.two.a2a_shift = m.two.?.a2a_shift;
            if (m.p16_k) |*k| try ov.two.useP16(&m.arena, k, chunk_rows);
            ov.eng = m.eng;
            ov.eng.s = ov.stream;
            ov.eng.t.stream = ov.stream;
            ov.eng.comm = &ov.comm;
            ov.eng.two = &ov.two;
            ov.ch = try prompt.Chunk.init(&ov.eng, &m.arena, chunk_rows, m.ctx_cap);
            ov.blas = try cublas.Blas.open(ov.stream, ov.ch.blas_ws);
            ov.eng.blas = &ov.blas;
            ov.ev = try a.alloc(cuda.Event, c.layers / 2 + 1);
            for (ov.ev) |*x| x.* = try cuda.Event.init(d, false);
            ov.done = try cuda.Event.init(d, false);
            try ov.stream.synchronize();
            m.ov = ov;
        }
        try m.stream.synchronize();
        try d.check(d.api.cuCtxSynchronize(), "cuCtxSynchronize"); // the setup's legacy-stream copies too

        // Engram on the host: the compressed token map, the multipliers, the tables and 64 readers
        if (o.engram_dir) |edir| {
            const map_text = try std.Io.Dir.cwd().readFileAlloc(io, o.token_map orelse return error.NoTokenMap, a, .limited(1 << 26));
            const map = try std.json.parseFromSliceLeaky([]i32, a, map_text, .{});
            const EngramJson = struct { multipliers: [][]i64 };
            const ej_text = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ o.kit_dir, "engram.json" }), a, .limited(1 << 26));
            const ej = try std.json.parseFromSliceLeaky(EngramJson, a, ej_text, .{ .ignore_unknown_fields = true });
            var mult: [engram.max_layers][engram.max_ngram]i64 = @splat(@splat(0));
            for (ej.multipliers, 0..) |row, l| for (row, 0..) |v, k| {
                mult[l][k] = v;
            };
            m.tables = try engram_io.Tables.open(gpa, io, edir);
            m.epool = try engram_io.Pool.init(gpa, io, 64);
            m.eh = try prompt.EngramHost.init(a, c, engram.Hasher.init(c.*, map, mult), &m.tables.?, m.epool.?, sp.rank, sp.world, m.pass_rows);
            if (o.engram_aio) {
                // a round's reads, 2 x 12 x its rows at most: 1,024 slots for each 16 rows of the round-row setting
                m.aio = try engram_aio.Aio.init(gpa, io, 1024 * ((o.round_rows + 15) / 16));
                m.eh.?.aio = m.aio;
                // a prompt chunk's reads (2 x 12 x 2,048 at most): a ring of 4,096 slots (32 MiB), 1,024 rows a batch
                m.paio = try engram_aio.Aio.init(gpa, io, 4096);
                m.eh.?.paio = m.paio;
            }
            if (epf_set.on()) {
                // its own ring of the prompt ring's 4,096 slots; with scales the tables' scale rows in RAM
                const eh = &m.eh.?;
                const t0 = std.Io.Timestamp.now(io, .awake).nanoseconds;
                m.epf = try engram_pf.Prefetch.init(gpa, io, c, eh.hasher, &m.tables.?, eh.lo, eh.hi, m.pass_rows, chunk_rows, epf_set, 4096);
                eh.pf = m.epf;
                const secs = @as(f64, @floatFromInt(std.Io.Timestamp.now(io, .awake).nanoseconds - t0)) / 1e9;
                var nb: [32]u8 = undefined;
                std.log.info("TF_DS_ENGRAM_PREFETCH={s}: prompt Engram rows read up to {d} chunks ahead by a reader of their own ({d} MiB held, {d:.1} s to set up)", .{ epf_set.names(&nb), epf_set.depth, m.epf.?.bytes() >> 20, secs });
                if (es_set.on()) {
                    // TF_DS_ENGRAM_SPLIT: the column partner's link (every rank, in this order: the addresses go down
                    // the Fan's links, which carry nothing else here), then the reader reads its part alone
                    const t1 = std.Io.Timestamp.now(io, .awake).nanoseconds;
                    const pl = try engram_pf.pairLink(io, &m.fan.links, o.master, o.port, o.rank, o.world, es_set, eh.lo, eh.hi - eh.lo);
                    const first = o.rank < 2;
                    try m.epf.?.usePair(pl.fd, es_set, first);
                    const s2 = @as(f64, @floatFromInt(std.Io.Timestamp.now(io, .awake).nanoseconds - t1)) / 1e9;
                    var sb: [16]u8 = undefined;
                    std.log.info("TF_DS_ENGRAM_SPLIT={s}: column partners split each chunk's Engram reads, this rank reads its {s} part ({s}) and takes the rest from rank {d} over TCP port {d} ({d} MiB held in all, {d:.1} s to link)", .{ es_set.names(&sb), if (first) "first" else "second", if (first) "the share" else "1 - the share", o.rank ^ 2, o.port + engram_pf.pair_port_offset + @as(u16, @intCast(o.rank % 2)), m.epf.?.bytes() >> 20, s2 });
                }
            }
        } else if (c.engram_layers.slice().len > 0) return error.NoEngramTables;

        // a vision checkpoint's image routing bias (without it a prompt with images is refused; text runs as ever)
        if (c.vision) {
            const path = o.vision_bias orelse try std.fs.path.join(a, &.{ o.kit_dir, "vision", "gate_bias_vl.safetensors" });
            m.loadVisionBias(path) catch |err| if (o.vision_bias != null or err != error.FileNotFound) return err;
        }

        m.host_pos = try a.alloc(i64, m.pass_rows);
        m.ids64 = try a.alloc(i64, m.pass_rows);
        m.vocab = sp.world * (if (m.two) |t| t.hw[0] + t.hw[1] else m.w.head.n); // 2D: both pairs' vocabulary parts
        m.logits = try a.alloc(f32, round.max_rows * m.vocab);
        if (pprof_mod.wanted()) {
            // TF_DS_PROMPT_PROF=1: timing events for the prompt chunks' phases (nothing else changes)
            const p = try gpa.create(pprof_mod.Prof);
            errdefer gpa.destroy(p);
            p.* = try pprof_mod.Prof.init(gpa, d, o.rank);
            m.pprof = p;
        }
        return m;
    }

    /// The fields with a meaning before anything is set up (null, zero, empty): open builds the Model field by field
    /// on `undefined` memory, so a field's default never applies there and every one of these is set here. A field
    /// added with a default (an optional, a counter) belongs in this list, else it is garbage (0xaa.. in a safe build)
    /// when its switch is off: stage ZE2's `epf` was, and its first prompt's fill jumped through it.
    fn clearState(m: *Model) void {
        m.fill_slot = 0;
        m.fill_seq = &.{};
        m.ahead = null;
        m.prof = false;
        m.timer = null;
        m.ptimer = null;
        m.t_enqueue = 0;
        m.t_forward = 0;
        m.t_absorb = 0;
        m.clk_ph = @splat(0);
        m.clk_cnt = @splat(0);
        m.clk_rounds = 0;
        m.dpool = null;
        m.dr = null;
        m.tables = null;
        m.epool = null;
        m.aio = null;
        m.paio = null;
        m.eh = null;
        m.epf = null;
        m.rdma_mod = null;
        m.ring = null;
        m.graphs = null;
        m.pgraphs = null;
        m.two = null;
        m.rdma_rings = null;
        m.pass_rows = chunk_rows;
        m.ov = null;
        m.sp_rows = false;
        m.sp_mode = .off;
        m.pprof = null;
        m.ovl_side = null;
        m.attn_st = .{};
        m.attn_k = null;
        m.router_st = .{};
        m.router_k = null;
        m.p16_k = null;
        m.x2 = null;
        m.img_buf = null;
    }

    pub fn close(m: *Model) void {
        m.joinAhead();
        if (m.epf) |p| {
            p.deinit();
            m.epf = null;
            if (m.eh) |*eh| eh.pf = null;
        }
        m.stream.synchronize() catch {};
        if (m.img_buf) |*b| b.free();
        m.spans.deinit(m.gpa);
        if (m.graphs) |*g| g.deinit();
        if (m.pgraphs) |*g| g.deinit();
        if (m.timer) |t| {
            t.deinit();
            m.gpa.destroy(t);
        }
        if (m.ptimer) |t| {
            t.deinit();
            m.gpa.destroy(t);
        }
        if (m.pprof) |p| {
            p.summary();
            p.deinit();
            m.gpa.destroy(p);
            m.pprof = null;
        }
        if (m.ovl_side) |*x| {
            x.stream.synchronize() catch {};
            x.deinit();
            m.ovl_side = null;
        }
        if (m.ov) |ov| {
            ov.stream.synchronize() catch {};
            ov.blas.close();
            for (ov.ev) |*x| x.deinit();
            ov.done.deinit();
            ov.comm.deinit();
            ov.stream.deinit();
            m.gpa.destroy(ov);
            m.ov = null;
        }
        m.rd.dropSide(&m.eng);
        m.rd.dropPrefetch();
        m.rd.dropPar();
        if (m.aio) |x| x.deinit();
        if (m.paio) |x| x.deinit();
        if (m.epool) |p| p.deinit(m.gpa);
        if (m.tables) |*t| t.close();
        if (m.rdma_rings) |*r| r.close(m.gpa);
        m.blas.close();
        m.arena.deinit();
        m.rope_buf.free();
        if (m.par) |*x| x.unload();
        if (m.x2) |*x| x.unload();
        if (m.attn_k) |*x| x.unload();
        if (m.router_k) |*x| x.unload();
        if (m.p16_k) |*x| x.unload();
        m.ex.unload();
        m.ops.unload();
        m.exk.unload();
        m.lg.unload();
        m.pf.unload();
        m.set.deinit();
        if (m.ring) |r| {
            r.stop();
            r.destroy(m.gpa);
        }
        if (m.rdma_mod) |*x| x.unload();
        m.comm.deinit();
        m.fan.close();
        m.w.deinit();
        m.cache_file.close(m.io);
        m.ix.deinit();
        m.cache_dir.close(m.io);
        m.stream.deinit();
        m.host.deinit();
        m.gpa.destroy(m);
    }

    /// TP2's RDMA ring for the decode gathers: created on both ranks, the queue-pair infos exchanged over NCCL,
    /// connected, and started once both are (an all-gather as the barrier).
    pub fn openRing(gpa: std.mem.Allocator, d: *const cuda.Driver, k: rdma.Kernels, devices: []const []const u8, rank: u32, comm: *const comm_mod.Comm, stream: cuda.Stream) !*rdma.Ring {
        if (comm.world != 2) return error.NotTwoRanks;
        const r = try rdma.Ring.create(gpa, d, k, devices, rank, 2, .{ .max_bytes = ring_bytes, .gid_index = rdmaGid() });
        errdefer r.destroy(gpa);
        const per = @sizeOf(rdma.Info);
        const mine = [2]rdma.Info{ r.info(0), r.info(1) };
        var dev = try cuda.DeviceBuffer.alloc(d, 3 * 2 * per);
        defer dev.free();
        try dev.upload(0, std.mem.sliceAsBytes(&mine));
        try comm.allGather(dev.ptr, dev.ptr + 2 * per, 2 * per, .u8, stream);
        try stream.synchronize();
        var all: [2][2]rdma.Info = undefined;
        try dev.download(2 * per, std.mem.sliceAsBytes(&all));
        try r.connect(&.{ all[0][rank], all[1][rank] });
        // every queue pair ready on both ranks before either sends
        try comm.allGather(dev.ptr, dev.ptr + 2 * per, 1, .u8, stream);
        try stream.synchronize();
        try r.start();
        return r;
    }

    fn readKit(a: std.mem.Allocator, io: std.Io, kit: []const u8, name: []const u8) ![]const u8 {
        return std.Io.Dir.cwd().readFileAllocOptions(io, try std.fs.path.join(a, &.{ kit, "cubins", name }), a, .limited(1 << 28), .@"16", null);
    }

    /// Every stream's queued work done (cuCtxSynchronize): after a refused primitive, before the ranks agree to step past
    /// it (lanes.zig, refusal.zig). Blocks for good when this rank posted a collective another rank never will.
    pub fn drain(m: *const Model) !void {
        const d = m.ctx.d;
        try d.check(d.api.cuCtxSynchronize(), "cuCtxSynchronize");
    }

    /// The link to the other rank (rank 0's to rank 1, rank 1's to rank 0).
    pub fn peer(m: *const Model) link.Link {
        return m.fan.links[0].?;
    }

    /// Rank 0's links to its followers: rank 1's, or ranks 1-3's on the four-node split.
    pub fn followers(m: *const Model) []const ?link.Link {
        return m.fan.links[0 .. m.world - 1];
    }

    pub fn drafting(m: *const Model) bool {
        return m.dr != null;
    }

    // -- the primitives (both ranks, in the same order) ----------------------------------------------------------

    /// A stream's prompt is about to fill pool slot `slot`, its compressed rows in its extent from position `base`.
    /// Wait for the next chunk's Engram read-ahead, if one runs (before the prompt it reads changes or goes).
    /// TF_DS_ENGRAM_SPLIT is on: the prefetch reader shares its reads with a column partner.
    pub fn splitOn(m: *const Model) bool {
        const p = m.epf orelse return false;
        return p.pair != null;
    }

    pub fn joinAhead(m: *Model) void {
        if (m.ahead) |th| th.join();
        m.ahead = null;
        // TF_DS_ENGRAM_PREFETCH: the reader's plan dropped (a chunk it is on cancelled and waited for)
        if (m.epf) |p| p.stop();
    }

    /// A kept prompt state's own bytes (the prompt cache's snapshots; the compressed plane stays in the stream's
    /// extent): one slot's window ring of every layer and positional stores of every ratio-2 layer.
    pub fn snapBytes(m: *const Model) usize {
        const pb = prompt.raw_rows * m.eng.c.head_dim * 4;
        var n: usize = m.rings.len * m.eng.ringBytes();
        for (m.caches.raw_kv[0..m.eng.c.layers], m.caches.raw_score[0..m.eng.c.layers]) |k, sc| {
            if (k != 0) n += pb;
            if (sc != 0) n += pb;
        }
        return n;
    }

    /// Slot `slot`'s own state into `buf` (save: true) or back from it, in the stream's order (no wait): with the
    /// extent's compressed rows, the state a prompt pass stands in at a chunk end.
    pub fn snapCopy(m: *Model, slot: usize, buf: u64, save: bool) !void {
        const rb = m.eng.ringBytes();
        const pb = prompt.raw_rows * m.eng.c.head_dim * 4;
        var at = buf;
        for (m.rings) |r| {
            try m.copyState(r + slot * rb, at, rb, save);
            at += rb;
        }
        for (m.caches.raw_kv[0..m.eng.c.layers], m.caches.raw_score[0..m.eng.c.layers]) |k, sc| {
            for ([_]u64{ k, sc }) |live| if (live != 0) {
                try m.copyState(live + slot * pb, at, pb, save);
                at += pb;
            };
        }
    }

    /// The compressed rows of positions [0, at) of the extent at `from` copied into the extent at `to` (a kept prompt
    /// resumed in another extent while a live stream holds its own), in the stream's order: every kv-source layer's
    /// codes and scales and its index keys, the rows view() places at those positions.
    pub fn copyExtent(m: *Model, from: usize, to: usize, at: usize) !void {
        const c = m.eng.c;
        const cs = &m.caches;
        for (m.eng.w.layers, 0..) |lay, i| {
            if (lay.comp_wkv == null) continue;
            const r: usize = lay.ratio;
            if (from % r != 0 or to % r != 0 or at % r != 0) return error.BadView;
            const n = at / r;
            try m.copySpan(cs.comp_codes[i], from / r, to / r, n, c.head_dim / 2);
            try m.copySpan(cs.comp_scales[i], from / r, to / r, n, c.head_dim / 16);
            if (lay.idx_wk != null) {
                try m.copySpan(cs.idx_codes[i], from / r, to / r, n, c.index_head_dim / 2);
                try m.copySpan(cs.idx_scales[i], from / r, to / r, n, c.index_head_dim / 32);
            }
        }
    }

    fn copySpan(m: *Model, plane: u64, from: usize, to: usize, n: usize, row: usize) !void {
        const d = m.ctx.d;
        try d.check(d.api.cuMemcpyDtoDAsync_v2(plane + to * row, plane + from * row, n * row, m.stream.handle), "cuMemcpyDtoDAsync");
    }

    fn copyState(m: *Model, live: u64, kept: u64, bytes: usize, save: bool) !void {
        const d = m.ctx.d;
        const dst = if (save) kept else live;
        const src = if (save) live else kept;
        try d.check(d.api.cuMemcpyDtoDAsync_v2(dst, src, bytes, m.stream.handle), "cuMemcpyDtoDAsync");
    }

    /// Each layer's gate_bias_vl (f32 [experts], the checkpoint's extra file) on the device: Gate.forward's bias at
    /// image-span positions.
    fn loadVisionBias(m: *Model, path: []const u8) !void {
        const core = @import("core");
        var f = try core.safetensors.File.open(m.gpa, m.io, path);
        defer f.close(m.io);
        var nb: [64]u8 = undefined;
        for (m.w.layers, 0..) |*lay, i| {
            const t = f.get(try std.fmt.bufPrint(&nb, "layers.{d}.ffn.gate.bias_vl", .{i})) orelse return error.MissingVisionBias;
            if (!t.is(.f32, &.{m.cfg.experts})) return error.BadVisionBias;
            const dst = try m.arena.take(t.bytes.len);
            const tmp = try m.gpa.dupe(u8, t.bytes); // (a copy out of the file mapping first: see vit.zig)
            defer m.gpa.free(tmp);
            try m.ctx.d.check(m.ctx.d.api.cuMemcpyHtoD_v2(dst, tmp.ptr, tmp.len), "cuMemcpyHtoD");
            lay.gate_b_vl = dst;
        }
    }

    /// The filling prompt's image spans (absolute positions, in order) and their rows bf16 [tokens, hidden] from the
    /// host, after fillBegin and before its chunks.
    pub fn setImages(m: *Model, spans: []const prompt.Span, rows: []const u8) !void {
        if (rows.len > 0) {
            if (m.img_buf == null or m.img_buf.?.len < rows.len) {
                if (m.img_buf) |*b| b.free();
                m.img_buf = null;
                m.img_buf = try cuda.DeviceBuffer.alloc(m.ctx.d, rows.len);
            }
            const d = m.ctx.d;
            try d.check(d.api.cuMemcpyHtoDAsync_v2(m.img_buf.?.ptr, rows.ptr, rows.len, m.stream.handle), "cuMemcpyHtoDAsync");
            try m.stream.synchronize(); // (the caller's rows may go once this returns)
        }
        m.spans.clearRetainingCapacity();
        try m.spans.appendSlice(m.gpa, spans);
        m.ch.spans = m.spans.items;
        m.ch.img_rows = if (m.img_buf) |b| b.ptr else 0;
    }

    pub fn fillBegin(m: *Model, slot: usize, base: usize) !void {
        if (slot >= m.streams) return error.BadSlot;
        m.spans.clearRetainingCapacity();
        m.ch.spans = &.{};
        m.view = try m.caches.view(&m.eng, slot, base);
        for (m.rings, m.ring_view) |r, *v| v.* = r + slot * m.eng.ringBytes();
        m.fill_slot = slot;
    }

    /// The prompt's chunk of n rows from position `start` (seq: the stream's ids through the chunk's end) through every
    /// layer with the served bounded replay (replay = the prompt's length - the window: an encoder-only chunk stops at
    /// the decoder's first layer). A chunk that reaches the head leaves the logits of its last row in ch.head_g and,
    /// drafting, its DSpark taps absorbed into the slot's drafter rings. True when it reached the head.
    pub fn fillChunk(m: *Model, seq: []const i32, start: usize, n: usize, replay: usize) !bool {
        if (n == 0 or n > m.pass_rows or seq.len != start + n) return error.BadChunk;
        // a pass of two chunks: whole chunks from a chunk end, encoder-only (passRows)
        if (n > chunk_rows and (n != 2 * chunk_rows or start % chunk_rows != 0 or start + n > replay)) return error.BadChunk;
        const e = &m.eng;
        const c = e.c;
        const ch = &m.ch;
        // an image position's id is negative in the sequence (its n-grams dead for Engram): its embedding's id is the
        // image token's, which its span row then replaces
        const image_id: i64 = if (c.image_token) |t| t else 0;
        for (0..n) |j| m.ids64[j] = if (seq[start + j] < 0) image_id else seq[start + j];
        // an encoder-only pass of two chunks, overlapped (the same choice on every rank: n and the spans)
        if (m.ov != null and n == 2 * chunk_rows and ch.spans.len == 0) return m.fillOverlap(seq, start, n, replay);
        var shared: prompt.Shared = .{};
        const pt = m.ptimer;
        if (pt) |t| try t.mark(m.stream, .start);
        // TF_DS_PROMPT_PROF=1: this chunk's phase marks (the engine carries them for the chunk alone)
        const pp = m.pprof;
        if (pp) |p| {
            try p.begin(m.stream, start, n, replay);
            m.eng.prof = p;
        }
        defer m.eng.prof = null;
        try prompt.begin(e, ch, m.ids64[0..n], start, m.host_pos);
        ch.sub_rows = if (n > chunk_rows) chunk_rows else 0;
        ch.sp = null;
        if (m.two) |t| ch.sp = prompt.spFor(m.sp_mode, t.r + 2 * t.p, n, start, replay, ch.spans.len);
        if (pp) |p| p.sp = if (ch.sp) |x| (if (x.cut) "rows2-cut" else if (x.n == 4 * x.q) "rows" else "rows2") else "off";
        // TF_DS_PROMPT_OVL: the comm stream and one launch order for this SP chunk alone (the same choice on every
        // rank: ch.sp depends on n, start, replay and the spans); every other chunk and every round as served
        const sd: ?*ovl.Side = if (m.ovl_side != null and ch.sp != null) &m.ovl_side.? else null;
        if (sd) |x| {
            x.reset();
            m.comm.order = &x.order;
        }
        m.eng.side = sd;
        defer {
            m.eng.side = null;
            m.comm.order = null;
        }
        if (ch.spans.len > 0) try prompt.embedSpans(e, ch);
        if (pt) |t| try t.mark(m.stream, .embed);
        try prompt.pm(e, .other);
        const taps = m.dr != null;
        var floor: usize = 0;
        var kv_done: ?usize = null;
        // Engram rows read ahead on a thread while the GPU runs the layers before them (the served reader pool's
        // overlap): this chunk's both layers' during the chunk before it (m.ahead), else the first layer's from the
        // chunk's start and the next's once the first is applied
        var ahead: ?std.Thread = m.ahead;
        m.ahead = null;
        defer if (ahead) |th| th.join();
        if (m.epf) |pf| {
            // TF_DS_ENGRAM_PREFETCH: the reader's plan has this chunk (or restarts at it), the chunks before it are
            // given back, and it reads on up to its depth past this chunk; no `ahead` thread
            const full = if (m.fill_seq.ptr == seq.ptr and m.fill_seq.len >= seq.len) m.fill_seq else seq;
            pf.begin(full, start, n, replay);
        } else if (m.eh) |*eh| {
            if (ahead != null and (eh.ahead_start != start or eh.ahead_n != n)) {
                ahead.?.join();
                ahead = null;
            }
            if (ahead == null) {
                eh.ahead_start = start;
                eh.ahead_n = n;
                eh.ahead_ready = .{ false, false };
                eh.ahead_err = null;
                ahead = std.Thread.spawn(.{}, prompt.engramReadAhead, .{ eh, c, seq, start, n, @as(usize, 0) }) catch null;
            }
        }
        for (0..c.layers) |li| {
            if (li == c.layers / 2) {
                if (!try prompt.replayCut(e, ch, &m.view, &shared, li, replay, m.host_pos)) {
                    // encoder-only: no synchronize either (as at the chunk's end below), so the next chunk's host steps
                    // run while this one's last layers do; the profiles alone wait
                    if (pt) |t| {
                        try m.stream.synchronize();
                        try t.finish();
                    }
                    if (pp) |p| try p.finish();
                    return false;
                }
                floor = replay;
                kv_done = li;
            }
            if (e.w.layers[li].engram_wkv != null) {
                const eh = if (m.eh) |*x| x else return error.NoEngramTables;
                if (ch.engram_staged != null and ch.engram_staged.? == li) {
                    // TF_DS_PROMPT_OVL=engram: projected and summed a layer early (engramStage1 / 2)
                    try prompt.engramFinish(e, ch, li);
                } else {
                    if (ahead) |th| {
                        th.join();
                        ahead = null;
                    }
                    try prompt.pm(e, .host); // (the GPU's wait for the read-ahead, if it was behind)
                    if (eh.ahead_err) |err| return err;
                    try prompt.engramApply(e, ch, eh, li, seq);
                }
                const j = std.mem.indexOfScalar(u16, c.engram_layers.slice(), @intCast(li)).?;
                const ne = @min(2, c.engram_layers.slice().len);
                if (m.epf != null) {
                    // TF_DS_ENGRAM_PREFETCH: the reader has the next layer's and the next chunks' rows in hand
                } else if (j + 1 < ne) {
                    if (!eh.ahead_ready[j + 1])
                        ahead = std.Thread.spawn(.{}, prompt.engramReadAhead, .{ eh, c, seq, start, n, j + 1 }) catch null;
                } else if (m.fill_seq.len > start + n and m.fill_seq.ptr == seq.ptr) {
                    // the next chunk's rows while this one's layers run
                    const nn = passRows(m.pass_rows, start + n, m.fill_seq.len, replay);
                    eh.ahead_start = start + n;
                    eh.ahead_n = nn;
                    eh.ahead_ready = .{ false, false };
                    eh.ahead_err = null;
                    m.ahead = std.Thread.spawn(.{}, prompt.engramReadAheadAll, .{ eh, c, m.fill_seq[0 .. start + n + nn], start + n, nn }) catch null;
                }
                if (pt) |t| try t.mark(m.stream, .engram);
            }
            // a DSpark tap reads the streams after the layer's Engram (_forward_k's order)
            if (taps) if (std.mem.indexOfScalar(u16, c.dspark_taps.slice(), @intCast(li))) |j| try prompt.tap(e, ch, j);
            try prompt.attnMixesX(e, ch, li, true);
            // TF_DS_PROMPT_OVL=engram: the next layer's Engram projection, and its sum's step 1 on the comm stream
            // under this layer's attention (its rows joined from the read-ahead here, a layer early)
            const staged: ?usize = if (sd != null and sd.?.lv.engram and ch.sp != null) ovl.engramAhead(c.engram_layers.slice(), li, c.layers / 2) else null;
            if (staged) |nx| {
                const eh = if (m.eh) |*x| x else return error.NoEngramTables;
                if (ahead) |th| {
                    th.join();
                    ahead = null;
                }
                if (eh.ahead_err) |err| return err;
                try prompt.engramStage1(e, ch, eh, nx, seq);
            }
            if (pt) |t| try t.mark(m.stream, .mix_attn);
            try prompt.attention(e, ch, &m.view, &shared, li, m.ring_view[li], floor, kv_done != null and kv_done.? == li);
            if (pt) |t| try t.mark(m.stream, .attn);
            try prompt.gather(e, ch, ch.pa, ch.ga);
            if (pt) |t| try t.mark(m.stream, .gather_a);
            try prompt.ffnMixes(e, ch, li);
            // its step 2 on the comm stream under this layer's experts
            if (staged) |nx| try prompt.engramStage2(e, ch, nx);
            if (pt) |t| try t.mark(m.stream, .mix_ffn);
            try prompt.moe(e, ch, li);
            if (pt) |t| try t.mark(m.stream, .moe);
            try prompt.gather(e, ch, ch.pm, ch.gm);
            if (pt) |t| try t.mark(m.stream, .gather_m);
            prompt.endLayer(ch);
        }
        try prompt.head(e, ch);
        if (pt) |t| try t.mark(m.stream, .head);
        if (m.dr) |*dr| {
            // its taps (the tap layers side by side) into the slot's drafter rings
            const dw = c.hidden;
            for (0..c.dspark_taps.slice().len) |j| try e.ops.copyRows(e.s, prompt.tapRows(e, ch, j), dw * 2, dr.at + j * dw * 2, dr.taps_w * 2, dw * 2, ch.n);
            m.eng.prof = null; // (the absorb's launches as one stretch: `other`)
            try dr.absorb(e, ch, &m.dpool.?, m.fill_slot, dr.at, ch.n, ch.start);
            if (pp) |p| try p.mark(m.stream, .other);
        }
        // no synchronize: the next chunk's host steps (its first Engram layer's hashing, table reads and decode, ~0.14 s
        // at 2,048 rows) run while this chunk's kernels do; readers of the results (the prompt's logits) synchronize.
        // The profile (--profile 2) times a chunk to its end, so it alone waits.
        if (pt) |t| {
            try m.stream.synchronize();
            try t.finish();
        }
        if (pp) |p| {
            try p.finish();
            p.summary(); // (a prompt's last chunk: the rank's sums so far, in case the process never reaches close)
        }
        return true;
    }

    /// An encoder-only pass of two chunks (fillChunk with Overlap): chunk A [start, start + 2,048) on this stream and
    /// buffers, chunk B after it on the Overlap's, one layer apart: step k issues A's layer k (and records its event),
    /// then B's layer k - 1 behind A's event for that layer, so B's layer reads A's window keys, compressed rows and
    /// positional stores as the next chunk would; each chunk is the 2,048-row chunk's launches. The main stream waits
    /// for B's end. The Engram rows of both come from the pass's read-ahead (read here if it is not ready); the next
    /// pass's starts after B's last Engram layer. False: an encoder-only pass reaches no head.
    fn fillOverlap(m: *Model, seq: []const i32, start: usize, n: usize, replay: usize) !bool {
        const ov = m.ov.?;
        const half = chunk_rows;
        const c = m.eng.c;
        const cut = c.layers / 2;
        const ne = @min(2, c.engram_layers.slice().len);
        if (m.eh) |*eh| {
            m.joinAhead();
            var ready = eh.ahead_start == start and eh.ahead_n == n and eh.ahead_err == null;
            for (0..ne) |j| ready = ready and eh.ahead_ready[j];
            if (!ready) {
                eh.ahead_start = start;
                eh.ahead_n = n;
                eh.ahead_ready = .{ false, false };
                eh.ahead_err = null;
                prompt.engramReadAheadAll(eh, c, seq, start, n);
            }
            if (eh.ahead_err) |err| return err;
        }
        const ea = &m.eng;
        const eb = &ov.eng;
        const cha = &m.ch;
        const chb = &ov.ch;
        try prompt.begin(ea, cha, m.ids64[0..half], start, m.host_pos[0..half]);
        try prompt.begin(eb, chb, m.ids64[half..n], start + half, m.host_pos[half..n]);
        for ([_]*prompt.Chunk{ cha, chb }, [_]usize{ start, start + half }) |ch, s0| {
            ch.sub_rows = 0;
            ch.sp = null;
            if (m.two) |t| ch.sp = prompt.spFor(m.sp_mode, t.r + 2 * t.p, half, s0, replay, 0);
        }
        var sha: prompt.Shared = .{};
        var shb: prompt.Shared = .{};
        const last_engram: ?usize = if (ne > 0) c.engram_layers.slice()[ne - 1] else null;
        for (0..cut + 2) |step| {
            if (step <= cut) {
                try m.encoderLayer(ea, cha, &sha, seq, step, replay);
                try ov.ev[step].record(m.stream);
            }
            if (step >= 1) {
                const li = step - 1;
                try ov.stream.wait(ov.ev[li]);
                try m.encoderLayer(eb, chb, &shb, seq, li, replay);
                if (m.eh) |*eh| if (last_engram != null and li == last_engram.? and m.fill_seq.len > start + n and m.fill_seq.ptr == seq.ptr) {
                    // the next pass's rows while this one's last layers run
                    const nn = passRows(m.pass_rows, start + n, m.fill_seq.len, replay);
                    eh.ahead_start = start + n;
                    eh.ahead_n = nn;
                    eh.ahead_ready = .{ false, false };
                    eh.ahead_err = null;
                    m.ahead = std.Thread.spawn(.{}, prompt.engramReadAheadAll, .{ eh, c, m.fill_seq[0 .. start + n + nn], start + n, nn }) catch null;
                };
            }
        }
        try ov.done.record(ov.stream);
        try m.stream.wait(ov.done);
        return false;
    }

    /// Layer li of an encoder-only chunk on engine `e` (fillChunk's layer without the profile): at the decoder's first
    /// layer its mixes and kv-source update, where the chunk ends.
    fn encoderLayer(m: *Model, e: *const prompt.Engine, ch: *prompt.Chunk, sh: *prompt.Shared, seq: []const i32, li: usize, replay: usize) !void {
        const c = e.c;
        if (li == c.layers / 2) {
            if (try prompt.replayCut(e, ch, &m.view, sh, li, replay, m.host_pos)) return error.OverlapPastReplay;
            return;
        }
        if (e.w.layers[li].engram_wkv != null) {
            const eh = if (m.eh) |*x| x else return error.NoEngramTables;
            if (eh.ahead_err) |err| return err;
            try prompt.engramApply(e, ch, eh, li, seq);
        }
        if (m.dr != null) if (std.mem.indexOfScalar(u16, c.dspark_taps.slice(), @intCast(li))) |j| try prompt.tap(e, ch, j);
        try prompt.attnMixes(e, ch, li);
        try prompt.attention(e, ch, &m.view, sh, li, m.ring_view[li], 0, false);
        try prompt.gather(e, ch, ch.pa, ch.ga);
        try prompt.ffnMixes(e, ch, li);
        try prompt.moe(e, ch, li);
        try prompt.gather(e, ch, ch.pm, ch.gm);
        prompt.endLayer(ch);
    }

    /// The prompt's logits (its last row's, the ranks' columns in rank order) after a fill's last chunk.
    pub fn promptLogits(m: *const Model) u64 {
        return m.ch.head_g;
    }

    /// One round over every stream's window (rows.windows, each with its stream's slot and extent), then each window
    /// whose stream drafts has its rows' taps absorbed into its slot's drafter rings at its positions (the served
    /// eager absorb: rows past the ones the round keeps sit where the drafter never reads, and the next window
    /// overwrites them). absorb[i]: window i's stream drafts.
    pub fn verify(m: *Model, rows: round.Rows, absorb: []const bool) !void {
        const e = &m.eng;
        if (m.prof) try m.stream.synchronize(); // (the GPU idle, so the issue time shows whether the host keeps it fed)
        const t0 = m.now();
        try round.forward(e, &m.rd, &m.ch, &m.caches, m.rings, if (m.eh) |*x| x else null, &.{}, rows, m.pool_cap, null, if (m.graphs) |*g| g else null);
        if (m.prof) {
            m.t_enqueue = m.now() - t0;
            try m.stream.synchronize();
            m.t_forward = m.now() - t0;
            if (m.timer) |t| try t.finish();
            if (m.rd.clk != 0) {
                var h: [1 + 2 * round.Round.max_clocks]u64 = undefined;
                const d = m.ctx.d;
                try d.check(d.api.cuMemcpyDtoH_v2(&h, m.rd.clk, h.len * 8), "cuMemcpyDtoH");
                const n = @min(h[0], round.Round.max_clocks);
                for (1..n) |i| {
                    const tag = h[2 + 2 * i] & 0xff;
                    m.clk_ph[tag] += h[1 + 2 * i] -% h[1 + 2 * (i - 1)];
                    m.clk_cnt[tag] += 1;
                }
                m.clk_rounds += 1;
            }
        }
        const t1 = m.now();
        if (m.dr) |*dr| {
            const wins = rows.windows orelse return error.BadRound;
            var items: [max_streams]draft.Item = undefined;
            var n: usize = 0;
            for (wins, absorb) |win, yes| {
                if (!yes) continue;
                items[n] = .{ .slot = @intCast(rows.slots.?[win.row]), .taps = m.rd.taps + win.row * dr.taps_w * 2, .n = win.n, .start = @intCast(rows.pos[win.row]) };
                n += 1;
            }
            if (n > 0) try dr.absorbMany(e, &m.ch, &m.dpool.?, items[0..n]);
        }
        try m.stream.synchronize();
        if (m.prof) m.t_absorb = m.now() - t1;
    }

    /// The rounds' GPU clocks (a profile): the forward's start and each stretch's ends, for its gaps.
    pub fn useClock(m: *Model, marks: u8) !void {
        m.rd.clk_marks = marks == 1 or marks == 3;
        // TF_DS_CLOCK_RATIO: the compress ratio whose layers mode 3 times (2 unless set)
        const r: u8 = if (std.c.getenv("TF_DS_CLOCK_RATIO")) |v| std.fmt.parseInt(u8, std.mem.span(v), 10) catch 2 else 2;
        m.rd.clk_ratio = if (marks == 3) r else null;
        m.rd.clk_layers = marks == 2 or marks == 3;
        if (m.rd.clk == 0) m.rd.clk = try m.arena.take((1 + 2 * round.Round.max_clocks) * 8);
    }

    /// The rounds' GPU time by phase (eager rounds; --profile 2): a timer on the round.
    pub fn usePhaseTimer(m: *Model) !void {
        if (m.timer != null) return;
        const t = try m.gpa.create(round.PhaseTimer);
        errdefer m.gpa.destroy(t);
        t.* = try round.PhaseTimer.init(m.gpa, m.ctx.d);
        m.timer = t;
        const pt = try m.gpa.create(round.PhaseTimer);
        errdefer m.gpa.destroy(pt);
        pt.* = try round.PhaseTimer.init(m.gpa, m.ctx.d);
        m.ptimer = pt;
        m.rd.timer = t;
    }

    /// The monotonic clock (ns).
    pub fn now(m: *const Model) u64 {
        return @intCast(std.Io.Timestamp.now(m.io, .awake).nanoseconds);
    }

    /// The round's logits, R rows of the vocabulary, on the host (after verify).
    /// The round's greedy tokens, R rows (the device's argmax of the round's logits, as sampling.argmax), on the host.
    pub fn roundArgmax(m: *Model, R: usize) ![]const u32 {
        const d = m.ctx.d;
        try m.ops.argmaxRows(m.stream, m.rd.logits, m.vocab, m.vocab, R, m.amax);
        try m.stream.synchronize();
        try d.check(d.api.cuMemcpyDtoH_v2(&m.amax_host, m.amax, R * 4), "cuMemcpyDtoH");
        return m.amax_host[0..R];
    }

    pub fn roundLogits(m: *Model, R: usize) ![]const f32 {
        const d = m.ctx.d;
        try m.stream.synchronize();
        try d.check(d.api.cuMemcpyDtoH_v2(m.logits.ptr, m.rd.logits, R * m.vocab * 4), "cuMemcpyDtoH");
        return m.logits[0 .. R * m.vocab];
    }

    /// The prompt's logits on the host (after a fill's last chunk).
    pub fn promptLogitsHost(m: *Model) ![]const f32 {
        const d = m.ctx.d;
        try m.stream.synchronize();
        try d.check(d.api.cuMemcpyDtoH_v2(m.logits.ptr, m.ch.head_g, m.vocab * 4), "cuMemcpyDtoH");
        return m.logits[0..m.vocab];
    }

    /// The drafter's batched pass for N streams: each one's pending token at position q0 in its slot (its rings hold
    /// its positions below q0), `steps` drafts each (the drafter's drafts and confidences on the host).
    pub fn pass(m: *Model, tokens: []const i64, q0: []const i64, slots: []const i64, steps: usize) !void {
        if (m.dr == null) return error.NoDrafter;
        const dr = &m.dr.?;
        const pool = &m.dpool.?;
        for (q0, slots) |q, s| pool.absorbed[@intCast(s)] = @intCast(q);
        try dr.pass(&m.eng, &m.ch, pool, tokens, q0, slots, steps, null);
    }
};

test "prompt passes: two chunks where both end by the replay, else one chunk; 2,048 as before" {
    const c = chunk_rows;
    // the served pass: one chunk, the last shorter
    try std.testing.expectEqual(@as(usize, c), passRows(c, 0, 10 * c, 9 * c));
    try std.testing.expectEqual(@as(usize, 100), passRows(c, 9 * c, 9 * c + 100, 9 * c - 28));
    // 4,096: two chunks while both end at or before the replay
    try std.testing.expectEqual(@as(usize, 2 * c), passRows(2 * c, 0, 8192, 8064));
    try std.testing.expectEqual(@as(usize, c), passRows(2 * c, 4096, 8192, 8064)); // 8,192 is past the replay
    try std.testing.expectEqual(@as(usize, 2 * c), passRows(2 * c, 4 * c, 6 * c + 128, 6 * c)); // ends exactly at it
    try std.testing.expectEqual(@as(usize, c), passRows(2 * c, c, 2 * c + 50, 2 * c - 78)); // the decoder's chunk alone
    // every pass starts at a chunk end, and the passes cover the prompt as the 2,048-row chunks do, merged in pairs
    for ([_]usize{ 1, 2047, 2048, 2049, 4096, 4224, 6143, 8192, 32768, 146421, 156941 }) |len| {
        const replay = len -| 128;
        var start: usize = 0;
        while (start < len) {
            const n = passRows(2 * c, start, len, replay);
            try std.testing.expect(start % c == 0);
            try std.testing.expect(n == 2 * c or n == passRows(c, start, len, replay));
            if (n > c) try std.testing.expect(start + n <= replay);
            start += n;
        }
        try std.testing.expectEqual(len, start);
    }
}

test "open's state on undefined memory: every optional null with the switches off (TF_DS_ENGRAM_PREFETCH unset)" {
    // as open builds it: the Model on fresh memory, `undefined` (0xaa..), then clearState
    const a = std.testing.allocator;
    const m = try a.create(Model);
    defer a.destroy(m);
    @memset(std.mem.asBytes(m), 0xaa);
    m.clearState();
    try std.testing.expect(m.epf == null);
    try std.testing.expect(m.eh == null);
    try std.testing.expect(m.aio == null);
    try std.testing.expect(m.paio == null);
    try std.testing.expect(m.ahead == null);
    try std.testing.expect(m.ov == null);
    // TF_DS_PROMPT_ROUTER (stage ZC4 merge): its kernels null and its state the unset levers (no own-row routing)
    try std.testing.expect(m.router_k == null);
    try std.testing.expect(m.router_st.k == null);
    try std.testing.expect(!m.router_st.lv.any());
    try std.testing.expectEqual(@as(@TypeOf(m.router_st.parts.n), 0), m.router_st.parts.n);
    try std.testing.expectEqual(@as(usize, 0), m.fill_seq.len);
    try std.testing.expectEqual(@as(u64, 0), m.clk_rounds);
    // every optional field with a null default is null after clearState (a new one left out of it fails here)
    const info = @typeInfo(Model).@"struct";
    inline for (info.field_names, info.field_types, info.field_attrs) |name, T, at| {
        if (@typeInfo(T) == .optional and at.default_value_ptr != null) {
            if (@field(m, name) != null) {
                std.debug.print("Model.{s} is not set by clearState\n", .{name});
                return error.TestUnexpectedResult;
            }
        }
    }
}
