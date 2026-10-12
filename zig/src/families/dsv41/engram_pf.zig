//! TF_DS_ENGRAM_PREFETCH (stage ZE2): a prompt's Engram rows read by a reader thread of their own, chunks ahead of
//! the chunk the GPU is on. The served read-ahead (model.fillChunk's `ahead` thread) starts the next chunk's rows only
//! once the chunk's last Engram layer (14) is applied, which leaves a slow drive the chunk's last encoder layers (on a
//! GX10's 1 TB NVMe not enough: its compute stream idles in `host` before layer 1). Here the prompt's chunks are a
//! plan (the chunk sizes model.passRows gives, from the first chunk the fill asks for) and the reader walks it into
//! `depth + 1` slots: chunk k + depth is read while the GPU runs chunk k, and a chunk's slot is given back when the
//! fill begins a chunk past it (its rows' uploads are copies from pageable memory, which return once the bytes are
//! staged: the guarantee the served read-ahead relies on when it overwrites its buffers).
//!
//! Only WHEN the rows are read changes. A chunk's rows are those engramApply reads itself (or the served read-ahead,
//! engramAhead1): the same Hasher.hashes of the same ids, this rank's columns [lo, hi) of layer j's, the same bytes of
//! the same table (AIO on the O_DIRECT descriptors, as the served prompt ring; engram_aio's test holds them equal to
//! the preads), decoded by the same engram.decodeRows, then uploaded by engramApply / engramStage1 to the same buffer
//! for the same kernels. With `scales` the scale rows come from a copy of the tables' scale columns in RAM (read once
//! at start, the same file bytes), and the drive serves only the weight rows: one read a row instead of two.
//! Unset, the engine never builds a Prefetch: the served read-ahead runs exactly as before.
//!
//! TF_DS_ENGRAM_SPLIT=<share> (stage ZE4, needs the prefetch and the four-node split): column partners (ranks g and
//! g ^ 2: castor + clytemnestra, pollux + helen) hash the same 12 columns and so read IDENTICAL rows. With the switch
//! each reads only its part of every chunk's rows (the pair-0 rank, a DGX Spark, rows [0, n * share); its partner, a
//! GX10, the rest), decodes it as before, and sends the decoded rows to the partner over a TCP link of their own on
//! the fabric (Pair): each slot ends up with the same bytes. A part comes with the row ids it was read for, and the
//! receiver takes it only when those ids are its own (u64 for u64): the bytes are those ids' rows of the same table,
//! decoded by the same decodeRows, so a slot's rows equal the unsplit reader's by construction (and a part left over
//! from a cancelled chunk is never taken for another prompt's rows). Unset, no Pair: the reader runs as ZE2's.
const std = @import("std");
const Config = @import("config.zig").Config;
const engram = @import("engram.zig");
const engram_io = @import("engram_io.zig");
const engram_aio = @import("engram_aio.zig");
const link = @import("link.zig");
const dio = @import("core").direct_io;

/// The most chunks read ahead ("prompt": a 4,096-row pass a chunk, 128 Ki rows ahead).
pub const max_depth = 32;

pub const Setting = struct {
    depth: usize = 0, // chunks read ahead of the chunk the GPU is on (0: off, the served read-ahead)
    prompt: bool = false, // spelled "prompt" (depth = max_depth)
    scales: bool = false, // the scale rows from RAM (the drive reads the weight rows alone)

    pub fn on(s: Setting) bool {
        return s.depth > 0;
    }

    /// The value as the start line prints it ("1", "2,scales", "prompt").
    pub fn names(s: Setting, buf: []u8) []const u8 {
        if (!s.on()) return "off";
        const sc: []const u8 = if (s.scales) ",scales" else "";
        if (s.prompt) return std.fmt.bufPrint(buf, "prompt{s}", .{sc}) catch "?";
        return std.fmt.bufPrint(buf, "{d}{s}", .{ s.depth, sc }) catch "?";
    }
};

/// "<depth>[,scales]": depth 1 .. max_depth or "prompt"; "" / "off" / "0": off. Anything else refuses to start.
pub fn parse(v: []const u8) !Setting {
    var st: Setting = .{};
    if (v.len == 0 or std.mem.eql(u8, v, "off") or std.mem.eql(u8, v, "0")) return st;
    var it = std.mem.splitScalar(u8, v, ',');
    var first = true;
    while (it.next()) |w| {
        if (first) {
            first = false;
            if (std.mem.eql(u8, w, "prompt")) {
                st.depth = max_depth;
                st.prompt = true;
                continue;
            }
            if (w.len == 0 or w.len > 3) return error.BadEngramPrefetch;
            for (w) |ch| if (ch < '0' or ch > '9') return error.BadEngramPrefetch;
            const d = std.fmt.parseInt(usize, w, 10) catch return error.BadEngramPrefetch;
            if (d < 1 or d > max_depth) return error.BadEngramPrefetch;
            st.depth = d;
        } else if (std.mem.eql(u8, w, "scales") and !st.scales) {
            st.scales = true;
        } else return error.BadEngramPrefetch;
    }
    return st;
}

/// TF_DS_ENGRAM_PREFETCH from the environment (unset: off).
pub fn fromEnv() !Setting {
    const v = std.c.getenv("TF_DS_ENGRAM_PREFETCH") orelse return .{};
    return parse(std.mem.span(v));
}

/// TF_DS_ENGRAM_SPLIT: the share of each chunk's rows the pair-0 rank of a column pair reads (per mille; 0: off).
pub const Split = struct {
    milli: u32 = 0,

    pub fn on(s: Split) bool {
        return s.milli > 0;
    }

    /// The value as the start line prints it ("1", "0.5", "0.75", "0.667").
    pub fn names(s: Split, buf: []u8) []const u8 {
        if (!s.on()) return "off";
        if (s.milli >= 1000) return "1";
        var b: [3]u8 = undefined;
        b[0] = @intCast('0' + s.milli / 100);
        b[1] = @intCast('0' + s.milli / 10 % 10);
        b[2] = @intCast('0' + s.milli % 10);
        var e: usize = 3;
        while (e > 1 and b[e - 1] == '0') e -= 1;
        return std.fmt.bufPrint(buf, "0.{s}", .{b[0..e]}) catch "?";
    }

    /// The rows [lo, hi) of an n-row chunk a rank reads itself: the pair-0 rank (first) [0, a), its partner [a, n),
    /// a = n * share rounded down (the same on both).
    pub fn own(s: Split, n: usize, first: bool) [2]usize {
        const a = n * @min(s.milli, 1000) / 1000;
        return if (first) .{ 0, a } else .{ a, n };
    }
};

/// "<share>": 1 (1.0, 1.00, 1.000) or 0.d / 0.dd / 0.ddd above 0: the pair-0 rank's share; "" / "off": off. Anything
/// else (0, a share above 1, more than three decimals, a sign, spaces) refuses to start.
pub fn parseSplit(v: []const u8) !Split {
    if (v.len == 0 or std.mem.eql(u8, v, "off")) return .{};
    if (v.len > 5) return error.BadEngramSplit;
    if (v[0] == '1') {
        if (v.len == 1) return .{ .milli = 1000 };
        if (v[1] != '.' or v.len < 3) return error.BadEngramSplit;
        for (v[2..]) |ch| if (ch != '0') return error.BadEngramSplit;
        return .{ .milli = 1000 };
    }
    if (v.len < 3 or v[0] != '0' or v[1] != '.') return error.BadEngramSplit;
    var mm: u32 = 0;
    var scale: u32 = 100;
    for (v[2..]) |ch| {
        if (ch < '0' or ch > '9') return error.BadEngramSplit;
        mm += (ch - '0') * scale;
        scale /= 10;
    }
    if (mm == 0) return error.BadEngramSplit;
    return .{ .milli = mm };
}

/// TF_DS_ENGRAM_SPLIT from the environment (unset: off).
pub fn splitFromEnv() !Split {
    const v = std.c.getenv("TF_DS_ENGRAM_SPLIT") orelse return .{};
    return parseSplit(std.mem.span(v));
}

/// The column pair's link port: the ranks' port + 16 + the pair-0 rank (the Fan takes port .. port + 2).
pub const pair_port_offset = 16;

/// The pair's link at start, on every rank of the four (the switch on): rank 0 sends each follower the four ranks'
/// fabric addresses (its own, then its links' peers) and the share down the Fan's links; each pair-0 rank g listens
/// on its address at port + 16 + g and its partner g + 2 connects; then both check the other is its column partner
/// with the same columns and share. `links` is the Fan's (rank 0: followers 1..3; a follower: rank 0).
pub fn pairLink(io: std.Io, links: []const ?link.Link, master: [4]u8, port: u16, rank: u32, world: u32, split: Split, lo: usize, k: usize) !link.Link {
    if (world != 4 or links.len < 3) return error.EngramSplitNeedsTheFourNodeSplit;
    var ips: [4][4]u8 = undefined;
    var frame: [24]u8 = undefined;
    if (rank == 0) {
        ips[0] = master;
        for (1..4) |g| ips[g] = try sockIp((links[g - 1] orelse return error.EngramSplitLink).fd, true);
        std.mem.writeInt(u32, frame[0..4], split_magic, .little);
        std.mem.writeInt(u32, frame[4..8], split.milli, .little);
        for (0..4) |g| @memcpy(frame[8 + 4 * g ..][0..4], &ips[g]);
        for (1..4) |g| try links[g - 1].?.send(&frame);
    } else {
        var buf: [64]u8 = undefined;
        const got = try (links[0] orelse return error.EngramSplitLink).recv(&buf);
        if (got.len != frame.len or std.mem.readInt(u32, got[0..4], .little) != split_magic) return error.EngramSplitLink;
        if (std.mem.readInt(u32, got[4..8], .little) != split.milli) return error.EngramSplitPartner;
        for (0..4) |g| @memcpy(&ips[g], got[8 + 4 * g ..][0..4]);
    }
    const base: u32 = rank % 2;
    const pp: u16 = port + pair_port_offset + @as(u16, @intCast(base));
    var l = if (rank < 2) try link.Link.listen(ips[rank], pp) else try link.Link.connect(io, ips[base], pp, 300);
    errdefer l.close();
    // hello both ways: the partner's rank, its columns and its share
    var hello: [20]u8 = undefined;
    std.mem.writeInt(u32, hello[0..4], hello_magic, .little);
    std.mem.writeInt(u32, hello[4..8], rank, .little);
    std.mem.writeInt(u32, hello[8..12], @intCast(lo), .little);
    std.mem.writeInt(u32, hello[12..16], @intCast(k), .little);
    std.mem.writeInt(u32, hello[16..20], split.milli, .little);
    try l.send(&hello);
    var hb: [64]u8 = undefined;
    const h = try l.recv(&hb);
    if (h.len != hello.len or std.mem.readInt(u32, h[0..4], .little) != hello_magic) return error.EngramSplitLink;
    if (std.mem.readInt(u32, h[4..8], .little) != rank ^ 2) return error.EngramSplitPartner;
    if (!std.mem.eql(u8, h[8..20], hello[8..20])) return error.EngramSplitPartner;
    return l;
}

const split_magic: u32 = 0x4c505345; // "ESPL"
const hello_magic: u32 = 0x52415045; // "EPAR"
const part_magic: u32 = 0x54524150; // "PART"

/// A socket's local (or peer's) IPv4 address.
fn sockIp(fd: std.c.fd_t, peer: bool) ![4]u8 {
    var a: std.c.sockaddr.in = undefined;
    var len: std.c.socklen_t = @sizeOf(std.c.sockaddr.in);
    const rc = if (peer) std.c.getpeername(fd, @ptrCast(&a), &len) else std.c.getsockname(fd, @ptrCast(&a), &len);
    if (rc != 0) return error.EngramSplitLink;
    return @bitCast(a.addr);
}

/// A part of a chunk's rows of one Engram layer, as one rank read and decoded it: this header, then the rows' ids
/// (i64 [cnt * k]), then the decoded rows (bf16 [cnt * k, row_w]).
const Hdr = extern struct {
    magic: u32 = part_magic,
    layer: u32 = 0, // the Engram layer's index j
    start: u64 = 0, // the chunk
    n: u64 = 0,
    lo: u64 = 0, // the part's first row and rows
    cnt: u64 = 0,
    k: u64 = 0,
    row_w: u64 = 0,

    fn same(a: Hdr, b: Hdr) bool {
        return a.magic == b.magic and a.layer == b.layer and a.start == b.start and a.n == b.n and a.lo == b.lo and a.cnt == b.cnt and a.k == b.k and a.row_w == b.row_w;
    }
};

/// The parts received and not yet taken (a part from a chunk its reader cancelled stays until evicted, oldest first).
const box_n = 4;

const Msg = struct {
    state: enum { free, filling, ready, pinned } = .free,
    order: u64 = 0,
    hdr: Hdr = .{},
    ids: []i64 = &.{},
    rows: []u16 = &.{},
};

/// The column pair's link and the parts the partner sent (the mailbox; under the Prefetch's mutex).
pub const Pair = struct {
    fd: std.c.fd_t,
    first: bool, // this rank is the pair-0 rank: rows [0, a) of each chunk
    split: Split,
    rx: ?std.Thread = null, // the receiver: drains the link into the mailbox, always (so the partner's sends finish)
    box: [box_n]Msg = @splat(.{}),
    arrived: u64 = 0,
    err: ?anyerror = null, // the receiver's (the link went down)
    closing: bool = false,
    // how long the reader waits for a partner's part before it reads those rows itself (the same bytes): a guard, so a
    // part that will never come (the partners' plans diverged) costs this wait instead of a hang
    wait_ns: u64 = default_wait_ms * std.time.ns_per_ms,
    // the reader's counts for the chunk it is on (the reader's alone), added to the fill's at its end
    own_rows: u64 = 0,
    peer_rows: u64 = 0,
    xchg_ns: u64 = 0,
    fallbacks: u64 = 0,
};

/// The partner's part waited for at most this long (ms) before the reader reads the rows itself.
pub const default_wait_ms = 2000;

fn writeFd(fd: std.c.fd_t, bytes: []const u8) !void {
    var done: usize = 0;
    while (done < bytes.len) {
        const n = std.c.send(fd, bytes.ptr + done, bytes.len - done, std.c.MSG.NOSIGNAL);
        if (n <= 0) {
            if (n < 0 and std.c.errno(n) == .INTR) continue;
            return error.EngramSplitPeerDown;
        }
        done += @intCast(n);
    }
}

fn readFd(fd: std.c.fd_t, out: []u8) !void {
    var done: usize = 0;
    while (done < out.len) {
        const n = std.c.recv(fd, out.ptr + done, out.len - done, 0);
        if (n <= 0) {
            if (n < 0 and std.c.errno(n) == .INTR) continue;
            return error.EngramSplitPeerDown;
        }
        done += @intCast(n);
    }
}

/// model.passRows: a pass's rows from `start` (two chunks where both end by the replay, else one chunk or the rest).
pub fn nextRows(pass: usize, chunk: usize, start: usize, len: usize, replay: usize) usize {
    if (pass > chunk and start % chunk == 0 and start + pass <= replay and start + pass <= len) return pass;
    return @min(chunk, len - start);
}

/// A table's scale rows [rows, row_s] in RAM, read once (through the O_DIRECT descriptor where there is one, in 64 MiB
/// aligned reads, so the page cache keeps nothing of them).
const ScaleRam = struct {
    buf: []align(dio.alignment) u8,
    at: usize,
    len: usize,

    fn load(gpa: std.mem.Allocator, t: engram_io.Table) !ScaleRam {
        const len: usize = @intCast(t.rows * @as(u64, t.row_s));
        const lo = std.mem.alignBackward(u64, t.base_s, dio.alignment);
        const hi = std.mem.alignForward(u64, t.base_s + @as(u64, len), dio.alignment);
        const span: usize = @intCast(hi - lo);
        const buf = try gpa.alignedAlloc(u8, .fromByteUnits(dio.alignment), span);
        errdefer gpa.free(buf);
        const fd = if (t.fd_sd >= 0) t.fd_sd else t.fd_s;
        var off: usize = 0;
        while (off < span) {
            const want = @min(@as(usize, 64 << 20), span - off);
            const r = std.c.pread(fd, buf.ptr + off, want, @intCast(lo + @as(u64, off)));
            if (r < 0) {
                if (std.c.errno(r) == .INTR) continue;
                return error.ReadFailed;
            }
            off += @intCast(r);
            if (@as(usize, @intCast(r)) < want) break; // the file's end (the aligned tail past it holds no row)
        }
        const at: usize = @intCast(t.base_s - lo);
        if (off < at + len) return error.EndOfFile;
        return .{ .buf = buf, .at = at, .len = len };
    }

    fn rows(r: ScaleRam) []const u8 {
        return r.buf[r.at..][0..r.len];
    }
};

const Slot = struct {
    live: bool = false, // a chunk of the plan: being read, or read
    start: usize = 0,
    n: usize = 0,
    ready: [2]bool = .{ false, false }, // Engram layer j's rows decoded into rows[j]
    err: ?anyerror = null,
    rows: [2][]u16 = .{ &.{}, &.{} }, // bf16 [n, k * row_w]
};

/// A fill's counts (one JSON line when the prompt ends: stop).
pub const Stats = struct {
    chunks: u64 = 0, // chunks read
    rows: u64 = 0, // their prompt rows
    read_ns: u64 = 0, // the reader's time on them (hash, read, decode)
    takes: u64 = 0, // rows asked for (a chunk's Engram layer)
    waits: u64 = 0, // of them, asked before they were ready
    wait_ns: u64 = 0, // the asking thread's time waiting
    misses: u64 = 0, // asked for rows of no chunk of the plan (read by the asker, as unset)
    resets: u64 = 0, // the plan restarted at a chunk it did not have (a chunk size the plan did not predict)
    // TF_DS_ENGRAM_SPLIT: the read chunks' rows this rank read itself and those it took from its partner; the
    // reader's time sending its parts and waiting for / copying the partner's; parts evicted untaken
    own_rows: u64 = 0,
    peer_rows: u64 = 0,
    xchg_ns: u64 = 0,
    stale: u64 = 0,
    fallbacks: u64 = 0, // partner parts not received in Pair.wait_ns: those rows read here instead
};

pub const Prefetch = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    set: Setting,
    hasher: engram.Hasher,
    lo: usize, // this rank's hash columns [lo, lo + k) of each layer's
    k: usize,
    ne: usize, // Engram layers (at most 2)
    tabs: [2]engram_io.Table,
    aio: *engram_aio.Aio, // a ring of its own: the main thread's prompt ring and the rounds' ring are untouched
    ram: [2]?ScaleRam = .{ null, null },
    ram_bytes: usize = 0,
    // the reader's scratch: a chunk's hashes, one layer's row ids, FP8 rows and scale rows
    hashes: []i64,
    flat: []i64,
    w: []u8,
    s: []u8,
    slots: []Slot,
    pass_rows: usize,
    chunk_rows: usize,
    mutex: std.Io.Mutex = .init,
    work_cond: std.Io.Condition = .init, // the reader waits here for a plan and a free slot
    ready_cond: std.Io.Condition = .init, // takers and resets wait here
    thread: ?std.Thread = null,
    quit: bool = false,
    cancel: std.atomic.Value(bool) = .init(false),
    // the plan (under mutex): the prompt's ids so far, its replay, the next chunk to read from `next` (first_n rows
    // when set: the chunk the fill asked for; then nextRows)
    gen: u64 = 0,
    seq: []const i32 = &.{},
    replay: usize = 0,
    next: usize = 0,
    first_n: usize = 0,
    busy: bool = false, // the reader is on a chunk (outside the mutex)
    st: Stats = .{},
    pair: ?*Pair = null, // TF_DS_ENGRAM_SPLIT: the column partner's link and mailbox (usePair; null: unsplit)

    /// The reader for this rank's columns [lo, hi) of every Engram layer, chunks of at most `cap` rows, `set.depth +
    /// 1` slots, its own AIO ring of `ring_slots` reads; with set.scales the tables' scale rows loaded into RAM.
    pub fn init(gpa: std.mem.Allocator, io: std.Io, c: *const Config, hasher: engram.Hasher, tables: *const engram_io.Tables, lo: usize, hi: usize, cap: usize, chunk_rows: usize, set: Setting, ring_slots: usize) !*Prefetch {
        if (!set.on()) return error.BadEngramPrefetch;
        const ne = @min(2, c.engram_layers.slice().len);
        if (ne == 0) return error.NoEngramTables;
        const p = try gpa.create(Prefetch);
        errdefer gpa.destroy(p);
        var tabs: [2]engram_io.Table = undefined;
        var row_w: usize = 0;
        var row_s: usize = 0;
        for (0..ne) |j| {
            tabs[j] = tables.layers.get(@intCast(c.engram_layers.slice()[j])) orelse return error.MissingEngramTable;
            row_w = @max(row_w, tabs[j].row_w);
            row_s = @max(row_s, tabs[j].row_s);
        }
        const k = hi - lo;
        const aio = try engram_aio.Aio.init(gpa, io, ring_slots);
        errdefer aio.deinit();
        p.* = .{
            .gpa = gpa,
            .io = io,
            .set = set,
            .hasher = hasher,
            .lo = lo,
            .k = k,
            .ne = ne,
            .tabs = tabs,
            .aio = aio,
            .hashes = &.{},
            .flat = &.{},
            .w = &.{},
            .s = &.{},
            .slots = &.{},
            .pass_rows = cap,
            .chunk_rows = chunk_rows,
        };
        errdefer p.freeBuffers();
        if (set.scales) for (0..ne) |j| {
            p.ram[j] = try ScaleRam.load(gpa, tabs[j]);
            p.ram_bytes += p.ram[j].?.buf.len;
        };
        p.hashes = try gpa.alloc(i64, cap * hasher.layers * hasher.cols());
        p.flat = try gpa.alloc(i64, cap * k);
        p.w = try gpa.alloc(u8, cap * k * row_w);
        p.s = try gpa.alloc(u8, cap * k * row_s);
        p.slots = try gpa.alloc(Slot, set.depth + 1);
        for (p.slots) |*sl| sl.* = .{};
        for (p.slots) |*sl| for (0..ne) |j| {
            sl.rows[j] = try gpa.alloc(u16, cap * k * tabs[j].row_w);
        };
        p.thread = try std.Thread.spawn(.{}, loop, .{p});
        return p;
    }

    /// Host bytes held: the slots, the scratch and the scale rows.
    pub fn bytes(p: *const Prefetch) usize {
        var n: usize = p.ram_bytes + p.hashes.len * 8 + p.flat.len * 8 + p.w.len + p.s.len + p.aio.ring.len;
        for (p.slots) |sl| for (sl.rows) |r| {
            n += r.len * 2;
        };
        if (p.pair) |pr| for (pr.box) |b| {
            n += b.ids.len * 8 + b.rows.len * 2;
        };
        return n;
    }

    fn freeBuffers(p: *Prefetch) void {
        for (p.slots) |sl| for (sl.rows) |r| if (r.len > 0) p.gpa.free(r);
        if (p.slots.len > 0) p.gpa.free(p.slots);
        if (p.hashes.len > 0) p.gpa.free(p.hashes);
        if (p.flat.len > 0) p.gpa.free(p.flat);
        if (p.w.len > 0) p.gpa.free(p.w);
        if (p.s.len > 0) p.gpa.free(p.s);
        for (&p.ram) |*r| if (r.*) |x| {
            p.gpa.free(x.buf);
            r.* = null;
        };
        p.slots = &.{};
        p.hashes = &.{};
        p.flat = &.{};
        p.w = &.{};
        p.s = &.{};
    }

    pub fn deinit(p: *Prefetch) void {
        p.mutex.lockUncancelable(p.io);
        p.quit = true;
        p.cancel.store(true, .release);
        p.work_cond.broadcast(p.io);
        if (p.pair) |pr| {
            // the link shut both ways: a send or a receive on it returns
            pr.closing = true;
            _ = std.c.shutdown(pr.fd, std.c.SHUT.RDWR);
            p.ready_cond.broadcast(p.io);
        }
        p.mutex.unlock(p.io);
        if (p.thread) |th| th.join();
        if (p.pair) |pr| {
            if (pr.rx) |th| th.join();
            _ = std.c.close(pr.fd);
            for (pr.box) |b| {
                if (b.ids.len > 0) p.gpa.free(b.ids);
                if (b.rows.len > 0) p.gpa.free(b.rows);
            }
            p.gpa.destroy(pr);
            p.pair = null;
        }
        p.aio.deinit();
        p.freeBuffers();
        p.gpa.destroy(p);
    }

    /// TF_DS_ENGRAM_SPLIT: read only this rank's part of each chunk's rows (`first`: the pair-0 rank's [0, a)) and
    /// take the rest from the column partner over `fd` (a connected stream socket, pairLink's; owned from here on,
    /// closed by deinit, also on an error here). Before the first chunk: the reader reads `pair` after taking the
    /// mutex.
    pub fn usePair(p: *Prefetch, fd: std.c.fd_t, split: Split, first: bool) !void {
        errdefer _ = std.c.close(fd);
        if (!split.on() or p.pair != null) return error.BadEngramSplit;
        const pr = try p.gpa.create(Pair);
        errdefer p.gpa.destroy(pr);
        pr.* = .{ .fd = fd, .first = first, .split = split };
        var row_w: usize = 0;
        for (0..p.ne) |j| row_w = @max(row_w, p.tabs[j].row_w);
        errdefer for (pr.box) |b| {
            if (b.ids.len > 0) p.gpa.free(b.ids);
            if (b.rows.len > 0) p.gpa.free(b.rows);
        };
        for (&pr.box) |*b| {
            b.ids = try p.gpa.alloc(i64, p.pass_rows * p.k);
            b.rows = try p.gpa.alloc(u16, p.pass_rows * p.k * row_w);
        }
        p.mutex.lockUncancelable(p.io);
        defer p.mutex.unlock(p.io);
        if (p.seq.len != 0 or p.busy) return error.BadEngramSplit; // only before the first chunk
        pr.rx = try std.Thread.spawn(.{}, rxLoop, .{ p, pr });
        p.pair = pr;
    }

    // -- the column pair's link (TF_DS_ENGRAM_SPLIT) ----------------------------------------------------------------

    /// The receiver: every part the partner sends into the mailbox (a free entry, else the oldest part not taken),
    /// until the link goes down or closes.
    fn rxLoop(p: *Prefetch, pr: *Pair) void {
        while (true) {
            var h: Hdr = undefined;
            readFd(pr.fd, std.mem.asBytes(&h)) catch |e| return p.rxFail(pr, e);
            if (h.magic != part_magic or h.layer >= p.ne or h.k != p.k or h.row_w != p.tabs[h.layer].row_w or h.cnt == 0 or h.cnt > p.pass_rows or h.lo + h.cnt > h.n)
                return p.rxFail(pr, error.EngramSplitFrame);
            const m: usize = @intCast(h.cnt * h.k);
            p.mutex.lockUncancelable(p.io);
            var bi: ?usize = null;
            for (pr.box, 0..) |b, i| if (b.state == .free) {
                bi = i;
                break;
            };
            if (bi == null) {
                var oldest: u64 = std.math.maxInt(u64);
                for (pr.box, 0..) |b, i| if (b.state == .ready and b.order < oldest) {
                    oldest = b.order;
                    bi = i;
                };
                if (bi != null) p.st.stale += 1;
            }
            const b = &pr.box[bi orelse {
                p.mutex.unlock(p.io);
                return p.rxFail(pr, error.EngramSplitFrame); // (cannot happen: one entry pinned at most)
            }];
            b.state = .filling;
            p.mutex.unlock(p.io);
            readFd(pr.fd, std.mem.sliceAsBytes(b.ids[0..m])) catch |e| return p.rxFail(pr, e);
            readFd(pr.fd, std.mem.sliceAsBytes(b.rows[0 .. m * h.row_w])) catch |e| return p.rxFail(pr, e);
            p.mutex.lockUncancelable(p.io);
            b.hdr = h;
            pr.arrived += 1;
            b.order = pr.arrived;
            b.state = .ready;
            p.ready_cond.broadcast(p.io);
            p.mutex.unlock(p.io);
        }
    }

    fn rxFail(p: *Prefetch, pr: *Pair, e: anyerror) void {
        p.mutex.lockUncancelable(p.io);
        defer p.mutex.unlock(p.io);
        for (&pr.box) |*b| if (b.state == .filling) {
            b.state = .free;
        };
        pr.err = if (pr.closing) error.Cancelled else e;
        p.ready_cond.broadcast(p.io);
    }

    /// Chunk [start, start + n)'s layer j with this rank's part decoded in `out`: the part sent to the partner (with
    /// its ids), and the partner's part (the one whose ids are this chunk's own for its rows) copied in. A cancel (a
    /// reset) stops the wait, never a send: a part goes out whole, so the link stays in step.
    fn exchange(p: *Prefetch, pr: *Pair, start: usize, n: usize, j: usize, out: []u16) !void {
        const t0 = now(p.io);
        defer pr.xchg_ns += now(p.io) - t0;
        const k = p.k;
        const rw = p.tabs[j].row_w;
        const mine = pr.split.own(n, pr.first);
        const theirs = pr.split.own(n, !pr.first);
        if (mine[1] > mine[0]) {
            const h: Hdr = .{ .layer = @intCast(j), .start = start, .n = n, .lo = mine[0], .cnt = mine[1] - mine[0], .k = k, .row_w = rw };
            try writeFd(pr.fd, std.mem.asBytes(&h));
            try writeFd(pr.fd, std.mem.sliceAsBytes(p.flat[mine[0] * k .. mine[1] * k]));
            try writeFd(pr.fd, std.mem.sliceAsBytes(out[mine[0] * k * rw .. mine[1] * k * rw]));
        }
        if (theirs[1] <= theirs[0]) return;
        const want: Hdr = .{ .layer = @intCast(j), .start = start, .n = n, .lo = theirs[0], .cnt = theirs[1] - theirs[0], .k = k, .row_w = rw };
        const ids = p.flat[theirs[0] * k .. theirs[1] * k];
        const deadline = now(p.io) + pr.wait_ns;
        p.mutex.lockUncancelable(p.io);
        const b = while (true) {
            var hit: ?*Msg = null;
            for (&pr.box) |*x| if (x.state == .ready and x.hdr.same(want) and std.mem.eql(i64, x.ids[0..ids.len], ids)) {
                hit = x;
                break;
            };
            if (hit) |x| break x;
            if (p.cancel.load(.acquire) or p.quit) {
                p.mutex.unlock(p.io);
                return error.Cancelled;
            }
            if (pr.err) |e| {
                p.mutex.unlock(p.io);
                return e;
            }
            const t = now(p.io);
            if (t >= deadline) {
                // the part has not come: these rows read here, as unsplit (the same bytes); if it comes later it is
                // never taken (stale)
                p.mutex.unlock(p.io);
                pr.fallbacks += 1;
                return p.readRows(p.tabs[j], j, ids, out[theirs[0] * k * rw ..][0 .. ids.len * rw]);
            }
            p.ready_cond.waitTimeout(p.io, &p.mutex, .{ .duration = .{ .raw = .fromNanoseconds(@intCast(deadline - t)), .clock = .awake } }) catch {};
        };
        b.state = .pinned;
        p.mutex.unlock(p.io);
        @memcpy(out[theirs[0] * k * rw ..][0 .. ids.len * rw], b.rows[0 .. ids.len * rw]);
        p.mutex.lockUncancelable(p.io);
        b.state = .free;
        p.mutex.unlock(p.io);
    }

    // -- the reader ------------------------------------------------------------------------------------------------

    fn loop(p: *Prefetch) void {
        p.mutex.lockUncancelable(p.io);
        while (!p.quit) {
            const si = p.claim() orelse {
                p.work_cond.waitUncancelable(p.io, &p.mutex);
                continue;
            };
            const gen = p.gen;
            const sl = &p.slots[si];
            const seq = p.seq;
            const start = sl.start;
            const n = sl.n;
            p.busy = true;
            const pr = p.pair;
            p.mutex.unlock(p.io);
            if (pr) |x| {
                x.own_rows = 0;
                x.peer_rows = 0;
                x.xchg_ns = 0;
                x.fallbacks = 0;
            }
            const t0 = now(p.io);
            var failed: ?anyerror = null;
            for (0..p.ne) |j| {
                p.readLayer(seq, start, n, j, sl.rows[j]) catch |err| {
                    failed = err;
                };
                p.mutex.lockUncancelable(p.io);
                if (p.gen == gen) {
                    if (failed) |e| sl.err = e else sl.ready[j] = true;
                    p.ready_cond.broadcast(p.io);
                }
                p.mutex.unlock(p.io);
                if (failed != null) break;
            }
            const dt = now(p.io) - t0;
            p.mutex.lockUncancelable(p.io);
            p.busy = false;
            if (p.gen == gen and failed == null) {
                p.st.chunks += 1;
                p.st.rows += n;
                p.st.read_ns += dt;
                if (pr) |x| {
                    p.st.own_rows += x.own_rows;
                    p.st.peer_rows += x.peer_rows;
                    p.st.xchg_ns += x.xchg_ns;
                    p.st.fallbacks += x.fallbacks;
                }
            }
            p.ready_cond.broadcast(p.io);
        }
        p.mutex.unlock(p.io);
    }

    /// The plan's next chunk into a free slot (`mutex` held), or null: no plan, the plan read to its end, or every
    /// slot holding a chunk the fill has not passed.
    fn claim(p: *Prefetch) ?usize {
        if (p.seq.len == 0 or p.next >= p.seq.len) return null;
        for (p.slots, 0..) |*sl, i| {
            if (sl.live) continue;
            const n = if (p.first_n > 0) p.first_n else nextRows(p.pass_rows, p.chunk_rows, p.next, p.seq.len, p.replay);
            sl.live = true;
            sl.start = p.next;
            sl.n = @min(n, p.seq.len - p.next);
            sl.ready = .{ false, false };
            sl.err = null;
            p.next += sl.n;
            p.first_n = 0;
            return i;
        }
        return null;
    }

    /// Chunk [start, start + n)'s rows of Engram layer j into out: engramApply's own read (and engramAhead1's), on
    /// this reader's ring and scratch. With a Pair: this rank's part of the rows read and decoded the same way (the
    /// ids of its rows [lo, hi), into out at their places), then the partner's part taken (exchange).
    fn readLayer(p: *Prefetch, seq: []const i32, start: usize, n: usize, j: usize, out: []u16) !void {
        const t = p.tabs[j];
        const cols = p.hasher.cols();
        const k = p.k;
        if (n * k * t.row_w > out.len) return error.BadChunk;
        if (j == 0) p.hasher.hashes(seq, start, n, p.hashes[0 .. n * p.hasher.layers * cols]);
        for (0..n) |r| {
            for (0..k) |q| p.flat[r * k + q] = p.hashes[(r * p.hasher.layers + j) * cols + p.lo + q];
        }
        // the rows read here: all of them, or this rank's part of a column pair's split
        const rows: [2]usize = if (p.pair) |pr| pr.split.own(n, pr.first) else .{ 0, n };
        const f0 = rows[0] * k;
        const m = (rows[1] - rows[0]) * k;
        if (m > 0) try p.readRows(t, j, p.flat[f0..][0..m], out[f0 * t.row_w ..][0 .. m * t.row_w]);
        if (p.pair) |pr| {
            if (j == 0) {
                pr.own_rows += rows[1] - rows[0];
                pr.peer_rows += n - (rows[1] - rows[0]);
            }
            try p.exchange(pr, start, n, j, out);
        }
    }

    /// The rows of `ids` (one Engram layer j's) read and decoded into `out` ([ids.len, row_w]): the weight rows by
    /// AIO, the scale rows from RAM with `scales` (else by AIO too), decodeRows. ids.len at most the scratch's.
    fn readRows(p: *Prefetch, t: engram_io.Table, j: usize, ids: []const i64, out: []u16) !void {
        const m = ids.len;
        const w = p.w[0 .. m * t.row_w];
        const s = p.s[0 .. m * t.row_s];
        if (p.ram[j]) |ram| {
            try p.gather(t, ids, w, null);
            const sr = ram.rows();
            for (0..m) |i| {
                const id: usize = @intCast(ids[i]);
                if (id >= t.rows) return error.RowOutOfRange;
                @memcpy(s[i * t.row_s ..][0..t.row_s], sr[id * t.row_s ..][0..t.row_s]);
            }
        } else try p.gather(t, ids, w, s);
        engram.decodeRows(w, s, out[0 .. m * t.row_w], m, t.row_w, t.row_s, 8);
    }

    /// prompt.aioGather on this reader's ring (batches of a quarter of its slots, all submitted, then waited for),
    /// the weight rows alone when out_s is null; stops submitting when a reset cancels the chunk.
    fn gather(p: *Prefetch, t: engram_io.Table, ids: []const i64, out_w: []u8, out_s: ?[]u8) !void {
        const aio = p.aio;
        const per = @max(aio.slots / 4, 1);
        var batch: [64]u64 = undefined;
        var nb: usize = 0;
        var at: usize = 0;
        var failed: ?anyerror = null;
        while (at < ids.len) : (at += per) {
            if (p.cancel.load(.acquire)) {
                failed = error.Cancelled;
                break;
            }
            const n = @min(per, ids.len - at);
            if (nb == batch.len) {
                failed = error.TooManyRows;
                break;
            }
            const id = (if (out_s) |s|
                aio.start(t, ids[at..][0..n], out_w[at * t.row_w ..][0 .. n * t.row_w], s[at * t.row_s ..][0 .. n * t.row_s])
            else
                aio.startW(t, ids[at..][0..n], out_w[at * t.row_w ..][0 .. n * t.row_w])) catch |err| {
                failed = err;
                break;
            };
            batch[nb] = id;
            nb += 1;
        }
        // every submitted batch lands before its buffers can be reused
        for (batch[0..nb]) |id| aio.wait(id) catch |err| {
            if (failed == null) failed = err;
        };
        if (failed) |e| return e;
    }

    // -- the fill's side (the thread that runs the chunks) ---------------------------------------------------------

    /// The slot holding chunk [start, start + n) whole (`mutex` held).
    fn covering(p: *Prefetch, start: usize, n: usize) ?usize {
        for (p.slots, 0..) |sl, i| if (sl.live and sl.start <= start and start + n <= sl.start + sl.n) return i;
        return null;
    }

    /// Drop the plan (`mutex` held): the reader's chunk cancelled and waited for, every slot free.
    fn resetLocked(p: *Prefetch) void {
        p.gen += 1;
        p.seq = &.{};
        p.next = 0;
        p.first_n = 0;
        p.cancel.store(true, .release);
        if (p.pair != null) p.ready_cond.broadcast(p.io); // (a reader waiting for its partner's part: it stops)
        while (p.busy) p.ready_cond.waitUncancelable(p.io, &p.mutex);
        p.cancel.store(false, .release);
        for (p.slots) |*sl| sl.live = false;
    }

    /// The fill is at chunk [start, start + n) of prompt `full` (its ids so far: the lane's fill_seq, else the chunk's
    /// own sequence), with `replay`: the slots of chunks before it are given back, and the plan restarts at this
    /// chunk unless it already has it (read, being read, or next to read).
    pub fn begin(p: *Prefetch, full: []const i32, start: usize, n: usize, replay: usize) void {
        p.mutex.lockUncancelable(p.io);
        defer p.mutex.unlock(p.io);
        // chunks wholly before this one are done: their rows' uploads have returned
        for (p.slots) |*sl| if (sl.live and sl.start + sl.n <= start) {
            sl.live = false;
        };
        const same = p.seq.len > 0 and p.seq.ptr == full.ptr and p.seq.len == full.len and p.replay == replay;
        var have = same and p.covering(start, n) != null;
        if (same and !have and p.next == start) {
            const nn = if (p.first_n > 0) p.first_n else nextRows(p.pass_rows, p.chunk_rows, p.next, p.seq.len, p.replay);
            have = nn >= n;
        }
        if (!have) {
            if (p.seq.len > 0) p.st.resets += 1;
            p.resetLocked();
            p.seq = full;
            p.replay = replay;
            p.next = start;
            p.first_n = n;
        }
        p.work_cond.broadcast(p.io);
    }

    /// Chunk [start, start + n)'s rows of Engram layer l (its offset in its slot's), waiting for the reader when it is
    /// behind; null when no chunk of the plan holds them (the caller reads them itself, as unset).
    pub fn take(p: *Prefetch, l: usize, start: usize, n: usize) !?[]u16 {
        if (l >= p.ne) return null;
        p.mutex.lockUncancelable(p.io);
        defer p.mutex.unlock(p.io);
        p.st.takes += 1;
        var si = p.covering(start, n);
        const t0 = now(p.io);
        var waited = false;
        // the plan's next chunk, not yet taken up by the reader (which takes chunks in order into the free slots
        // begin left): it will be
        while (si == null and p.seq.len > 0 and p.next == start and !p.quit) {
            waited = true;
            p.ready_cond.waitUncancelable(p.io, &p.mutex);
            si = p.covering(start, n);
        }
        const sl = if (si) |i| &p.slots[i] else {
            p.st.misses += 1;
            return null;
        };
        while (!sl.ready[l] and sl.err == null and sl.live) {
            waited = true;
            p.ready_cond.waitUncancelable(p.io, &p.mutex);
        }
        if (waited) {
            p.st.waits += 1;
            p.st.wait_ns += now(p.io) - t0;
        }
        if (sl.err) |e| return e;
        if (!sl.live or !sl.ready[l]) return null;
        return sl.rows[l][(start - sl.start) * p.k * p.tabs[l].row_w ..];
    }

    /// The prompt is done (model.joinAhead): the plan dropped (a chunk being read cancelled and waited for), and the
    /// fill's counts printed as one line when it read anything.
    pub fn stop(p: *Prefetch) void {
        p.mutex.lockUncancelable(p.io);
        defer p.mutex.unlock(p.io);
        if (p.seq.len == 0 and p.st.takes == 0 and p.st.chunks == 0) return;
        p.resetLocked();
        const s = p.st;
        p.st = .{};
        if (s.takes == 0 and s.chunks == 0) return;
        if (p.pair) |pr| {
            var nb: [16]u8 = undefined;
            std.debug.print("{{\"engram_pf\": {{\"depth\": {d}, \"scales\": {s}, \"chunks\": {d}, \"rows\": {d}, \"read_ms\": {d:.1}, \"takes\": {d}, \"waits\": {d}, \"wait_ms\": {d:.1}, \"misses\": {d}, \"resets\": {d}, \"split\": \"{s}\", \"own_rows\": {d}, \"peer_rows\": {d}, \"xchg_ms\": {d:.1}, \"stale\": {d}, \"fallbacks\": {d}}}}}\n", .{ p.set.depth, if (p.set.scales) "true" else "false", s.chunks, s.rows, ms(s.read_ns), s.takes, s.waits, ms(s.wait_ns), s.misses, s.resets, pr.split.names(&nb), s.own_rows, s.peer_rows, ms(s.xchg_ns), s.stale, s.fallbacks });
            return;
        }
        std.debug.print("{{\"engram_pf\": {{\"depth\": {d}, \"scales\": {s}, \"chunks\": {d}, \"rows\": {d}, \"read_ms\": {d:.1}, \"takes\": {d}, \"waits\": {d}, \"wait_ms\": {d:.1}, \"misses\": {d}, \"resets\": {d}}}}}\n", .{ p.set.depth, if (p.set.scales) "true" else "false", s.chunks, s.rows, ms(s.read_ns), s.takes, s.waits, ms(s.wait_ns), s.misses, s.resets });
    }

    /// Stats so far (a test's view).
    pub fn stats(p: *Prefetch) Stats {
        p.mutex.lockUncancelable(p.io);
        defer p.mutex.unlock(p.io);
        return p.st;
    }
};

fn now(io: std.Io) u64 {
    return @intCast(std.Io.Timestamp.now(io, .awake).nanoseconds);
}

fn ms(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / 1e6;
}

test "TF_DS_ENGRAM_PREFETCH values: a depth or prompt, then scales; anything else refuses" {
    try std.testing.expect(!(try parse("")).on());
    try std.testing.expect(!(try parse("off")).on());
    try std.testing.expect(!(try parse("0")).on());
    try std.testing.expectEqual(@as(usize, 1), (try parse("1")).depth);
    try std.testing.expectEqual(@as(usize, 2), (try parse("2,scales")).depth);
    try std.testing.expect((try parse("2,scales")).scales);
    try std.testing.expectEqual(@as(usize, max_depth), (try parse("prompt")).depth);
    try std.testing.expect((try parse("prompt,scales")).scales);
    var nb: [32]u8 = undefined;
    try std.testing.expectEqualStrings("prompt,scales", (try parse("prompt,scales")).names(&nb));
    try std.testing.expectEqualStrings("1", (try parse("1")).names(&nb));
    for ([_][]const u8{ "scales", "33", "-1", "+1", "1,", "1,scales,scales", "1,2", "x", "1.5", ",1", "on", "all", "0,scales" }) |bad|
        try std.testing.expectError(error.BadEngramPrefetch, parse(bad));
}

test "nextRows is model.passRows" {
    const passRows = @import("model.zig").passRows;
    for ([_]usize{ 2048, 4096 }) |pass| for ([_]usize{ 0, 2048, 4096, 6144, 8192 }) |start| for ([_]usize{ 8193, 10000, 16384 }) |len| for ([_]usize{ 0, 4096, 6000, 12288 }) |replay|
        try std.testing.expectEqual(passRows(pass, start, len, replay), nextRows(pass, 2048, start, len, replay));
}

test "prefetched rows equal the rows engramApply reads itself, at every depth, with and without the scales in RAM" {
    if (@import("builtin").os.tag != .linux) return error.SkipZigTest;
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    // two toy tables (layers 1 and 14 of the test config), [rows, 64] FP8 and [rows, 2] scales each
    const rows = 80;
    var json_buf: [1024]u8 = undefined;
    const tw = rows * 64;
    const ts = rows * 2;
    const json = try std.fmt.bufPrint(&json_buf, "{{\"layers.1.engram.embed.weight\":{{\"dtype\":\"F8_E4M3\",\"shape\":[{d},64],\"data_offsets\":[0,{d}]}},\"layers.1.engram.embed.scale\":{{\"dtype\":\"F8_E8M0\",\"shape\":[{d},2],\"data_offsets\":[{d},{d}]}},\"layers.14.engram.embed.weight\":{{\"dtype\":\"F8_E4M3\",\"shape\":[{d},64],\"data_offsets\":[{d},{d}]}},\"layers.14.engram.embed.scale\":{{\"dtype\":\"F8_E8M0\",\"shape\":[{d},2],\"data_offsets\":[{d},{d}]}}}}", .{ rows, tw, rows, tw, tw + ts, rows, tw + ts, 2 * tw + ts, rows, 2 * tw + ts, 2 * tw + 2 * ts });
    var file: std.ArrayList(u8) = .empty;
    defer file.deinit(a);
    var len8: [8]u8 = undefined;
    std.mem.writeInt(u64, &len8, json.len, .little);
    try file.appendSlice(a, &len8);
    try file.appendSlice(a, json);
    for (0..2 * tw + 2 * ts) |i| try file.append(a, @intCast((i * 29 + 11) % 251));
    try tmp.dir.writeFile(io, .{ .sub_path = "engram.safetensors", .data = file.items });
    const dir = try std.fs.path.join(a, &.{ ".zig-cache", "tmp", &tmp.sub_path });
    defer a.free(dir);
    var tables = try engram_io.Tables.open(a, io, dir);
    defer tables.close();
    // the hasher of engram.zig's toy test: vocabulary 10, one head (three columns a layer: rows < 41 and < 71)
    var why: @import("config.zig").Why = .{};
    var c = try @import("config.zig").parse(a, @import("config.zig").test_config, &why);
    c.engram_vocab = 10;
    c.engram_heads = 1;
    c.engram_pad = 0;
    const map = [_]i32{ 7, 1, 2, 3, 4, 5 };
    const h = engram.Hasher.init(c, &map, .{ .{ 3, 5, 7, 9 }, .{ 11, 13, 15, 17 } });
    const pool = try engram_io.Pool.init(a, io, 2);
    defer pool.deinit(a);
    // the Engram host as model.open builds it with TF_DS_ENGRAM_PREFETCH unset: no reader (its rows read as served)
    {
        var arena = std.heap.ArenaAllocator.init(a);
        defer arena.deinit();
        const eh = try @import("prompt.zig").EngramHost.init(arena.allocator(), &c, h, &tables, pool, 0, 2, 32);
        try std.testing.expect(eh.pf == null);
        try std.testing.expect(eh.paio == null);
        try std.testing.expect(eh.aio == null);
        try std.testing.expect(!eh.ahead_ready[0] and !eh.ahead_ready[1]);
    }
    var prng = std.Random.DefaultPrng.init(11);
    var seq: [200]i32 = undefined;
    for (&seq) |*x| x.* = prng.random().intRangeLessThan(i32, 0, 6);
    seq[37] = -1; // an image position: its n-grams dead
    const lo = 1;
    const hi = 3;
    const k = hi - lo;
    const pass = 32;
    const chunk = 16;
    const replay = 120;
    // engramApply's own read of a chunk's layer j (the reference)
    const Ref = struct {
        fn rowsOf(al: std.mem.Allocator, hs: engram.Hasher, tabs: *const engram_io.Tables, pl: *engram_io.Pool, ids: []const i32, start: usize, n: usize, j: usize) ![]u16 {
            const t = tabs.layers.get(if (j == 0) 1 else 14).?;
            const cols = hs.cols();
            const hh = try al.alloc(i64, n * hs.layers * cols);
            defer al.free(hh);
            hs.hashes(ids, start, n, hh);
            const flat = try al.alloc(i64, n * k);
            defer al.free(flat);
            for (0..n) |r| for (0..k) |q| {
                flat[r * k + q] = hh[(r * hs.layers + j) * cols + lo + q];
            };
            const w = try al.alloc(u8, n * k * t.row_w);
            defer al.free(w);
            const s = try al.alloc(u8, n * k * t.row_s);
            defer al.free(s);
            try pl.gather(t, flat, w, s);
            const out = try al.alloc(u16, n * k * t.row_w);
            engram.decodeRows(w, s, out, n * k, t.row_w, t.row_s, 8);
            return out;
        }
    };
    for ([_][]const u8{ "1", "2", "prompt", "1,scales", "prompt,scales" }) |val| {
        const set = try parse(val);
        const pf = Prefetch.init(a, io, &c, h, &tables, lo, hi, pass, chunk, set, 64) catch |e| switch (e) {
            error.AioUnavailable => return error.SkipZigTest,
            else => return e,
        };
        defer pf.deinit();
        try std.testing.expect(pf.pair == null); // (TF_DS_ENGRAM_SPLIT unset: no partner, the rows all read here)
        // a prompt as the lane fills it (passes of two chunks up to the replay, then single chunks), then a prompt
        // whose second chunk is cut short (a kept state): the plan restarts where its sizes stop matching
        for (0..2) |round| {
            var start: usize = 0;
            var chunks: usize = 0;
            while (start < seq.len) : (chunks += 1) {
                var n = nextRows(pass, chunk, start, seq.len, replay);
                if (round == 1 and chunks == 1) n = chunk;
                pf.begin(&seq, start, n, replay);
                for (0..2) |j| {
                    const want = try Ref.rowsOf(a, h, &tables, pool, seq[0 .. start + n], start, n, j);
                    defer a.free(want);
                    const got = (try pf.take(j, start, n)) orelse return error.TestExpectedRows;
                    try std.testing.expectEqualSlices(u16, want, got[0..want.len]);
                }
                start += n;
            }
            const st = pf.stats();
            try std.testing.expectEqual(@as(u64, 0), st.misses);
            try std.testing.expectEqual(@as(u64, if (round == 1) 1 else 0), st.resets);
            pf.stop();
        }
        // rows of no planned chunk: null (the caller reads them itself)
        try std.testing.expectEqual(@as(?[]u16, null), try pf.take(0, 0, 16));
        // a prompt dropped half way (stop while the reader may be ahead), then another from a kept state
        pf.begin(&seq, 0, pass, replay);
        _ = (try pf.take(0, 0, pass)) orelse return error.TestExpectedRows; // (asked at once: taken up by the reader)
        pf.stop();
        pf.begin(&seq, 64, pass, replay);
        const want = try Ref.rowsOf(a, h, &tables, pool, seq[0 .. 64 + pass], 64, pass, 1);
        defer a.free(want);
        const got = (try pf.take(1, 64, pass)).?;
        try std.testing.expectEqualSlices(u16, want, got[0..want.len]);
        pf.stop();
    }
    try std.testing.expectError(error.BadEngramPrefetch, Prefetch.init(a, io, &c, h, &tables, lo, hi, pass, chunk, .{}, 64));
}

test "TF_DS_ENGRAM_SPLIT values: a share above 0 up to 1; anything else refuses" {
    try std.testing.expect(!(try parseSplit("")).on());
    try std.testing.expect(!(try parseSplit("off")).on());
    for ([_][]const u8{ "1", "1.0", "1.00", "1.000" }) |v| try std.testing.expectEqual(@as(u32, 1000), (try parseSplit(v)).milli);
    try std.testing.expectEqual(@as(u32, 500), (try parseSplit("0.5")).milli);
    try std.testing.expectEqual(@as(u32, 750), (try parseSplit("0.75")).milli);
    try std.testing.expectEqual(@as(u32, 667), (try parseSplit("0.667")).milli);
    try std.testing.expectEqual(@as(u32, 1), (try parseSplit("0.001")).milli);
    var nb: [16]u8 = undefined;
    for ([_][2][]const u8{ .{ "1", "1" }, .{ "1.00", "1" }, .{ "0.5", "0.5" }, .{ "0.50", "0.5" }, .{ "0.75", "0.75" }, .{ "0.667", "0.667" }, .{ "0.05", "0.05" }, .{ "0.001", "0.001" } }) |x|
        try std.testing.expectEqualStrings(x[1], (try parseSplit(x[0])).names(&nb));
    try std.testing.expectEqualStrings("off", (Split{}).names(&nb));
    for ([_][]const u8{ "0", "0.0", "0.000", "1.5", "1.01", "2", "10", "0.1234", ".5", "0.", "1.", "-0.5", "+1", " 1", "0,5", "0.5x", "on", "half", "1.0000" }) |bad|
        try std.testing.expectError(error.BadEngramSplit, parseSplit(bad));
    // the two parts tile the chunk, the pair-0 rank's first
    for ([_]u32{ 1, 500, 667, 750, 1000 }) |mm| for ([_]usize{ 1, 7, 2048, 4096 }) |n| {
        const sp: Split = .{ .milli = mm };
        const a = sp.own(n, true);
        const b = sp.own(n, false);
        try std.testing.expectEqual(@as(usize, 0), a[0]);
        try std.testing.expectEqual(a[1], b[0]);
        try std.testing.expectEqual(n, b[1]);
        try std.testing.expectEqual(n * mm / 1000, a[1]);
    };
    try std.testing.expectEqual([2]usize{ 0, 3072 }, (Split{ .milli = 750 }).own(4096, true));
}

test "the column pair's link: the four ranks' addresses down the Fan, each pair-0 rank listens, its partner connects" {
    const io = std.testing.io;
    const Fan = @import("prompt2d.zig").Fan;
    const port: u16 = 31000 + @as(u16, @intCast(@mod(std.Io.Timestamp.now(io, .awake).nanoseconds, 400))) * 4;
    const Rank = struct {
        fn run(g: u32, p: u16, ok: *bool) void {
            var f = Fan.open(std.testing.io, .{ 127, 0, 0, 1 }, p, g, 4) catch return;
            defer f.close();
            // ranks g and g ^ 2 hold the same columns
            var l = pairLink(std.testing.io, &f.links, .{ 127, 0, 0, 1 }, p, g, 4, .{ .milli = 750 }, (g % 2) * 12, 12) catch return;
            defer l.close();
            // one frame each way: the link is the partner's
            var b: [8]u8 = undefined;
            std.mem.writeInt(u32, b[0..4], g, .little);
            l.send(b[0..4]) catch return;
            const got = l.recv(&b) catch return;
            ok.* = got.len == 4 and std.mem.readInt(u32, got[0..4], .little) == g ^ 2;
        }
    };
    var ok = [_]bool{ false, false, false, false };
    var th: [4]std.Thread = undefined;
    for (0..4) |g| th[g] = try std.Thread.spawn(.{}, Rank.run, .{ @as(u32, @intCast(g)), port, &ok[g] });
    for (th) |t| t.join();
    for (ok) |x| try std.testing.expect(x);
}

test "split rows equal the rows engramApply reads itself, for both column partners, at every share" {
    if (@import("builtin").os.tag != .linux) return error.SkipZigTest;
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    // the toy tables of the prefetch test: layers 1 and 14, [rows, 64] FP8 and [rows, 2] scales each
    const rows = 80;
    var json_buf: [1024]u8 = undefined;
    const tw = rows * 64;
    const ts = rows * 2;
    const json = try std.fmt.bufPrint(&json_buf, "{{\"layers.1.engram.embed.weight\":{{\"dtype\":\"F8_E4M3\",\"shape\":[{d},64],\"data_offsets\":[0,{d}]}},\"layers.1.engram.embed.scale\":{{\"dtype\":\"F8_E8M0\",\"shape\":[{d},2],\"data_offsets\":[{d},{d}]}},\"layers.14.engram.embed.weight\":{{\"dtype\":\"F8_E4M3\",\"shape\":[{d},64],\"data_offsets\":[{d},{d}]}},\"layers.14.engram.embed.scale\":{{\"dtype\":\"F8_E8M0\",\"shape\":[{d},2],\"data_offsets\":[{d},{d}]}}}}", .{ rows, tw, rows, tw, tw + ts, rows, tw + ts, 2 * tw + ts, rows, 2 * tw + ts, 2 * tw + 2 * ts });
    var file: std.ArrayList(u8) = .empty;
    defer file.deinit(a);
    var len8: [8]u8 = undefined;
    std.mem.writeInt(u64, &len8, json.len, .little);
    try file.appendSlice(a, &len8);
    try file.appendSlice(a, json);
    for (0..2 * tw + 2 * ts) |i| try file.append(a, @intCast((i * 29 + 11) % 251));
    try tmp.dir.writeFile(io, .{ .sub_path = "engram.safetensors", .data = file.items });
    const dir = try std.fs.path.join(a, &.{ ".zig-cache", "tmp", &tmp.sub_path });
    defer a.free(dir);
    var tables = try engram_io.Tables.open(a, io, dir);
    defer tables.close();
    var why: @import("config.zig").Why = .{};
    var c = try @import("config.zig").parse(a, @import("config.zig").test_config, &why);
    c.engram_vocab = 10;
    c.engram_heads = 1;
    c.engram_pad = 0;
    const map = [_]i32{ 7, 1, 2, 3, 4, 5 };
    const h = engram.Hasher.init(c, &map, .{ .{ 3, 5, 7, 9 }, .{ 11, 13, 15, 17 } });
    const pool = try engram_io.Pool.init(a, io, 2);
    defer pool.deinit(a);
    var prng = std.Random.DefaultPrng.init(23);
    var seq: [200]i32 = undefined;
    for (&seq) |*x| x.* = prng.random().intRangeLessThan(i32, 0, 6);
    seq[41] = -1; // an image position
    var seq2: [120]i32 = undefined; // a second prompt (another's ids, the same chunk starts)
    for (&seq2) |*x| x.* = prng.random().intRangeLessThan(i32, 0, 6);
    const lo = 1;
    const hi = 3;
    const k = hi - lo;
    const pass = 32;
    const chunk = 16;
    const replay = 120;
    const Ref = struct {
        fn rowsOf(al: std.mem.Allocator, hs: engram.Hasher, tabs: *const engram_io.Tables, pl: *engram_io.Pool, ids: []const i32, start: usize, n: usize, j: usize) ![]u16 {
            const t = tabs.layers.get(if (j == 0) 1 else 14).?;
            const cols = hs.cols();
            const hh = try al.alloc(i64, n * hs.layers * cols);
            defer al.free(hh);
            hs.hashes(ids, start, n, hh);
            const flat = try al.alloc(i64, n * k);
            defer al.free(flat);
            for (0..n) |r| for (0..k) |q| {
                flat[r * k + q] = hh[(r * hs.layers + j) * cols + lo + q];
            };
            const w = try al.alloc(u8, n * k * t.row_w);
            defer al.free(w);
            const s = try al.alloc(u8, n * k * t.row_s);
            defer al.free(s);
            try pl.gather(t, flat, w, s);
            const out = try al.alloc(u16, n * k * t.row_w);
            engram.decodeRows(w, s, out, n * k, t.row_w, t.row_s, 8);
            return out;
        }
    };
    // the script both partners run, as their lanes would (the same frames in the same order): a prompt's chunks;
    // the same prompt with its second chunk cut short (one reset); the second prompt dropped after its first chunk's
    // first layer while the readers run ahead; the first prompt again from a kept state at 64
    const Step = struct { seq: []const i32, start: usize, n: usize, layers: usize, stop_after: bool };
    var steps: std.ArrayList(Step) = .empty;
    defer steps.deinit(a);
    for (0..2) |round| {
        var start: usize = 0;
        var chunks: usize = 0;
        while (start < seq.len) : (chunks += 1) {
            var n = nextRows(pass, chunk, start, seq.len, replay);
            if (round == 1 and chunks == 1) n = chunk;
            try steps.append(a, .{ .seq = &seq, .start = start, .n = n, .layers = 2, .stop_after = false });
            start += n;
        }
        steps.items[steps.items.len - 1].stop_after = true;
    }
    try steps.append(a, .{ .seq = &seq2, .start = 0, .n = pass, .layers = 1, .stop_after = true });
    try steps.append(a, .{ .seq = &seq, .start = 64, .n = pass, .layers = 2, .stop_after = true });
    // each step's expected rows of each layer (the reference read on this thread: the pool is the test's)
    var wants: std.ArrayList([]u16) = .empty;
    defer {
        for (wants.items) |x| a.free(x);
        wants.deinit(a);
    }
    for (steps.items) |st| for (0..2) |j| try wants.append(a, try Ref.rowsOf(a, h, &tables, pool, st.seq[0 .. st.start + st.n], st.start, st.n, j));
    const Drive = struct {
        fn run(pf: *Prefetch, sts: []const Step, ws: []const []u16, bad: *usize) void {
            for (sts, 0..) |st, i| {
                pf.begin(st.seq, st.start, st.n, replay);
                for (0..st.layers) |j| {
                    const got = (pf.take(j, st.start, st.n) catch {
                        bad.* += 1;
                        continue;
                    }) orelse {
                        bad.* += 1;
                        continue;
                    };
                    if (!std.mem.eql(u16, ws[2 * i + j], got[0..ws[2 * i + j].len])) bad.* += 1;
                }
                if (st.stop_after) pf.stop();
            }
        }
    };
    const port0: u16 = 32700 + @as(u16, @intCast(@mod(std.Io.Timestamp.now(io, .awake).nanoseconds, 300))) * 8;
    for ([_][]const u8{ "1", "0.5", "0.75", "0.333", "0.001" }, 0..) |share, it| for ([_][]const u8{ "1", "2,scales", "prompt" }) |val| {
        const split = try parseSplit(share);
        const set = try parse(val);
        const pa = Prefetch.init(a, io, &c, h, &tables, lo, hi, pass, chunk, set, 64) catch |e| switch (e) {
            error.AioUnavailable => return error.SkipZigTest,
            else => return e,
        };
        defer pa.deinit();
        const pb = try Prefetch.init(a, io, &c, h, &tables, lo, hi, pass, chunk, set, 64);
        defer pb.deinit();
        // the pair's link on the loopback
        const Acc = struct {
            fn run(p: u16, out: *?link.Link) void {
                out.* = link.Link.listen(.{ 127, 0, 0, 1 }, p) catch null;
            }
        };
        const port = port0 + @as(u16, @intCast(it));
        var acc: ?link.Link = null;
        const th = try std.Thread.spawn(.{}, Acc.run, .{ port, &acc });
        const cl = link.Link.connect(io, .{ 127, 0, 0, 1 }, port, 10) catch |e| {
            th.join();
            return e;
        };
        th.join();
        const la = acc orelse return error.TestLinkFailed;
        // parts of no chunk the partner wants (another prompt's ids at the first chunk's place, then a run of them
        // past the mailbox's size): never taken, evicted oldest first
        {
            const n0 = steps.items[0].n;
            const own0 = split.own(n0, true);
            if (own0[1] > own0[0]) for (0..box_n + 2) |_| {
                const cnt = own0[1] - own0[0];
                const hd: Hdr = .{ .layer = 0, .start = 0, .n = n0, .lo = own0[0], .cnt = cnt, .k = k, .row_w = 64 };
                try writeFd(la.fd, std.mem.asBytes(&hd));
                const junk_ids = try a.alloc(i64, cnt * k);
                defer a.free(junk_ids);
                @memset(junk_ids, 79);
                try writeFd(la.fd, std.mem.sliceAsBytes(junk_ids));
                const junk = try a.alloc(u16, cnt * k * 64);
                defer a.free(junk);
                @memset(junk, 0xdead);
                try writeFd(la.fd, std.mem.sliceAsBytes(junk));
            };
        }
        try pa.usePair(la.fd, split, true);
        try pb.usePair(cl.fd, split, false);
        try std.testing.expect(pa.pair != null and pb.pair != null);
        var bad_a: usize = 0;
        var bad_b: usize = 0;
        const ta = try std.Thread.spawn(.{}, Drive.run, .{ pa, steps.items, wants.items, &bad_a });
        Drive.run(pb, steps.items, wants.items, &bad_b);
        ta.join();
        if (bad_a != 0 or bad_b != 0) std.debug.print("split {s} prefetch {s}: {d} / {d} layers' rows wrong\n", .{ share, val, bad_a, bad_b });
        try std.testing.expectEqual(@as(usize, 0), bad_a);
        try std.testing.expectEqual(@as(usize, 0), bad_b);
        // the parts did cross: the partner of the rank that reads nothing (share 1: the second; 0.001: the first, its
        // part of these short chunks empty) received every row; the junk parts arrived and were never taken
        pa.mutex.lockUncancelable(io);
        const got_a = pa.pair.?.arrived;
        pa.mutex.unlock(io);
        pb.mutex.lockUncancelable(io);
        const got_b = pb.pair.?.arrived;
        pb.mutex.unlock(io);
        try std.testing.expect(got_a + got_b > 0);
        if (split.milli == 1000) try std.testing.expectEqual(@as(u64, 0), got_a);
        if (split.milli == 1) try std.testing.expectEqual(@as(u64, 0), got_b);
        if (split.milli != 1) try std.testing.expect(got_b > box_n + 2); // (the junk and the real parts)
        if (split.milli < 1000) try std.testing.expect(got_a > 0);
    };
}

test "a follower's plan dropped inside a long prompt (stage ZE4's hang): its partner's parts already taken, the rows still exact" {
    // ZE4's S100 hung on a 156,941-token prompt: rank 0 sent a snap_drop inside the fill (a save evicting a kept
    // state); the followers stopped their readers there (rank 0 does not), restarted their plans at the last chunk and
    // waited for rank 0's part of it, which rank 0 had sent (and they had taken) before. lanes.stopsAhead now keeps the
    // followers' plans on snap_drop with the split; here the guard under it: one partner (b, the follower) drops its
    // plan after its reader has read the whole prompt, the other (a, rank 0) does not; b's rows of the chunks after
    // must still be exact, and come from its own reads after the wait instead of a hang.
    if (@import("builtin").os.tag != .linux) return error.SkipZigTest;
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    const rows = 80;
    var json_buf: [1024]u8 = undefined;
    const tw = rows * 64;
    const ts = rows * 2;
    const json = try std.fmt.bufPrint(&json_buf, "{{\"layers.1.engram.embed.weight\":{{\"dtype\":\"F8_E4M3\",\"shape\":[{d},64],\"data_offsets\":[0,{d}]}},\"layers.1.engram.embed.scale\":{{\"dtype\":\"F8_E8M0\",\"shape\":[{d},2],\"data_offsets\":[{d},{d}]}},\"layers.14.engram.embed.weight\":{{\"dtype\":\"F8_E4M3\",\"shape\":[{d},64],\"data_offsets\":[{d},{d}]}},\"layers.14.engram.embed.scale\":{{\"dtype\":\"F8_E8M0\",\"shape\":[{d},2],\"data_offsets\":[{d},{d}]}}}}", .{ rows, tw, rows, tw, tw + ts, rows, tw + ts, 2 * tw + ts, rows, 2 * tw + ts, 2 * tw + 2 * ts });
    var file: std.ArrayList(u8) = .empty;
    defer file.deinit(a);
    var len8: [8]u8 = undefined;
    std.mem.writeInt(u64, &len8, json.len, .little);
    try file.appendSlice(a, &len8);
    try file.appendSlice(a, json);
    for (0..2 * tw + 2 * ts) |i| try file.append(a, @intCast((i * 29 + 11) % 251));
    try tmp.dir.writeFile(io, .{ .sub_path = "engram.safetensors", .data = file.items });
    const dir = try std.fs.path.join(a, &.{ ".zig-cache", "tmp", &tmp.sub_path });
    defer a.free(dir);
    var tables = try engram_io.Tables.open(a, io, dir);
    defer tables.close();
    var why: @import("config.zig").Why = .{};
    var c = try @import("config.zig").parse(a, @import("config.zig").test_config, &why);
    c.engram_vocab = 10;
    c.engram_heads = 1;
    c.engram_pad = 0;
    const map = [_]i32{ 7, 1, 2, 3, 4, 5 };
    const h = engram.Hasher.init(c, &map, .{ .{ 3, 5, 7, 9 }, .{ 11, 13, 15, 17 } });
    const pool = try engram_io.Pool.init(a, io, 2);
    defer pool.deinit(a);
    // a long prompt: 37 chunks of passes of 32 rows (two 16-row chunks) up to the replay, then 16-row chunks
    var prng = std.Random.DefaultPrng.init(31);
    var seq: [1170]i32 = undefined;
    for (&seq) |*x| x.* = prng.random().intRangeLessThan(i32, 0, 6);
    const lo = 1;
    const hi = 3;
    const k = hi - lo;
    const pass = 32;
    const chunk = 16;
    const replay = 1100;
    var plan: std.ArrayList([2]usize) = .empty;
    defer plan.deinit(a);
    {
        var start: usize = 0;
        while (start < seq.len) {
            const n = nextRows(pass, chunk, start, seq.len, replay);
            try plan.append(a, .{ start, n });
            start += n;
        }
    }
    try std.testing.expect(plan.items.len >= 37);
    var wants: std.ArrayList([]u16) = .empty;
    defer {
        for (wants.items) |x| a.free(x);
        wants.deinit(a);
    }
    for (plan.items) |ch| for (0..2) |j| {
        const t = tables.layers.get(if (j == 0) 1 else 14).?;
        const n = ch[1];
        const cols = h.cols();
        const hh = try a.alloc(i64, n * h.layers * cols);
        defer a.free(hh);
        h.hashes(seq[0 .. ch[0] + n], ch[0], n, hh);
        const flat = try a.alloc(i64, n * k);
        defer a.free(flat);
        for (0..n) |r| for (0..k) |q| {
            flat[r * k + q] = hh[(r * h.layers + j) * cols + lo + q];
        };
        const w = try a.alloc(u8, n * k * t.row_w);
        defer a.free(w);
        const sc = try a.alloc(u8, n * k * t.row_s);
        defer a.free(sc);
        try pool.gather(t, flat, w, sc);
        const out = try a.alloc(u16, n * k * t.row_w);
        engram.decodeRows(w, sc, out, n * k, t.row_w, t.row_s, 8);
        try wants.append(a, out);
    };
    const Drive = struct {
        // the fill's chunks; `drop_before`: stop (the follower's snap_drop) before that chunk, once its reader has
        // read the whole prompt (so the partner's parts of the chunks after were all taken already)
        fn run(pf: *Prefetch, sq: []const i32, pl: []const [2]usize, ws: []const []u16, drop_before: ?usize, bad: *usize) void {
            for (pl, 0..) |ch, i| {
                if (drop_before != null and drop_before.? == i) {
                    while (pf.stats().chunks < pl.len) std.Io.sleep(std.testing.io, .fromMilliseconds(2), .awake) catch {};
                    pf.stop();
                }
                pf.begin(sq, ch[0], ch[1], replay);
                for (0..2) |j| {
                    const got = (pf.take(j, ch[0], ch[1]) catch {
                        bad.* += 1;
                        continue;
                    }) orelse {
                        bad.* += 1;
                        continue;
                    };
                    if (!std.mem.eql(u16, ws[2 * i + j], got[0..ws[2 * i + j].len])) bad.* += 1;
                }
            }
        }
    };
    const port0: u16 = 36000 + @as(u16, @intCast(@mod(std.Io.Timestamp.now(io, .awake).nanoseconds, 300))) * 8;
    for ([_][]const u8{ "1", "0.5" }, 0..) |share, it| for ([_][]const u8{ "prompt,scales", "prompt" }, 0..) |val, iv| {
        const split = try parseSplit(share);
        const set = try parse(val);
        const pa = Prefetch.init(a, io, &c, h, &tables, lo, hi, pass, chunk, set, 64) catch |e| switch (e) {
            error.AioUnavailable => return error.SkipZigTest,
            else => return e,
        };
        defer pa.deinit();
        const pb = try Prefetch.init(a, io, &c, h, &tables, lo, hi, pass, chunk, set, 64);
        defer pb.deinit();
        const Acc = struct {
            fn run(p: u16, out: *?link.Link) void {
                out.* = link.Link.listen(.{ 127, 0, 0, 1 }, p) catch null;
            }
        };
        const port = port0 + @as(u16, @intCast(2 * it + iv));
        var acc: ?link.Link = null;
        const th = try std.Thread.spawn(.{}, Acc.run, .{ port, &acc });
        const cl = link.Link.connect(io, .{ 127, 0, 0, 1 }, port, 10) catch |e| {
            th.join();
            return e;
        };
        th.join();
        const la = acc orelse return error.TestLinkFailed;
        try pa.usePair(la.fd, split, true);
        try pb.usePair(cl.fd, split, false);
        pa.pair.?.wait_ns = 300 * std.time.ns_per_ms;
        pb.pair.?.wait_ns = 300 * std.time.ns_per_ms;
        var bad_a: usize = 0;
        var bad_b: usize = 0;
        const last = plan.items.len - 1;
        const ta = try std.Thread.spawn(.{}, Drive.run, .{ pa, &seq, plan.items, wants.items, @as(?usize, null), &bad_a });
        Drive.run(pb, &seq, plan.items, wants.items, last, &bad_b);
        ta.join();
        try std.testing.expectEqual(@as(usize, 0), bad_a);
        try std.testing.expectEqual(@as(usize, 0), bad_b);
        // b read the last chunk's partner rows itself, after the wait (both layers): the parts it needed were taken
        // before the drop and a's plan never re-sends them
        const sb = pb.stats();
        try std.testing.expectEqual(@as(u64, 2), sb.fallbacks);
        try std.testing.expectEqual(@as(u64, 0), pa.stats().fallbacks);
        pa.stop();
        pb.stop();
    };
}
