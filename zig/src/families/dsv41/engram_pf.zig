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
const std = @import("std");
const Config = @import("config.zig").Config;
const engram = @import("engram.zig");
const engram_io = @import("engram_io.zig");
const engram_aio = @import("engram_aio.zig");
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
        p.mutex.unlock(p.io);
        if (p.thread) |th| th.join();
        p.aio.deinit();
        p.freeBuffers();
        p.gpa.destroy(p);
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
            p.mutex.unlock(p.io);
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
    /// this reader's ring and scratch.
    fn readLayer(p: *Prefetch, seq: []const i32, start: usize, n: usize, j: usize, out: []u16) !void {
        const t = p.tabs[j];
        const cols = p.hasher.cols();
        const k = p.k;
        if (n * k * t.row_w > out.len) return error.BadChunk;
        if (j == 0) p.hasher.hashes(seq, start, n, p.hashes[0 .. n * p.hasher.layers * cols]);
        for (0..n) |r| {
            for (0..k) |q| p.flat[r * k + q] = p.hashes[(r * p.hasher.layers + j) * cols + p.lo + q];
        }
        const m = n * k;
        const w = p.w[0 .. m * t.row_w];
        const s = p.s[0 .. m * t.row_s];
        if (p.ram[j]) |ram| {
            try p.gather(t, p.flat[0..m], w, null);
            const sr = ram.rows();
            for (0..m) |i| {
                const id: usize = @intCast(p.flat[i]);
                if (id >= t.rows) return error.RowOutOfRange;
                @memcpy(s[i * t.row_s ..][0..t.row_s], sr[id * t.row_s ..][0..t.row_s]);
            }
        } else try p.gather(t, p.flat[0..m], w, s);
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
