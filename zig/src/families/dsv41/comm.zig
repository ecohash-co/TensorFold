//! The TP ranks' collectives: NCCL opened at run time, its unique id passed from rank 0 over the link. Every model
//! reduction is an all-gather of fp32 partials that each rank then adds in rank order (DESIGN.md section 3), so the
//! ranks hold the same bits without a broadcast.
const std = @import("std");
const cuda = @import("cuda");
const Link = @import("link.zig").Link;

const nccl = cuda.nccl;

/// Whether an NCCL operation on stream `cur` waits for the last one's end (on stream `last`; 0: none yet): only when
/// that went on the other stream (on one stream the stream's order already holds them in issue order).
pub fn crossWait(last: usize, cur: usize) bool {
    return last != 0 and last != cur;
}

/// One launch order for one communicator's NCCL operations on two streams of this GPU (TF_DS_PROMPT_OVL, ovl.Side:
/// some of a chunk's exchanges go on a comm stream beside the compute stream): every operation starts after the one
/// issued before it ends, whichever stream either went on. The ranks issue the same operations in the same order, so
/// each GPU runs them one at a time in that order, exactly as when they all went on one stream. (Ported from stage
/// ZP3's two-communicator fix; here there is one communicator, so this is the whole ordering argument.)
pub const Order = struct {
    ev: cuda.Event, // recorded after each operation, on its stream
    last: usize = 0, // the stream (its handle's address) the last operation went on; 0: none since the chunk began
    waits: usize = 0, // cross-stream waits issued

    fn key(s: cuda.Stream) usize {
        return @intFromPtr(s.handle);
    }

    /// Before an NCCL operation on `stream`.
    pub fn before(o: *Order, stream: cuda.Stream) !void {
        if (crossWait(o.last, key(stream))) {
            try stream.wait(o.ev);
            o.waits += 1;
        }
    }

    /// After it (issued).
    pub fn after(o: *Order, stream: cuda.Stream) !void {
        try o.ev.record(stream);
        o.last = key(stream);
    }
};

pub const Comm = struct {
    lib: nccl.Library,
    comm: nccl.Comm,
    rank: u32,
    world: u32,
    // TF_DS_PROMPT_OVL's SP chunks: the launch order kept between the compute stream and the comm stream (set for the
    // chunk only; null: issued as they come, as served)
    order: ?*Order = null,

    /// Rank 0 makes the id and sends it; the others take it from the link. Needs the CUDA context current.
    pub fn init(link: Link, rank: u32, world: u32) !Comm {
        var lib = try nccl.Library.open();
        errdefer lib.close();
        var id: nccl.UniqueId = undefined;
        if (rank == 0) {
            try lib.check(lib.api.ncclGetUniqueId(&id), "ncclGetUniqueId");
            try link.send(std.mem.asBytes(&id));
        } else {
            var buf: [@sizeOf(nccl.UniqueId)]u8 = undefined;
            const got = try link.recv(&buf);
            if (got.len != buf.len) return error.BadUniqueId;
            id = std.mem.bytesToValue(nccl.UniqueId, got);
        }
        var comm: nccl.Comm = null;
        try lib.check(lib.api.ncclCommInitRank(&comm, @intCast(world), id, @intCast(rank)), "ncclCommInitRank");
        return .{ .lib = lib, .comm = comm, .rank = rank, .world = world };
    }

    pub fn deinit(c: *Comm) void {
        _ = c.lib.api.ncclCommDestroy(c.comm);
        c.lib.close();
        c.* = undefined;
    }

    /// `count` elements of `dtype` from every rank into `recv`, rank after rank.
    pub fn allGather(c: Comm, send: u64, recv: u64, count: usize, dtype: nccl.DataType, stream: cuda.Stream) !void {
        if (c.order) |o| try o.before(stream);
        try c.lib.check(c.lib.api.ncclAllGather(send, recv, count, dtype, c.comm, stream.handle), "ncclAllGather");
        if (c.order) |o| try o.after(stream);
    }

    /// `count` elements of `dtype` to rank `peer` and `rcount` from it, in one NCCL group (point to point).
    pub fn exchange(c: Comm, send: u64, count: usize, recv: u64, rcount: usize, dtype: nccl.DataType, peer: u32, stream: cuda.Stream) !void {
        if (c.order) |o| try o.before(stream);
        try c.lib.check(c.lib.api.ncclGroupStart(), "ncclGroupStart");
        const s = c.lib.api.ncclSend(send, count, dtype, @intCast(peer), c.comm, stream.handle);
        const r = c.lib.api.ncclRecv(recv, rcount, dtype, @intCast(peer), c.comm, stream.handle);
        try c.lib.check(c.lib.api.ncclGroupEnd(), "ncclGroupEnd");
        try c.lib.check(s, "ncclSend");
        try c.lib.check(r, "ncclRecv");
        if (c.order) |o| try o.after(stream);
    }

    /// One side of a point-to-point transfer: `count` elements at `ptr` to or from rank `peer`.
    pub const P2p = struct { ptr: u64, count: usize, peer: u32 };

    /// Every send and receive in one NCCL group (all-to-all-v by point to point): the same bytes as each pair's
    /// exchange, in one launch. Each rank must post the matching receive of every send.
    pub fn sendRecv(c: Comm, sends: []const P2p, recvs: []const P2p, dtype: nccl.DataType, stream: cuda.Stream) !void {
        if (c.order) |o| try o.before(stream);
        try c.lib.check(c.lib.api.ncclGroupStart(), "ncclGroupStart");
        var first: nccl.Result = 0;
        for (sends) |x| {
            const s = c.lib.api.ncclSend(x.ptr, x.count, dtype, @intCast(x.peer), c.comm, stream.handle);
            if (first == 0) first = s;
        }
        for (recvs) |x| {
            const r = c.lib.api.ncclRecv(x.ptr, x.count, dtype, @intCast(x.peer), c.comm, stream.handle);
            if (first == 0) first = r;
        }
        try c.lib.check(c.lib.api.ncclGroupEnd(), "ncclGroupEnd");
        try c.lib.check(first, "ncclSend/ncclRecv");
        if (c.order) |o| try o.after(stream);
    }
};

test "comm.Order: a cross-stream wait only when the stream changes, none for a chunk's first operation" {
    try std.testing.expect(!crossWait(0, 1));
    try std.testing.expect(!crossWait(1, 1));
    try std.testing.expect(crossWait(1, 2));
    try std.testing.expect(crossWait(2, 1));
}
