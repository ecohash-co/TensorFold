//! The exact 2D-TP4 split on the prompt path (DESIGN-zig-tp4.md sections 3-4; model.py Comm2D, wo_2d, moe_2d and
//! exl3/experts2d.py on four nodes): node g = 2 p + r is TP2 rank r of pair p. A node computes every output column it
//! holds whole, over its TP2 slice's K in TP2's order, and every exchange only moves bytes, so after an exchange a
//! node holds TP2 rank r's bytes: `quarters` assembles TP2's [2, n, N] gather of the two ranks' partials (which the
//! mixes then add in rank order, as on TP2), `catRank` the rank's whole wo_a output or expert intermediate from the
//! two pairs' parts. Prompt chunks exchange over NCCL across the four (the RDMA rings drop prompt-size bursts on the
//! switch: slots P6, P7b), each as one all-gather of every node's part then copies into place: the bytes Comm2D's
//! NCCL paths (gather4, _p2p) put there. With RDMA rings (Two.rings, ring2d.zig) the exchanges the lane sends over
//! them take them, written straight into place: quarters of fewer than 64 rows (decode windows; TF_DS_2D_PROMPT_NCCL
//! keeps prompt chunks on NCCL) and every cat_rank its column pair's ring takes; the same bytes either way.
//!
//! prompt.zig takes it through thin hooks behind Engine.two; a TP2 engine has none and runs as before. The decode
//! rounds' part is round2d.zig's, on this node state.
const std = @import("std");
const cuda = @import("cuda");
const Config = @import("config.zig").Config;
const plan = @import("plan.zig");
const prompt = @import("prompt.zig");
const weights = @import("weights.zig");
const exl3_experts = @import("exl3_experts.zig");
const exl3_experts2d = @import("exl3_experts2d.zig");
const exl3_prompt = @import("exl3_prompt.zig");
const ring2d = @import("ring2d.zig");
const rdma_mod = @import("rdma.zig");
const round2d = @import("round2d.zig");
const round_rows = @import("round.zig").max_rows;
const Link = @import("link.zig").Link;
const Comm = @import("comm.zig").Comm;
const ovl = @import("ovl.zig");

const nccl = cuda.nccl;

/// The split of node `g` of `world` nodes: TP over 1 or 2, the 2D split over 4.
pub fn split(g: u32, world: u32) !plan.Split {
    return switch (world) {
        1, 2 => .{ .rank = g, .world = world },
        4 => .{ .rank = g % 2, .world = 2, .pair = g / 2 },
        else => error.UnsupportedWorld,
    };
}

/// The links of a run and its NCCL world: rank 0 listens on PORT + g - 1 for follower g, one follower at a time (each
/// follower retries its connect while rank 0 gets to it), so two ranks link exactly as before (rank 1 on PORT).
/// NCCL's unique id goes from rank 0 down every link. (The port's link to N followers replaces this.)
pub const Fan = struct {
    links: [3]?Link = .{ null, null, null },

    pub fn open(io: std.Io, ip: [4]u8, port: u16, rank: u32, world: u32) !Fan {
        var f: Fan = .{};
        errdefer f.close();
        if (world > 4) return error.UnsupportedWorld;
        if (world < 2) return f;
        if (rank == 0) {
            for (1..world) |g| f.links[g - 1] = try Link.listen(ip, port + @as(u16, @intCast(g - 1)));
        } else {
            f.links[0] = try Link.connect(io, ip, port + @as(u16, @intCast(rank - 1)), 300);
        }
        return f;
    }

    pub fn close(f: *Fan) void {
        for (&f.links) |*l| {
            if (l.*) |*x| x.close();
            l.* = null;
        }
    }

    /// Comm.init over every link: rank 0 makes the id and sends it to each follower. Needs the CUDA context current.
    pub fn comm(f: *const Fan, rank: u32, world: u32) !Comm {
        var lib = try nccl.Library.open();
        errdefer lib.close();
        var id: nccl.UniqueId = undefined;
        if (world > 1) {
            if (rank == 0) {
                try lib.check(lib.api.ncclGetUniqueId(&id), "ncclGetUniqueId");
                for (f.links) |l| if (l) |x| try x.send(std.mem.asBytes(&id));
            } else {
                var buf: [@sizeOf(nccl.UniqueId)]u8 = undefined;
                const got = try f.links[0].?.recv(&buf);
                if (got.len != buf.len) return error.BadUniqueId;
                id = std.mem.bytesToValue(nccl.UniqueId, got);
            }
        } else try lib.check(lib.api.ncclGetUniqueId(&id), "ncclGetUniqueId");
        var c: nccl.Comm = null;
        try lib.check(lib.api.ncclCommInitRank(&c, @intCast(world), id, @intCast(rank)), "ncclCommInitRank");
        return .{ .lib = lib, .comm = c, .rank = rank, .world = world };
    }
};

fn len(r: ?[2]usize) usize {
    const x = r orelse return 0;
    return x[1] - x[0];
}

/// A 2D node's part of the prompt path and the decode rounds: the pairs' widths of each split output and the exchange
/// buffers.
pub const Two = struct {
    r: usize,
    p: usize,
    heads: usize, // the heads here (its pair's half of the rank's)
    uw: [2]usize, // wo_a's output a pair (its groups x o_lora): pair 0's, pair 1's
    ow: [2]usize, // wo_b's output columns a pair
    gu: [2]usize, // gate / up's columns a pair
    dw: [2]usize, // down's output columns a pair
    ew: [2]usize, // Engram wkv's output columns a pair (0 without an Engram layer)
    hw: [2]usize, // the head's vocabulary columns a pair (253 / 252 blocks of a TP2 half)
    parts: u64, // [4, rows, widest] where the all-gathers land
    pad: u64, // this node's part at the wider pair's width (when the pairs' widths differ)
    part_bytes: usize, // the largest part a node sends
    u_full: u64, // bf16 [cap, uw0 + uw1]: the rank's wo_a output, pair 0's groups then pair 1's
    xd_full: u64, // fp16 [cap * slots, gu0 + gu1]: the rank's expert intermediate, pair 0's blocks then pair 1's
    own: u64, // fp16 [round rows, uw[p]]: a round's half of wo_b's rotated input rows (wo_a's epilogue, round2d.woRot)
    rings: ?*const ring2d.Rings, // the RDMA rings (TF_DS_2D_INTO, TF_DS_2D_GROUPS), when the caller opened them
    // TF_DS_2D_GU=parity: the rest block's width (0: "first"), the rest launches' scratch for a decode window and for a
    // prompt chunk, and the column partner's pack (its main rows, then its rest rows) as the exchange delivers it
    ir: usize = 0,
    rest_rows: usize = 0, // the prompt chunks' rows the rest scratch holds (init's cap)
    rest_dec: ?exl3_experts2d.RestScratch = null,
    rest_chunk: ?exl3_experts2d.RestScratch = null,
    peer: u64 = 0,
    // TF_DS_PROMPT_SP=rows (model.spRows): catRank's NCCL path as a send / recv with the column partner (the only part
    // it uses) instead of an all-gather over the four; the same bytes in place
    p2p_cat: bool = false,
    // TF_DS_PROMPT_ENGRAM=sp (model.promptEngram; SP chunks only): Engram's sum for the rows this node owns alone
    // (engramSumSp), the only rows its gate reads
    engram_sp: bool = false,
    // TF_DS_PROMPT_A2A=shift (model.promptA2a): toOwners as three shift steps (send to g + k, receive from g - k), so
    // no node receives from two peers at once; the same transfers
    a2a_shift: bool = false,

    /// The widths from the split (both pairs'), and the buffers for chunks of up to `cap` rows (and rounds) from `a`.
    pub fn init(c: *const Config, w: *const weights.Weights, a: *prompt.Arena, s: plan.Split, cap: usize) !Two {
        const pp = s.pair orelse return error.NotA2DSplit;
        const s0 = s.atPair(0);
        const s1 = s.atPair(1);
        const x0 = s0.expertParts(c.*);
        const x1 = s1.expertParts(c.*);
        var t: Two = undefined;
        t.r = s.rank;
        t.p = pp;
        t.heads = s.headSpan(c.*)[1];
        t.uw = .{ s0.woAGroups(c.*)[1] * c.o_lora, s1.woAGroups(c.*)[1] * c.o_lora };
        t.ow = .{ len(s0.woBCols(c.*)), len(s1.woBCols(c.*)) };
        t.gu = .{ len(x0.gu), len(x1.gu) };
        t.dw = .{ len(x0.dcols), len(x1.dcols) };
        var engram = false;
        for (w.layers) |lay| engram = engram or lay.engram_wkv != null;
        t.ew = if (engram) .{ len(s0.engramCols(c.*)), len(s1.engramCols(c.*)) } else .{ 0, 0 };
        t.hw = .{ len(s0.headCols(c.*)), len(s1.headCols(c.*)) };
        // the weights loaded here are this split's
        const l0 = w.layers[0];
        if (l0.wq_b.n != t.heads * c.head_dim or l0.groups * c.o_lora != t.uw[pp] or l0.wo_b.n != t.ow[pp] or
            l0.experts.width != t.gu[pp] or l0.experts.down_n != t.dw[pp]) return error.WeightsNotThisSplit;
        // prompt.gather assembles either sublayer's partials: both splits must give the same widths
        if (t.ow[0] != t.dw[0] or t.ow[1] != t.dw[1]) return error.UnequalPartialSplits;
        const slots = c.top_k + 1;
        const biggest = @max(@max(@max(cap * @max(t.ow[0], t.ow[1]) * 4, cap * @max(t.ew[0], t.ew[1]) * 4), @max(cap * @max(t.uw[0], t.uw[1]) * 2, cap * slots * @max(t.gu[0], t.gu[1]) * 2)), @max(t.hw[0], t.hw[1]) * 4);
        t.part_bytes = biggest;
        t.parts = try a.take(4 * biggest);
        t.pad = try a.take(biggest);
        t.u_full = try a.take(cap * (t.uw[0] + t.uw[1]) * 2);
        // parity: the rest block after both pairs' main blocks in the assembled intermediate
        const ex0 = l0.experts;
        t.ir = if (ex0.rest) |r| r.width else 0;
        t.rest_rows = cap;
        t.rest_dec = null;
        t.rest_chunk = null;
        t.peer = 0;
        t.xd_full = try a.take(@max(cap, round_rows) * slots * (t.gu[0] + t.gu[1] + t.ir) * 2);
        if (t.ir != 0) {
            if (t.gu[0] != t.gu[1]) return error.UnequalParityParts;
            t.rest_dec = try restScratch(a, ex0, exl3_experts.exact_rows, slots, false);
            t.rest_chunk = try restScratch(a, ex0, cap, slots, true);
            t.peer = try a.take(@max(cap, round_rows) * slots * (t.gu[t.p] + t.ir) * 2);
        }
        t.own = try a.take(round_rows * t.uw[pp] * 2);
        t.rings = null;
        t.p2p_cat = false;
        t.engram_sp = false;
        t.a2a_shift = false;
        return t;
    }

    fn restScratch(a: *prompt.Arena, ex: weights.Experts, rows: usize, slots: usize, chunk: bool) !exl3_experts2d.RestScratch {
        const sz = try exl3_experts2d.RestScratch.sizes(rows, slots, ex, chunk);
        var rs: exl3_experts2d.RestScratch = undefined;
        inline for (.{ "ids", "count", "counts", "members", "cnt_gu", "z", "work_gu", "work_d", "rpick" }, 0..) |f, j| @field(rs, f) = try a.take(sz[j]);
        return rs;
    }

    /// The rest scratch's first use: members -1, the counters and partials zero (as the main scratch starts).
    pub fn clearRest(t: *const Two, e: *const prompt.Engine) !void {
        for ([_]?exl3_experts2d.RestScratch{ t.rest_dec, t.rest_chunk }, [_]bool{ false, true }) |o, chunk| {
            const rs = o orelse continue;
            const ex = e.w.layers[0].experts;
            const sz = try exl3_experts2d.RestScratch.sizes(if (chunk) t.rest_rows else exl3_experts.exact_rows, e.slots(), ex, chunk);
            try prompt.fill32(e, rs.members, 0xffffffff, sz[3] / 4);
            for ([_]u64{ rs.ids, rs.count, rs.counts, rs.cnt_gu, rs.z }, [_]usize{ sz[0], sz[1], sz[2], sz[4], sz[5] }) |ptr, n| try prompt.fill(e, ptr, 0, n);
        }
    }

    /// The column partner's pack (a parity split's main rows, then its rest rows: `bytes` each way) into t.peer: over
    /// the column ring when the caller opened the rings and it fits, else NCCL send / recv with the partner.
    pub fn packExchange(t: *const Two, e: *const prompt.Engine, mine: u64, rows: usize, width: usize) !void {
        if (t.rings) |rs| {
            var dsts: [2]rdma_mod.Ring.Dst = .{ .{}, .{} };
            dsts[1 - t.p] = .{ .ptr = t.peer, .row4 = @intCast(width * 2 / 16), .stride4 = width * 2 / 16 };
            if (rs.col.gatherInto(e.s, mine, @intCast(rows), @intCast(width * 2 / 16), &dsts)) |_| return else |err| switch (err) {
                error.TooLarge => {},
                else => return err,
            }
        }
        const g: u32 = @intCast(t.r + 2 * t.p);
        try e.comm.exchange(mine, rows * width, t.peer, rows * width, .f16, g ^ 2, e.s);
    }

    /// moe_2d's parity experts after the main gate / up and the rest's: the packs exchanged with the column partner,
    /// the half's intermediate assembled into t.xd_full (pair 0's main, pair 1's main, each slot's rest from its
    /// owner), for down.
    pub fn assembleParity(t: *const Two, e: *const prompt.Engine, ex: weights.Experts, xd: u64, pick: u64, p: usize) !void {
        const rest = ex.rest orelse return error.NoRest;
        const ig = t.gu[t.p];
        try t.packExchange(e, xd, p, ig + t.ir);
        try prompt.pm(e, .comm);
        const part0 = if (t.p == 0) xd else t.peer;
        const part1 = if (t.p == 1) xd else t.peer;
        try e.ops.assembleRest(e.s, part0, part1, pick, rest.owner, t.xd_full, p, ig, t.ir);
    }

    /// Every node's part over NCCL into `parts` [4, n, wmax] (bytes), this node's `src` [n, w[p]] padded to wmax.
    fn allParts(t: *const Two, e: *const prompt.Engine, src: u64, n: usize, w: [2]usize, esize: usize) !usize {
        const wmax = @max(w[0], w[1]);
        if (n * wmax * esize > t.part_bytes) return error.PartTooBig;
        var send = src;
        if (w[t.p] != wmax) {
            try e.ops.copyRows(e.s, src, w[t.p] * esize, t.pad, wmax * esize, w[t.p] * esize, n);
            send = t.pad;
        }
        try e.comm.allGather(send, t.parts, n * wmax * esize, .u8, e.s);
        return wmax;
    }

    /// Whether a ring took the exchange: false when there is none or the part does not fit it (nothing sent then).
    fn ringed(r: anyerror!void) !bool {
        r catch |err| switch (err) {
            error.TooLarge, error.NotFloat4 => return false,
            else => return err,
        };
        return true;
    }

    /// quarters (Comm2D.quarters): `src` [n, w[p]] (this node's columns of its TP2 rank's partial) -> `dst`
    /// [2, n, w0 + w1]: rank r's partial, pair 0's columns then pair 1's, for every r.
    pub fn quarters(t: *const Two, e: *const prompt.Engine, src: u64, dst: u64, n: usize, w: [2]usize, esize: usize) !void {
        if (t.rings) |rs| if (n < 64 and try ringed(rs.quarters(e.s, src, dst, n, w, esize))) return;
        const wmax = try t.allParts(e, src, n, w, esize);
        const row = (w[0] + w[1]) * esize;
        for (0..4) |g| {
            const rr = g % 2;
            const pg = g / 2;
            try e.ops.copyRows(e.s, t.parts + g * n * wmax * esize, wmax * esize, dst + rr * n * row + pg * w[0] * esize, row, w[pg] * esize, n);
        }
    }

    /// cat_rank (Comm2D.cat_rank): `src` [n, w[p]] -> `dst` [n, w0 + w1]: this node's TP2 rank's whole width, pair 0's
    /// part then pair 1's (from its column partner, the same rank of the other pair).
    pub fn catRank(t: *const Two, e: *const prompt.Engine, src: u64, dst: u64, n: usize, w: [2]usize, esize: usize) !void {
        if (t.rings) |rs| if (try ringed(rs.catRank(e.s, src, dst, n, w, esize))) return;
        if (t.p2p_cat) return t.catRankP2p(e, src, dst, n, w, esize);
        const wmax = try t.allParts(e, src, n, w, esize);
        const row = (w[0] + w[1]) * esize;
        for (0..2) |pg| {
            const g = t.r + 2 * pg;
            try e.ops.copyRows(e.s, t.parts + g * n * wmax * esize, wmax * esize, dst + pg * w[0] * esize, row, w[pg] * esize, n);
        }
    }

    /// catRank over NCCL send / recv with the column partner g ^ 2 (TF_DS_PROMPT_SP=rows): the all-gather's other two
    /// parts are never read, so only the partner's part moves (into `parts`); then this node's part and the partner's
    /// go into their columns of dst, the bytes the all-gather path puts there.
    fn catRankP2p(t: *const Two, e: *const prompt.Engine, src: u64, dst: u64, n: usize, w: [2]usize, esize: usize) !void {
        const q = 1 - t.p;
        if (n * w[q] * esize > t.part_bytes) return error.PartTooBig;
        const g: u32 = @intCast(t.r + 2 * t.p);
        try e.comm.exchange(src, n * w[t.p] * esize, t.parts, n * w[q] * esize, .u8, g ^ 2, e.s);
        const row = (w[0] + w[1]) * esize;
        try e.ops.copyRows(e.s, src, w[t.p] * esize, dst + t.p * w[0] * esize, row, w[t.p] * esize, n);
        try e.ops.copyRows(e.s, t.parts, w[q] * esize, dst + q * w[0] * esize, row, w[q] * esize, n);
    }

    /// SP rows (TF_DS_PROMPT_SP=rows / rows2): quarters for the rows this node owns. Of a chunk of o.n rows node
    /// g = r + 2 p owns rows [o.r0(g), o.r0(g) + o.rows(g)) (Owners: n = 4 q under rows, node 3 the rest under rows2).
    /// `src` [n, w[p]] (this node's columns of its TP2 rank's partial, every row) -> `dst` [2, rows, w0 + w1]: for the
    /// owned rows, rank r's partial, pair 0's columns then pair 1's, for every r; quarters' dst rows of the owned rows
    /// laid out as quarters lays out a chunk of that many rows. Each node sends every other node that node's rows of
    /// its part and receives its own rows of theirs (one NCCL group of send / recv): ~3 n / 4 rows of one part in,
    /// where quarters takes 3 n; only bytes move.
    pub fn toOwners(t: *const Two, e: *const prompt.Engine, src: u64, dst: u64, o: Owners, w: [2]usize, esize: usize) !void {
        try t.toOwnersSend(e, e.s, src, o, w, esize);
        try t.toOwnersCopy(e, src, dst, o, w, esize);
    }

    /// toOwners' transfers (its NCCL part) on stream `s`: every other owner's rows of `src` to it, this node's rows of
    /// theirs into `parts`. This node's own rows of `src` are not read (TF_DS_PROMPT_OVL=wob computes them meanwhile).
    pub fn toOwnersSend(t: *const Two, e: *const prompt.Engine, s: cuda.Stream, src: u64, o: Owners, w: [2]usize, esize: usize) !void {
        return t.toOwnersSendAt(e, s, src, t.parts, o, w, esize);
    }

    /// toOwnersSend with this node's rows of the others' parts landing at `parts` (t.parts, or a half's area of it).
    fn toOwnersSendAt(t: *const Two, e: *const prompt.Engine, s: cuda.Stream, src: u64, parts: u64, o: Owners, w: [2]usize, esize: usize) !void {
        if (!o.valid()) return error.BadOwnerRows;
        const me = t.r + 2 * t.p;
        if (o.rows(3) * @max(w[0], w[1]) * esize > t.part_bytes) return error.PartTooBig;
        if (t.a2a_shift) {
            // TF_DS_PROMPT_A2A=shift: step k (1..3) sends to me + k and receives from me - k, one NCCL group a step and
            // the same order on every node, so each step is a permutation (no incast at any node's port); each
            // transfer is the one the single group posts (ownerExchange), only their timing differs
            for (1..4) |k| {
                const st = shiftStep(me, k);
                const xs = ownerExchange(me, st.to, o, w, esize);
                const xr = ownerExchange(me, st.from, o, w, esize);
                const s1 = [1]Comm.P2p{.{ .ptr = src + xs.send_off, .count = xs.send_bytes, .peer = @intCast(st.to) }};
                const r1 = [1]Comm.P2p{.{ .ptr = parts + xr.recv_off, .count = xr.recv_bytes, .peer = @intCast(st.from) }};
                try e.comm.sendRecv(&s1, &r1, .u8, s);
            }
        } else {
            var sends: [3]Comm.P2p = undefined;
            var recvs: [3]Comm.P2p = undefined;
            var k: usize = 0;
            for (0..4) |h| {
                if (h == me) continue;
                const x = ownerExchange(me, h, o, w, esize);
                sends[k] = .{ .ptr = src + x.send_off, .count = x.send_bytes, .peer = @intCast(h) };
                recvs[k] = .{ .ptr = parts + x.recv_off, .count = x.recv_bytes, .peer = @intCast(h) };
                k += 1;
            }
            try e.comm.sendRecv(&sends, &recvs, .u8, s);
        }
    }

    /// toOwners' copies into dst (on the compute stream, after the transfers).
    pub fn toOwnersCopy(t: *const Two, e: *const prompt.Engine, src: u64, dst: u64, o: Owners, w: [2]usize, esize: usize) !void {
        const me = t.r + 2 * t.p;
        for (ownerCopies(me, o, w, esize)) |cp| {
            const base = if (cp.from == me) src else t.parts;
            try e.ops.copyRows(e.s, base + cp.src_off, cp.src_ld, dst + cp.dst_off, cp.dst_ld, cp.bytes, o.rows(me));
        }
    }

    /// TF_DS_PROMPT_OVL=dn: toOwners' transfers of half h (0, 1) of a half-major part `src` [2, n, w[p] / 2] (half h's
    /// columns of every row at src + h n w[p] / 2 esize), on stream s: each half's exchange is toOwners' on a part of
    /// half the width, its receives in that half's area of `parts` (dnPartsOff).
    pub fn toOwnersSendHalf(t: *const Two, e: *const prompt.Engine, s: cuda.Stream, src: u64, o: Owners, w: [2]usize, esize: usize, h: usize) !void {
        const wh = try halves(w);
        if (dnPartsOff(o, w, esize, 2) > 4 * t.part_bytes) return error.PartTooBig;
        return t.toOwnersSendAt(e, s, src + h * o.n * wh[t.p] * esize, t.parts + dnPartsOff(o, w, esize, h), o, wh, esize);
    }

    /// TF_DS_PROMPT_OVL=dn: toOwnersCopy's bytes from a half-major part (after both halves' transfers): dst [2, rows,
    /// w0 + w1] exactly as toOwnersCopy fills it from the row-major part.
    pub fn toOwnersCopyHalves(t: *const Two, e: *const prompt.Engine, src: u64, dst: u64, o: Owners, w: [2]usize, esize: usize) !void {
        const me = t.r + 2 * t.p;
        const wh = try halves(w);
        for (0..2) |h| for (ownerCopiesHalf(me, o, w, esize, h)) |cp| {
            const base = if (cp.from == me) src + h * o.n * wh[t.p] * esize else t.parts + dnPartsOff(o, w, esize, h);
            try e.ops.copyRows(e.s, base + cp.src_off, cp.src_ld, dst + cp.dst_off, cp.dst_ld, cp.bytes, o.rows(me));
        };
    }

    /// Comm2D.gather (TP2 inside a pair: the DSpark drafter's sublayer partials): `src` [n, width] -> `dst` [2, n, width],
    /// this pair's two ranks in TP2 rank order: over the row pair's ring, else the four-rank all-gather's pair half
    /// (Comm2D.gather4(x)[2p:2p + 2]).
    pub fn pairGather(t: *const Two, e: *const prompt.Engine, src: u64, dst: u64, n: usize, width: usize, esize: usize) !void {
        if (t.rings) |rs| if (try ringed(rs.pairGather(e.s, src, dst, n, width, esize))) return;
        const wmax = try t.allParts(e, src, n, .{ width, width }, esize);
        try e.ops.copyRows(e.s, t.parts + 2 * t.p * n * wmax * esize, wmax * esize, dst, width * esize, width * esize, 2 * n);
    }

    /// wo_2d after wo_a: the rank's four groups (this node's two in ch.u [n, uw[p]], the partner's two), then wo_b
    /// for this pair's output columns: ch.pa [n, ow[p]] fp32, the matching columns of TP2's partial.
    pub fn woB(t: *const Two, e: *const prompt.Engine, ch: *prompt.Chunk, lay: weights.Layer) !void {
        try t.catRank(e, ch.u, t.u_full, ch.n, t.uw, 2);
        try prompt.pm(e, .comm);
        const k = t.uw[0] + t.uw[1];
        if (e.side) |sd| if (sd.lv.wob) if (ch.sp) |sp| if (ovl.ownFirst(sp.owners(), sp.g)) |ps| {
            // TF_DS_PROMPT_OVL=wob: wo_b of the other owners' rows, their transfers on the comm stream (toOwnersSend
            // reads no row of this node's), wo_b of this node's rows meanwhile; prompt.gather waits, then copies
            for ([_]ovl.Range{ ps[1], ps[2] }) |r| if (r.rows != 0) try t.woBRows(e, ch, lay, k, r);
            try prompt.pm(e, .dense);
            try sd.fork(e.s);
            try t.toOwnersSend(e, sd.stream, ch.pa, sp.owners(), t.ow, 4);
            try sd.end(.wob);
            try t.woBRows(e, ch, lay, k, ps[0]);
            try prompt.pm(e, .dense);
            ch.wob_sent = true;
            return;
        };
        try prompt.mm(e, ch, lay.wo_b, t.u_full, k, ch.pa, .fp32, lay.wo_b.n);
        try prompt.pm(e, .dense);
    }

    /// wo_b of rows [r.r0, + r.rows): the same prompt GEMM on those rows (each output row depends on its own input row
    /// alone: the kernel's tiles are row blocks, its K loop the same for every row).
    fn woBRows(t: *const Two, e: *const prompt.Engine, ch: *const prompt.Chunk, lay: weights.Layer, k: usize, r: ovl.Range) !void {
        try prompt.mmRowsAt(e, ch, r.rows, lay.wo_b, t.u_full + r.r0 * k * 2, k, ch.pa + r.r0 * lay.wo_b.n * 4, .fp32, lay.wo_b.n);
    }

    /// moe_2d's experts after the routing: this node's gate / up blocks, the rank's whole intermediate assembled, down
    /// for this node's output columns: ch.pm [n, dw[p]] fp32. A chunk of fewer than EXACT_ROWS rows takes the decode
    /// window's fused path and scratch (experts2d.fused_ok), as the rounds do.
    pub fn experts(t: *const Two, e: *const prompt.Engine, ch: *prompt.Chunk, lay: weights.Layer) !void {
        const c = e.c;
        const n = ch.n;
        if (n < exl3_experts.exact_rows) {
            // (decode-sized: its exchange inside, timed with the experts)
            defer prompt.pm(e, .moe) catch {};
            return round2d.experts(t, e, lay.experts, ch.xsd, ch.x, c.hidden, ch.pick, ch.wts, ch.pm, n, c.swiglu_limit);
        }
        if (e.x2) |x2| {
            // TF_DS_PROMPT_OVL=pack (an SP chunk on v2's parity experts): the pack exchange in two parts (packOvl)
            if (t.ir != 0) if (t.packSide(e, ch)) |sd| return t.packOvl(e, ch, lay, x2, sd);
            // TF_DS_EXL3_PROMPT=v2: gate / up with the rotation in the program (main and rest), the same pack bytes
            try exl3_prompt.gateUp(x2, e.ex, e.ops, e.s, lay.experts, ch.xs, if (t.ir != 0) t.rest_chunk.? else null, ch.x, c.hidden, ch.pick, n, c.swiglu_limit);
        } else {
            try exl3_experts2d.gateUp(e.ex, e.s, lay.experts, ch.xs, ch.x, c.hidden, ch.pick, n, c.swiglu_limit);
            // parity: the rest columns of the experts this pair computes them for
            if (t.ir != 0) try exl3_experts2d.gateUpRest(e.ex, e.ops, e.s, lay.experts, ch.xs, t.rest_chunk.?, ch.pick, n, c.swiglu_limit);
        }
        if (t.ir != 0) {
            // parity: the packs exchanged and assembled
            try prompt.pm(e, .moe);
            try t.assembleParity(e, lay.experts, ch.xs.xd, ch.pick, n * ch.xs.slots);
            defer prompt.pm(e, .moe) catch {};
            if (e.x2) |x2| return t.downV2(e, ch, lay, x2);
            return exl3_experts2d.down(e.ex, e.s, lay.experts, ch.xs, t.xd_full, ch.pick, ch.wts, ch.pm, n);
        }
        try prompt.pm(e, .moe);
        try t.catRank(e, ch.xs.xd, t.xd_full, n * ch.xs.slots, t.gu, 2);
        try prompt.pm(e, .comm);
        if (e.x2) |x2| {
            try t.downV2(e, ch, lay, x2);
        } else try exl3_experts2d.down(e.ex, e.s, lay.experts, ch.xs, t.xd_full, ch.pick, ch.wts, ch.pm, n);
        try prompt.pm(e, .moe);
    }

    /// The comm stream when TF_DS_PROMPT_OVL=pack acts on this chunk: an SP chunk (Engine.side is set for those alone)
    /// whose pack the column ring would not take (every prompt-size pack: over the ring's slot, so it is NCCL today;
    /// the same test as Ring.gatherInto's TooLarge, the same on every node).
    fn packSide(t: *const Two, e: *const prompt.Engine, ch: *const prompt.Chunk) ?*ovl.Side {
        const sd = e.side orelse return null;
        if (!(sd.lv.pack or sd.lv.pack3) or ch.sp == null) return null;
        const bytes = ch.n * ch.xs.slots * (t.gu[t.p] + t.ir) * 2;
        if (t.rings) |rs| if (bytes <= rs.col.s.max_bytes) return null;
        return sd;
    }

    /// TF_DS_PROMPT_OVL=pack (pack3: the main columns and their rows by halves, ovl.zig): gate / up's main columns and
    /// their epilogue (sc.xd's main rows final), the main rows to
    /// the column partner on the comm stream while the rest block's gate / up and epilogue run here, then the rest rows
    /// (packExchange's one send / recv as two: the same bytes from the same offsets into the same offsets of t.peer);
    /// assemble waits for both. Then down as packOvl's caller would run it.
    fn packOvl(t: *const Two, e: *const prompt.Engine, ch: *prompt.Chunk, lay: weights.Layer, x2: *const exl3_prompt.Kernels, sd: *ovl.Side) !void {
        const c = e.c;
        const n = ch.n;
        const rows = n * ch.xs.slots;
        const ig = t.gu[t.p];
        const g: u32 = @intCast(t.r + 2 * t.p);
        const xd = ch.xs.xd;
        if (sd.lv.pack3) {
            // pack3: the main columns by halves, each half's rows (half-major) sent once its epilogue is done
            const half = ovl.packHalf(ig) orelse return error.OvlPackShape;
            for (0..2) |h| {
                try exl3_prompt.gateUpHalf(x2, e.ex, e.ops, e.s, lay.experts, ch.xs, ch.x, c.hidden, ch.pick, n, c.swiglu_limit, h);
                try prompt.pm(e, .moe);
                try sd.fork(e.s);
                try e.comm.exchange(xd + h * rows * half * 2, rows * half, t.peer + h * rows * half * 2, rows * half, .f16, g ^ 2, sd.stream);
            }
        } else {
            try exl3_prompt.gateUpPart(x2, e.ex, e.ops, e.s, lay.experts, ch.xs, t.rest_chunk.?, ch.x, c.hidden, ch.pick, n, c.swiglu_limit, .main);
            try prompt.pm(e, .moe);
            try sd.fork(e.s);
            try e.comm.exchange(xd, rows * ig, t.peer, rows * ig, .f16, g ^ 2, sd.stream);
        }
        try exl3_prompt.gateUpPart(x2, e.ex, e.ops, e.s, lay.experts, ch.xs, t.rest_chunk.?, ch.x, c.hidden, ch.pick, n, c.swiglu_limit, .rest);
        try prompt.pm(e, .moe);
        try sd.fork(e.s);
        try e.comm.exchange(xd + rows * ig * 2, rows * t.ir, t.peer + rows * ig * 2, rows * t.ir, .f16, g ^ 2, sd.stream);
        try sd.end(.pack);
        try sd.join(e.s, .pack);
        try prompt.pm(e, .comm);
        const rest = lay.experts.rest orelse return error.NoRest;
        const part0 = if (t.p == 0) xd else t.peer;
        const part1 = if (t.p == 1) xd else t.peer;
        if (sd.lv.pack3) {
            try e.ops.assembleRestHalves(e.s, part0, part1, ch.pick, rest.owner, t.xd_full, rows, ig, t.ir);
        } else try e.ops.assembleRest(e.s, part0, part1, ch.pick, rest.owner, t.xd_full, rows, ig, t.ir);
        defer prompt.pm(e, .moe) catch {};
        return t.downV2(e, ch, lay, x2);
    }

    /// v2's down and combine into ch.pm: today's two launches, or with TF_DS_PROMPT_OVL=dn / dnc on an SP chunk the
    /// MoE partial's toOwners started on the comm stream from inside them (dnOvl; prompt.gather finishes it).
    fn downV2(t: *const Two, e: *const prompt.Engine, ch: *prompt.Chunk, lay: weights.Layer, x2: *const exl3_prompt.Kernels) !void {
        if (e.side) |sd| if (sd.lv.dn or sd.lv.dnc) if (ch.sp) |sp| if (ovl.othersFirst(sp.owners(), sp.g)) |rp| return t.dnOvl(e, ch, lay, x2, sd, sp.owners(), rp);
        return exl3_prompt.down(x2, e.ex, e.s, lay.experts, ch.xs, t.xd_full, ch.pick, ch.wts, ch.pm, ch.n);
    }

    /// TF_DS_PROMPT_OVL=dn / dnc (the plan in ovl.zig). dnc: down's GEMM as today, the combine of the other owners' rows,
    /// toOwners' transfers on the comm stream (they read no row of this node's), this node's rows' combine under them.
    /// dn: the same per column half, half A's transfers under half B's GEMM and combine; the partial half-major
    /// [2, n, dw / 2] (the halves' partials compact in sc.z, each half's svh_d table copied at load), so each half's
    /// owner rows are contiguous for its exchange. Every partial and every combined element is today's (the same
    /// program and combine arithmetic on the same operands); only launch extents and where results land change.
    fn dnOvl(t: *const Two, e: *const prompt.Engine, ch: *prompt.Chunk, lay: weights.Layer, x2: *const exl3_prompt.Kernels, sd: *ovl.Side, o: Owners, rp: [3]ovl.Range) !void {
        const ex = lay.experts;
        const n = ch.n;
        const sc = ch.xs;
        const dn: usize = ex.down_n;
        const nE: usize = ex.count;
        if (!sd.lv.dn) {
            // dnc
            try exl3_prompt.downGemm(x2, e.s, ex, sc, t.xd_full, n, .{ .c0 = 0, .cols = dn, .z = sc.z, .compact = false });
            for (rp[0..2]) |r| try exl3_prompt.combineRows(e.ex, e.s, ex.svh_d, sc, ch.pick, ch.wts, ch.pm, n, dn, nE, sc.z, r.r0, r.rows);
            try sd.fork(e.s);
            try t.toOwnersSend(e, sd.stream, ch.pm, o, t.dw, 4);
            try sd.end(.dn);
            try exl3_prompt.combineRows(e.ex, e.s, ex.svh_d, sc, ch.pick, ch.wts, ch.pm, n, dn, nE, sc.z, rp[2].r0, rp[2].rows);
            ch.pm_sent = .whole;
            return;
        }
        const half = ovl.dnHalf(dn) orelse return error.OvlDnShape;
        if (ex.svh_d_halves == 0) return error.OvlDnNoTables;
        const p = n * sc.slots;
        for (0..2) |h| {
            const z = sc.z + h * p * half * 4;
            const svh = ex.svh_d_halves + h * nE * half * 2;
            const out = ch.pm + h * n * half * 4;
            try exl3_prompt.downGemm(x2, e.s, ex, sc, t.xd_full, n, .{ .c0 = h * half, .cols = half, .z = z, .compact = true });
            for (rp[0..2]) |r| try exl3_prompt.combineRows(e.ex, e.s, svh, sc, ch.pick, ch.wts, out, n, half, nE, z, r.r0, r.rows);
            try sd.fork(e.s);
            try t.toOwnersSendHalf(e, sd.stream, ch.pm, o, t.dw, 4, h);
        }
        try sd.end(.dn);
        for (0..2) |h| {
            const z = sc.z + h * p * half * 4;
            try exl3_prompt.combineRows(e.ex, e.s, ex.svh_d_halves + h * nE * half * 2, sc, ch.pick, ch.wts, ch.pm + h * n * half * 4, n, half, nE, z, rp[2].r0, rp[2].rows);
        }
        ch.pm_sent = .halves;
    }

    /// Engram's projection summed over the TP2 ranks and cast, kv [n, ew0 + ew1] bf16 = bf16(g0 + g1) (model.py esum
    /// then .to(bf16)), from this node's fp32 columns `src` [n, ew[p]] with less traffic than the quarters: the row
    /// partner's columns over NCCL send / recv, one fp32 add (two operands: the same bits in either order) cast to
    /// bf16, then the column partner's bf16 sums the same way. `tmp` holds n * (ew[p] * 6 + ew[1 - p] * 2) bytes.
    pub fn engramSum(t: *const Two, e: *const prompt.Engine, src: u64, tmp: u64, kv: u64, n: usize) !void {
        const w = t.ew;
        const mine = n * w[t.p];
        const other = tmp;
        const half = tmp + mine * 4;
        const their = half + mine * 2;
        const g: u32 = @intCast(t.r + 2 * t.p);
        try e.comm.exchange(src, mine, other, mine, .f32, g ^ 1, e.s);
        try prompt.pm(e, .comm);
        try e.ops.add2Bf16(e.s, src, other, half, mine);
        try prompt.pm(e, .engram);
        try e.comm.exchange(half, mine, their, n * w[1 - t.p], .bf16, g ^ 2, e.s);
        try prompt.pm(e, .comm);
        const row = (w[0] + w[1]) * 2;
        for (0..2) |pg| {
            const from = if (pg == t.p) half else their;
            try e.ops.copyRows(e.s, from, w[pg] * 2, kv + pg * w[0] * 2, row, w[pg] * 2, n);
        }
    }

    /// engramSum for SP rows (TF_DS_PROMPT_ENGRAM=sp): kv's rows that this node owns, [o.r0(g), + o.rows(g)) of
    /// [n, ew0 + ew1] bf16, the only rows its Engram gate reads (prompt.engramApply on the mixes' view). The same two
    /// steps on fewer rows (engramPlan): the row partner's fp32 columns of this node's rows and of its column partner's
    /// rows (the row partner gets the other half), the same add2Bf16 (this node's operand first, as engramSum), then the
    /// column partner's bf16 sums of this node's rows. Every element of kv's owned rows is the add engramSum makes of
    /// the same two fp32 words; nothing else moves. Bytes in a chunk row: ~(ew·4 / 2 + ew·2 / 4) against ew·4 + ew·2.
    pub fn engramSumSp(t: *const Two, e: *const prompt.Engine, src: u64, tmp: u64, kv: u64, o: Owners) !void {
        if (!o.valid()) return error.BadOwnerRows;
        const me = t.r + 2 * t.p;
        const pl = engramPlan(me, o, t.ew);
        // (within what engramSum's tmp holds for o.n rows: Chunk.ekg)
        if (pl.tmp_bytes > o.n * (t.ew[t.p] * 6 + t.ew[1 - t.p] * 2)) return error.PartTooBig;
        var s1: [2]Comm.P2p = undefined;
        var r1: [2]Comm.P2p = undefined;
        for (0..2) |j| {
            s1[j] = .{ .ptr = src + pl.s1_send[j].off, .count = pl.s1_send[j].bytes, .peer = @intCast(pl.row_peer) };
            r1[j] = .{ .ptr = tmp + pl.s1_recv[j].off, .count = pl.s1_recv[j].bytes, .peer = @intCast(pl.row_peer) };
        }
        try e.comm.sendRecv(&s1, &r1, .u8, e.s);
        try prompt.pm(e, .comm);
        for (pl.adds) |ad| try e.ops.add2Bf16(e.s, src + ad.mine, tmp + ad.other, tmp + ad.out, ad.count);
        try prompt.pm(e, .engram);
        const s2 = [1]Comm.P2p{.{ .ptr = tmp + pl.s2_send.off, .count = pl.s2_send.bytes, .peer = @intCast(pl.col_peer) }};
        const r2 = [1]Comm.P2p{.{ .ptr = tmp + pl.s2_recv.off, .count = pl.s2_recv.bytes, .peer = @intCast(pl.col_peer) }};
        try e.comm.sendRecv(&s2, &r2, .u8, e.s);
        for (pl.copies) |cp| try e.ops.copyRows(e.s, tmp + cp.src_off, cp.src_ld, kv + cp.dst_off, cp.dst_ld, cp.bytes, cp.rows);
        try prompt.pm(e, .comm);
    }

    /// engramSumSp's first half on stream `s` (TF_DS_PROMPT_OVL=engram, the comm stream): step 1 with the row partner
    /// and the adds, the same transfers and the same add2Bf16 launches as engramSumSp (engramPlan).
    pub fn engramSp1(t: *const Two, e: *const prompt.Engine, s: cuda.Stream, src: u64, tmp: u64, o: Owners) !void {
        if (!o.valid()) return error.BadOwnerRows;
        const me = t.r + 2 * t.p;
        const pl = engramPlan(me, o, t.ew);
        if (pl.tmp_bytes > o.n * (t.ew[t.p] * 6 + t.ew[1 - t.p] * 2)) return error.PartTooBig;
        var s1: [2]Comm.P2p = undefined;
        var r1: [2]Comm.P2p = undefined;
        for (0..2) |j| {
            s1[j] = .{ .ptr = src + pl.s1_send[j].off, .count = pl.s1_send[j].bytes, .peer = @intCast(pl.row_peer) };
            r1[j] = .{ .ptr = tmp + pl.s1_recv[j].off, .count = pl.s1_recv[j].bytes, .peer = @intCast(pl.row_peer) };
        }
        try e.comm.sendRecv(&s1, &r1, .u8, s);
        for (pl.adds) |ad| try e.ops.add2Bf16(s, src + ad.mine, tmp + ad.other, tmp + ad.out, ad.count);
    }

    /// engramSumSp's second half on stream `s`: step 2 with the column partner and the copies into kv's owned rows.
    pub fn engramSp2(t: *const Two, e: *const prompt.Engine, s: cuda.Stream, tmp: u64, kv: u64, o: Owners) !void {
        if (!o.valid()) return error.BadOwnerRows;
        const me = t.r + 2 * t.p;
        const pl = engramPlan(me, o, t.ew);
        const s2 = [1]Comm.P2p{.{ .ptr = tmp + pl.s2_send.off, .count = pl.s2_send.bytes, .peer = @intCast(pl.col_peer) }};
        const r2 = [1]Comm.P2p{.{ .ptr = tmp + pl.s2_recv.off, .count = pl.s2_recv.bytes, .peer = @intCast(pl.col_peer) }};
        try e.comm.sendRecv(&s2, &r2, .u8, s);
        for (pl.copies) |cp| try e.ops.copyRows(s, tmp + cp.src_off, cp.src_ld, kv + cp.dst_off, cp.dst_ld, cp.bytes, cp.rows);
    }

    /// Engram's projection summed over the ranks from its column quarters (model.py esum): the gathered width.
    pub fn engramWidth(t: *const Two) usize {
        return t.ew[0] + t.ew[1];
    }
};

/// SP rows' owners of a chunk of n rows: node h owns rows [r0(h), r0(h) + rows(h)); q rows each for nodes 0-2, node 3
/// the rest. TF_DS_PROMPT_SP=rows: n a multiple of 64, q = n / 4 (four quarters). rows2 (of): q = 16 (n / 64), a whole
/// number of 16-row tiles, node 3 n - 3 q (at least q, under q + 64, the same %16 class as n).
pub const Owners = struct {
    q: usize,
    n: usize,

    pub fn of(n: usize) Owners {
        return .{ .q = 16 * (n / 64), .n = n };
    }

    pub fn r0(o: Owners, h: usize) usize {
        return h * o.q;
    }

    pub fn rows(o: Owners, h: usize) usize {
        return if (h == 3) o.n - 3 * o.q else o.q;
    }

    pub fn valid(o: Owners) bool {
        return o.q > 0 and 3 * o.q <= o.n and o.rows(3) >= o.q;
    }

    /// Node 3's rows past 4 q (the all-gather of equal parts misses them): [first, count].
    pub fn tail(o: Owners) [2]usize {
        return .{ 4 * o.q, o.n - @min(o.n, 4 * o.q) };
    }
};

/// toOwners' transfer between owner `me` and node `h` (h != me) of a chunk's rows: me sends h's rows of its part
/// (row stride w[me / 2]) and receives h's part of its own rows into `parts` slot h (slots of node 3's rows at row
/// stride wmax, so every slot fits the most rows of the widest part). Offsets and sizes in bytes. Four quarters
/// (n = 4 q): slot h at h q rows, as before.
pub const OwnerXfer = struct { send_off: usize, send_bytes: usize, recv_off: usize, recv_bytes: usize };

pub fn ownerExchange(me: usize, h: usize, o: Owners, w: [2]usize, esize: usize) OwnerXfer {
    const wmax = @max(w[0], w[1]);
    const slot = o.rows(3);
    return .{ .send_off = o.r0(h) * w[me / 2] * esize, .send_bytes = o.rows(h) * w[me / 2] * esize, .recv_off = h * slot * wmax * esize, .recv_bytes = o.rows(me) * w[h / 2] * esize };
}

/// One of toOwners' copies into dst [2, q, w0 + w1]: q rows of `bytes` from node `from`'s part of the owned rows (the
/// owner's own src when from == me, at src_off; else what `from` sent, at src_off of parts), into rank from % 2's
/// partial at pair from / 2's columns (quarters' placement of node `from`).
pub const OwnerCopy = struct { from: usize, src_off: usize, src_ld: usize, dst_off: usize, dst_ld: usize, bytes: usize };

pub fn ownerCopies(me: usize, o: Owners, w: [2]usize, esize: usize) [4]OwnerCopy {
    const q = o.rows(me);
    const row = (w[0] + w[1]) * esize;
    var out: [4]OwnerCopy = undefined;
    for (0..4) |h| {
        const rh = h % 2;
        const ph = h / 2;
        out[h] = .{
            .from = h,
            .src_off = if (h == me) o.r0(me) * w[ph] * esize else ownerExchange(me, h, o, w, esize).recv_off,
            .src_ld = w[ph] * esize,
            .dst_off = rh * q * row + ph * w[0] * esize,
            .dst_ld = row,
            .bytes = w[ph] * esize,
        };
    }
    return out;
}

/// TF_DS_PROMPT_OVL=dn: each pair's half width (w[p] / 2; every width even).
pub fn halves(w: [2]usize) ![2]usize {
    if (w[0] % 2 != 0 or w[1] % 2 != 0) return error.OddPartWidth;
    return .{ w[0] / 2, w[1] / 2 };
}

/// TF_DS_PROMPT_OVL=dn: where half h's receives start in `parts` (h = 2: the end of both): half 0's slots (toOwners'
/// at half the width) then half 1's.
pub fn dnPartsOff(o: Owners, w: [2]usize, esize: usize, h: usize) usize {
    return h * 4 * o.rows(3) * @max(w[0] / 2, w[1] / 2) * esize;
}

/// TF_DS_PROMPT_OVL=dn: toOwners' copies of half h of half-major parts into dst [2, q, w0 + w1]: half h of node
/// `from`'s columns (its own src's half h, or what it sent into half h's area of parts) into the columns ownerCopies
/// puts them in: rank from % 2's partial, pair from / 2's columns, offset by h halves. Source offsets are ownerCopies'
/// at the half width (relative to the half's src, or to dnPartsOff(h)).
pub fn ownerCopiesHalf(me: usize, o: Owners, w: [2]usize, esize: usize, h: usize) [4]OwnerCopy {
    const wh = [2]usize{ w[0] / 2, w[1] / 2 };
    const q = o.rows(me);
    const row = (w[0] + w[1]) * esize;
    var out = ownerCopies(me, o, wh, esize);
    for (&out) |*cp| {
        const rh = cp.from % 2;
        const ph = cp.from / 2;
        cp.dst_off = rh * q * row + ph * w[0] * esize + h * wh[ph] * esize;
        cp.dst_ld = row;
        cp.bytes = wh[ph] * esize;
    }
    return out;
}

/// TF_DS_PROMPT_A2A=shift's step k (1..3) on node me: it sends to me + k and receives from me - k (mod 4).
pub const Shift = struct { to: usize, from: usize };

pub fn shiftStep(me: usize, k: usize) Shift {
    return .{ .to = (me + k) % 4, .from = (me + 4 - k) % 4 };
}

/// engramSumSp's transfers, adds and copies on node me (offsets in bytes: sends from src, receives / adds / copies in
/// tmp, copies' destinations in kv). Areas of tmp: `other` fp32 [rows(me) + rows(mate), ew[p]] (the row partner's
/// columns of me's rows, then of its column partner's: mate = me ^ 2), `half` bf16 of the same rows (the sums), `their`
/// bf16 [rows(me), ew[1 - p]] (the column partner's sums of me's rows).
pub const Seg = struct { off: usize, bytes: usize };
pub const EngramAdd = struct { mine: usize, other: usize, out: usize, count: usize };
pub const EngramCopy = struct { src_off: usize, src_ld: usize, dst_off: usize, dst_ld: usize, bytes: usize, rows: usize };
pub const EngramPlan = struct {
    row_peer: usize,
    col_peer: usize,
    s1_send: [2]Seg, // to the row partner: its rows, then its column partner's (fp32, this node's columns)
    s1_recv: [2]Seg, // from it: me's rows, then mate's
    adds: [2]EngramAdd, // me's rows, mate's: bf16(mine + the row partner's)
    s2_send: Seg, // to the column partner: mate's sums
    s2_recv: Seg, // from it: me's rows of the other pair's sums
    copies: [2]EngramCopy, // into kv's rows of me: pair 0's columns, then pair 1's
    tmp_bytes: usize,
};

pub fn engramPlan(me: usize, o: Owners, w: [2]usize) EngramPlan {
    const p = me / 2;
    const rp = me ^ 1;
    const mate = me ^ 2;
    const wp = w[p];
    const wq = w[1 - p];
    const row = (w[0] + w[1]) * 2;
    const rm = o.rows(me);
    const rt = o.rows(mate);
    const other: usize = 0;
    const half = std.mem.alignForward(usize, other + (rm + rt) * wp * 4, 256);
    const their = std.mem.alignForward(usize, half + (rm + rt) * wp * 2, 256);
    var pl: EngramPlan = undefined;
    pl.row_peer = rp;
    pl.col_peer = mate;
    pl.s1_send = .{
        .{ .off = o.r0(rp) * wp * 4, .bytes = o.rows(rp) * wp * 4 },
        .{ .off = o.r0(rp ^ 2) * wp * 4, .bytes = o.rows(rp ^ 2) * wp * 4 },
    };
    pl.s1_recv = .{ .{ .off = other, .bytes = rm * wp * 4 }, .{ .off = other + rm * wp * 4, .bytes = rt * wp * 4 } };
    pl.adds = .{
        .{ .mine = o.r0(me) * wp * 4, .other = other, .out = half, .count = rm * wp },
        .{ .mine = o.r0(mate) * wp * 4, .other = other + rm * wp * 4, .out = half + rm * wp * 2, .count = rt * wp },
    };
    pl.s2_send = .{ .off = half + rm * wp * 2, .bytes = rt * wp * 2 };
    pl.s2_recv = .{ .off = their, .bytes = rm * wq * 2 };
    for (0..2) |pg| {
        pl.copies[pg] = .{
            .src_off = if (pg == p) half else their,
            .src_ld = w[pg] * 2,
            .dst_off = o.r0(me) * row + pg * w[0] * 2,
            .dst_ld = row,
            .bytes = w[pg] * 2,
            .rows = rm,
        };
    }
    pl.tmp_bytes = their + rm * wq * 2;
    return pl;
}

/// fp32 -> bf16, round to nearest even (__float2bfloat16_rn on finite values), for the host tests.
fn bf16rn(x: f32) u16 {
    const b: u32 = @bitCast(x);
    const lsb = (b >> 16) & 1;
    return @intCast((b +% (0x7fff + lsb)) >> 16);
}

test "TF_DS_PROMPT_A2A=shift: three permutation steps that post exactly toOwners' transfers" {
    for (1..4) |k| {
        var hit: [4]u8 = @splat(0);
        for (0..4) |me| {
            const st = shiftStep(me, k);
            try std.testing.expect(st.to != me and st.from != me);
            // the node me sends to receives from me in the same step
            try std.testing.expectEqual(me, shiftStep(st.to, k).from);
            hit[st.to] += 1;
        }
        // a permutation: every node receives from exactly one peer a step
        for (hit) |x| try std.testing.expectEqual(@as(u8, 1), x);
    }
    // over the three steps each node sends to and receives from every other node once
    for (0..4) |me| {
        var to: [4]u8 = @splat(0);
        var from: [4]u8 = @splat(0);
        for (1..4) |k| {
            to[shiftStep(me, k).to] += 1;
            from[shiftStep(me, k).from] += 1;
        }
        for (0..4) |h| {
            try std.testing.expectEqual(@as(u8, if (h == me) 0 else 1), to[h]);
            try std.testing.expectEqual(@as(u8, if (h == me) 0 else 1), from[h]);
        }
    }
}

test "TF_DS_PROMPT_ENGRAM=sp: each owner's rows of kv equal engramSum's, from the bytes the four nodes send" {
    // four nodes' fp32 parts of the two ranks' Engram projections (rank g % 2, pair g / 2's columns, every row); the
    // reference kv = bf16(p0 + p1) of every element (engramSum: two operands, the same bits in either order)
    inline for (.{ Owners{ .q = 2, .n = 8 }, Owners{ .q = 2, .n = 11 }, Owners{ .q = 3, .n = 12 } }) |o| {
        const n: usize = o.n;
        const w = [2]usize{ 5, 3 };
        const dw: usize = w[0] + w[1];
        var prng = std.Random.DefaultPrng.init(17 + n);
        const rnd = prng.random();
        var proj: [2][n][dw]f32 = undefined;
        for (0..2) |r| for (0..n) |i| for (0..dw) |j| {
            proj[r][i][j] = (rnd.float(f32) - 0.5) * std.math.pow(f32, 2, @floatFromInt(rnd.intRangeAtMost(i32, -20, 20)));
        };
        var ref: [n][dw]u16 = undefined;
        for (0..n) |i| for (0..dw) |j| {
            ref[i][j] = bf16rn(proj[0][i][j] + proj[1][i][j]);
        };
        // node g's src [n, w[p]] fp32 as bytes
        var src: [4][n * 5]f32 = undefined;
        for (0..4) |g| for (0..n) |i| for (0..w[g / 2]) |j| {
            src[g][i * w[g / 2] + j] = proj[g % 2][i][(g / 2) * w[0] + j];
        };
        // every node's tmp, then the two steps as the four nodes run them
        var tmp: [4][4096]u8 align(4) = undefined;
        for (&tmp) |*x| @memset(x, 0xAB);
        var pls: [4]EngramPlan = undefined;
        for (0..4) |g| {
            pls[g] = engramPlan(g, o, w);
            try std.testing.expect(pls[g].tmp_bytes <= 4096);
        }
        // step 1: g's receive j from its row partner = the partner's send j (the same sizes, in the posted order)
        for (0..4) |g| {
            const rp = pls[g].row_peer;
            try std.testing.expectEqual(g, pls[rp].row_peer);
            for (0..2) |j| {
                const s = pls[rp].s1_send[j];
                const r = pls[g].s1_recv[j];
                try std.testing.expectEqual(s.bytes, r.bytes);
                const sb = std.mem.sliceAsBytes(&src[rp]);
                @memcpy(tmp[g][r.off..][0..r.bytes], sb[s.off..][0..s.bytes]);
            }
        }
        // the adds (add2Bf16: __float2bfloat16_rn(__fadd_rn(mine, other)), elementwise)
        for (0..4) |g| {
            const sb = std.mem.sliceAsBytes(&src[g]);
            for (pls[g].adds) |ad| {
                for (0..ad.count) |i| {
                    const a = std.mem.bytesToValue(f32, sb[ad.mine + 4 * i ..][0..4]);
                    const b = std.mem.bytesToValue(f32, tmp[g][ad.other + 4 * i ..][0..4]);
                    std.mem.writeInt(u16, tmp[g][ad.out + 2 * i ..][0..2], bf16rn(a + b), .little);
                }
            }
        }
        // step 2: from the column partner (copied from a snapshot: every node sends before any receives)
        const snap = tmp;
        for (0..4) |g| {
            const cp = pls[g].col_peer;
            try std.testing.expectEqual(g, pls[cp].col_peer);
            const s = pls[cp].s2_send;
            const r = pls[g].s2_recv;
            try std.testing.expectEqual(s.bytes, r.bytes);
            @memcpy(tmp[g][r.off..][0..r.bytes], snap[cp][s.off..][0..s.bytes]);
        }
        // the copies into kv, and kv's owned rows against the reference
        for (0..4) |g| {
            var kv: [n * dw]u16 = @splat(0xffff);
            const kb = std.mem.sliceAsBytes(&kv);
            for (pls[g].copies) |cp| for (0..cp.rows) |i| {
                @memcpy(kb[cp.dst_off + i * cp.dst_ld ..][0..cp.bytes], tmp[g][cp.src_off + i * cp.src_ld ..][0..cp.bytes]);
            };
            for (0..o.rows(g)) |i| for (0..dw) |j| {
                try std.testing.expectEqual(ref[o.r0(g) + i][j], kv[(o.r0(g) + i) * dw + j]);
            };
        }
        // bytes a node receives against engramSum's (n rows of its fp32 columns, then n of the other pair's bf16)
        for (0..4) |g| {
            const got = pls[g].s1_recv[0].bytes + pls[g].s1_recv[1].bytes + pls[g].s2_recv.bytes;
            try std.testing.expect(got < n * w[g / 2] * 4 + n * w[1 - g / 2] * 2);
        }
    }
}

test "toOwners: every owner's rows of quarters' [2, n, D], from the bytes the four nodes send (even and rows2's split)" {
    // four nodes' parts of two TP2 ranks' partials, uneven pair widths, as bytes (esize 4); four quarters (rows) and
    // node 3 taking the rest (rows2)
    inline for (.{ Owners{ .q = 2, .n = 8 }, Owners{ .q = 2, .n = 11 }, Owners{ .q = 3, .n = 12 } }) |o| {
        const n: usize = o.n;
        const w = [2]usize{ 5, 3 };
        const es: usize = 4;
        const dw: usize = w[0] + w[1];
        const wmax: usize = @max(w[0], w[1]);
        try std.testing.expect(o.valid());
        // partial[r][row][col]: a distinct u32 per entry
        var partial: [2][n][dw]u32 = undefined;
        for (0..2) |r| for (0..n) |i| for (0..dw) |j| {
            partial[r][i][j] = @intCast(r * 1000000 + i * 1000 + j);
        };
        // node g's part: rank g % 2's partial at pair g / 2's columns, every row
        var parts: [4][n * wmax]u32 = undefined;
        for (0..4) |g| for (0..n) |i| for (0..w[g / 2]) |j| {
            parts[g][i * w[g / 2] + j] = partial[g % 2][i][(g / 2) * w[0] + j];
        };
        // the owners' rows cover the chunk once, in order
        var covered: usize = 0;
        for (0..4) |h| {
            try std.testing.expectEqual(covered, o.r0(h));
            covered += o.rows(h);
        }
        try std.testing.expectEqual(n, covered);
        // quarters' dst [2, n, D] is partial itself; each owner's toOwners dst must be its rows of it
        for (0..4) |me| {
            const rm = o.rows(me);
            var staged: [4 * n * wmax]u32 = @splat(0);
            const staged_b = std.mem.sliceAsBytes(&staged);
            for (0..4) |h| {
                if (h == me) continue;
                // h's send to me (its view of the same exchange) lands where me's receive from h says
                const sx = ownerExchange(h, me, o, w, es);
                const rx = ownerExchange(me, h, o, w, es);
                try std.testing.expectEqual(sx.send_bytes, rx.recv_bytes);
                // the slots never overlap and fit node 3's rows at the widest part
                try std.testing.expect(rx.recv_off + rx.recv_bytes <= (h + 1) * o.rows(3) * wmax * es);
                const src_b = std.mem.sliceAsBytes(&parts[h]);
                @memcpy(staged_b[rx.recv_off..][0..rx.recv_bytes], src_b[sx.send_off..][0..sx.send_bytes]);
            }
            var dst: [2 * n * dw]u32 = @splat(0xffffffff);
            const dst_b = std.mem.sliceAsBytes(&dst);
            for (ownerCopies(me, o, w, es)) |cp| {
                const base = if (cp.from == me) std.mem.sliceAsBytes(&parts[me]) else staged_b;
                for (0..rm) |i| @memcpy(dst_b[cp.dst_off + i * cp.dst_ld ..][0..cp.bytes], base[cp.src_off + i * cp.src_ld ..][0..cp.bytes]);
            }
            for (0..2) |r| for (0..rm) |i| for (0..dw) |j| {
                try std.testing.expectEqual(partial[r][o.r0(me) + i][j], dst[(r * rm + i) * dw + j]);
            };
        }
    }
}

test "TF_DS_PROMPT_OVL=dn: toOwners of half-major parts puts every byte where toOwners of the row-major part does" {
    // even pair widths (halves of whole elements), four quarters and rows2's uneven node 3
    inline for (.{ Owners{ .q = 2, .n = 8 }, Owners{ .q = 2, .n = 11 }, Owners{ .q = 3, .n = 12 } }) |o| {
        inline for (.{ [2]usize{ 6, 4 }, [2]usize{ 4, 4 } }) |w| {
            const n: usize = o.n;
            const es: usize = 4;
            const dw: usize = w[0] + w[1];
            const wh = comptime halves(w) catch unreachable;
            const whmax = comptime @max(wh[0], wh[1]);
            var partial: [2][n][dw]u32 = undefined;
            for (0..2) |r| for (0..n) |i| for (0..dw) |j| {
                partial[r][i][j] = @intCast(r * 1000000 + i * 1000 + j);
            };
            // node g's part half-major: half h of its columns of every row at h n wh[g/2]
            var parts: [4][n * dw]u32 = undefined;
            for (0..4) |g| for (0..2) |h| for (0..n) |i| for (0..wh[g / 2]) |j| {
                parts[g][h * n * wh[g / 2] + i * wh[g / 2] + j] = partial[g % 2][i][(g / 2) * w[0] + h * wh[g / 2] + j];
            };
            for (0..4) |me| {
                const rm = o.rows(me);
                var staged: [2 * 4 * n * whmax]u32 = @splat(0);
                const staged_b = std.mem.sliceAsBytes(&staged);
                try std.testing.expect(dnPartsOff(o, w, es, 2) <= staged_b.len);
                for (0..2) |h| for (0..4) |peer| {
                    if (peer == me) continue;
                    const sx = ownerExchange(peer, me, o, wh, es);
                    const rx = ownerExchange(me, peer, o, wh, es);
                    try std.testing.expectEqual(sx.send_bytes, rx.recv_bytes);
                    const base = dnPartsOff(o, w, es, h);
                    // inside half h's area
                    try std.testing.expect(base + rx.recv_off + rx.recv_bytes <= dnPartsOff(o, w, es, h + 1));
                    const src_b = std.mem.sliceAsBytes(&parts[peer])[h * n * wh[peer / 2] * es ..];
                    @memcpy(staged_b[base + rx.recv_off ..][0..rx.recv_bytes], src_b[sx.send_off..][0..sx.send_bytes]);
                };
                var dst: [2 * n * dw]u32 = @splat(0xffffffff);
                const dst_b = std.mem.sliceAsBytes(&dst);
                for (0..2) |h| for (ownerCopiesHalf(me, o, w, es, h)) |cp| {
                    const base = if (cp.from == me) std.mem.sliceAsBytes(&parts[me])[h * n * wh[me / 2] * es ..] else staged_b[dnPartsOff(o, w, es, h)..];
                    for (0..rm) |i| @memcpy(dst_b[cp.dst_off + i * cp.dst_ld ..][0..cp.bytes], base[cp.src_off + i * cp.src_ld ..][0..cp.bytes]);
                };
                // every element of the owner's [2, rows, D]: today's toOwners result (the test above checks that is
                // partial's rows)
                for (0..2) |r| for (0..rm) |i| for (0..dw) |j| {
                    try std.testing.expectEqual(partial[r][o.r0(me) + i][j], dst[(r * rm + i) * dw + j]);
                };
            }
        }
    }
    try std.testing.expectError(error.OddPartWidth, halves(.{ 5, 4 }));
}

test "rows2's owners: whole 16-row tiles for nodes 0-2, node 3 the rest in n's %16 class; four quarters when n % 64 == 0" {
    for ([_]usize{ 256, 2048, 4096 }) |n| {
        const o = Owners.of(n);
        try std.testing.expectEqual(n / 4, o.q);
        try std.testing.expectEqual(n / 4, o.rows(3));
        try std.testing.expectEqual([2]usize{ n, 0 }, o.tail());
    }
    var n: usize = 256;
    while (n <= 4096) : (n += 1) {
        const o = Owners.of(n);
        try std.testing.expect(o.valid());
        try std.testing.expect(o.q % 16 == 0 and o.q >= 64);
        try std.testing.expect(o.rows(3) >= o.q and o.rows(3) < o.q + 64);
        try std.testing.expectEqual(n % 16, o.rows(3) % 16); // the class of today's replicated launch on n rows
        try std.testing.expect(o.r0(3) % 16 == 0);
        const t = o.tail();
        try std.testing.expectEqual(n, t[0] + t[1]);
        try std.testing.expect(t[1] < 64);
    }
}

test "a node's split: TP over two, the 2D split over four" {
    try std.testing.expectEqual(plan.Split{ .rank = 1, .world = 2 }, try split(1, 2));
    try std.testing.expectEqual(plan.Split{ .rank = 0, .world = 2, .pair = 1 }, try split(2, 4));
    try std.testing.expectEqual(plan.Split{ .rank = 1, .world = 2, .pair = 1 }, try split(3, 4));
    try std.testing.expectError(error.UnsupportedWorld, split(0, 3));
}
