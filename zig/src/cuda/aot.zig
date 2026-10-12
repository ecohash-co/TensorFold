//! A captured set of Triton cubins (aot.json + cubins/): each launch picks the variant Triton itself would have picked.

const std = @import("std");
const abi = @import("abi.zig");
const Driver = @import("driver.zig").Driver;
const Stream = @import("stream.zig").Stream;
const launch = @import("launch.zig");
const triton = @import("triton.zig");

pub const ParamJson = struct { name: []const u8, type: []const u8, div16: bool, nospec: bool };
pub const ConstJson = struct { int: ?i64 = null, f32: ?u32 = null };
pub const KernelJson = struct {
    @"fn": []const u8,
    hash: []const u8,
    name: []const u8,
    num_warps: u32,
    num_ctas: u32 = 1,
    shared: u32 = 0,
    global_scratch: u32 = 0,
    global_align: u32 = 1,
    profile_scratch: u32 = 0,
    pdl: bool = false,
    params: []ParamJson,
    consts: std.json.ArrayHashMap(ConstJson),
};
pub const SetJson = struct { kernels: []KernelJson };

/// One argument of a launch, by the kernel's parameter name.
pub const Arg = struct {
    name: []const u8,
    value: Value,

    pub const Value = union(enum) { ptr: struct { addr: u64, ty: []const u8 }, i32: i32, f32: f32, u64: u64 };
};

/// A constexpr the call site compiled the kernel with (Python's keyword arguments): ints, bools, fp32 bits.
pub const Const = struct { name: []const u8, int: ?i64 = null, f32: ?f32 = null };

pub fn ptr(name: []const u8, ty: []const u8, addr: u64) Arg {
    return .{ .name = name, .value = .{ .ptr = .{ .addr = addr, .ty = ty } } };
}
pub fn int(name: []const u8, v: i32) Arg {
    return .{ .name = name, .value = .{ .i32 = v } };
}
pub fn float(name: []const u8, v: f32) Arg {
    return .{ .name = name, .value = .{ .f32 = v } };
}
pub fn word(name: []const u8, v: u64) Arg {
    return .{ .name = name, .value = .{ .u64 = v } };
}
pub fn ci(name: []const u8, v: i64) Const {
    return .{ .name = name, .int = v };
}
pub fn cf(name: []const u8, v: f32) Const {
    return .{ .name = name, .f32 = v };
}

const Variant = struct {
    spec: KernelJson,
    kernel: triton.Kernel,
    /// Said once on stderr when a launch first takes this variant in place of its own specialization.
    told: std.atomic.Value(bool) = .init(false),
};

/// What a call site has shown its kernel's code does not depend on, beyond the fallback `fits` makes for every launch
/// (homelab, from 969d13b): `ptrs8` may be 8- but not 16-aligned where the variant assumed 16 (arguments the code
/// reads as one 8-byte scalar, or not at all). Such an argument counts as not held. Empty: upstream's rule only.
pub const Relax = struct {
    ptrs8: []const []const u8 = &.{},
    /// Per captured variant (by its hash): pointers its SASS reads and writes only with accesses of at most 8 bytes,
    /// or not at all, so they may be 8-aligned where it assumed 16 (a kit's generated table, e.g. dsv41's
    /// aot_ptr8.zig). Taken only when nothing fits without it (`pick`): a launch that ran before keeps its variant.
    proven: []const Proven = &.{},

    pub const Proven = struct { hash: []const u8, ptrs8: []const []const u8 };

    fn has(names: []const []const u8, name: []const u8) bool {
        for (names) |x| if (std.mem.eql(u8, x, name)) return true;
        return false;
    }

    /// Whether `name` may be 8-aligned on variant `hash`.
    fn allows(r: Relax, hash: []const u8, name: []const u8) bool {
        if (has(r.ptrs8, name)) return true;
        for (r.proven) |v| if (std.mem.eql(u8, v.hash, hash)) return has(v.ptrs8, name);
        return false;
    }
};

pub const Set = struct {
    parsed: std.json.Parsed(SetJson),
    variants: []Variant,
    gpa: std.mem.Allocator,
    /// TF_AOT_WEAKEST=1: every launch takes its most general fitting variant (a gate of the fallback's exactness).
    weakest: bool = false,

    /// Loads every cubin listed in `dir`/aot.json into its own module.
    pub fn load(gpa: std.mem.Allocator, io: std.Io, d: *const Driver, device: abi.Device, dir: []const u8) !Set {
        const path = try std.fs.path.join(gpa, &.{ dir, "aot.json" });
        defer gpa.free(path);
        const text = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 24));
        defer gpa.free(text);
        const parsed = try parseSet(gpa, text);
        errdefer parsed.deinit();
        const variants = try gpa.alloc(Variant, parsed.value.kernels.len);
        var n: usize = 0;
        errdefer {
            for (variants[0..n]) |*v| v.kernel.unload();
            gpa.free(variants);
        }
        for (parsed.value.kernels) |k| {
            if (k.global_scratch != 0 or k.profile_scratch != 0) return error.ScratchUnsupported;
            const file = try std.fmt.allocPrint(gpa, "{s}/cubins/{s}.cubin", .{ dir, k.hash });
            defer gpa.free(file);
            const cubin = try std.Io.Dir.cwd().readFileAllocOptions(io, file, gpa, .limited(1 << 26), .@"16", null);
            defer gpa.free(cubin);
            const name_z = try gpa.dupeSentinel(u8, k.name, 0);
            defer gpa.free(name_z);
            const meta: triton.Meta = .{ .name = k.name, .num_warps = k.num_warps, .num_ctas = k.num_ctas, .shared = k.shared, .launch_pdl = k.pdl };
            variants[n] = .{ .spec = k, .kernel = try triton.Kernel.load(d, device, cubin, meta, name_z) };
            n += 1;
        }
        const weakest = if (std.c.getenv("TF_AOT_WEAKEST")) |v| std.mem.eql(u8, std.mem.span(v), "1") else false;
        return .{ .parsed = parsed, .variants = variants, .gpa = gpa, .weakest = weakest };
    }

    pub fn deinit(self: *Set) void {
        for (self.variants) |*v| v.kernel.unload();
        self.gpa.free(self.variants);
        self.parsed.deinit();
        self.* = undefined;
    }

    /// The variant of `function` compiled for these constexprs and these arguments' specialization; without one, the
    /// fitting variant that assumes least less (fits: the same constexprs, each assumption it makes held by the
    /// launch), named once on stderr.
    pub fn find(self: *const Set, function: []const u8, args: []const Arg, consts: []const Const) !*const Variant {
        return self.findRelaxed(function, args, consts, .{});
    }

    /// `find`, the fallback's fit also allowing what `relax` lists (Relax).
    pub fn findRelaxed(self: *const Set, function: []const u8, args: []const Arg, consts: []const Const, relax: Relax) !*const Variant {
        if (pick(self.parsed.value.kernels, function, args, consts, relax, self.weakest)) |got| {
            const v = &self.variants[got.index];
            if (got.exact) return v;
            if (!v.told.swap(true, .monotonic)) std.log.warn("{s}: a launch whose specialization no captured variant has runs variant {s} (assumes less)", .{ function, v.spec.hash[0..@min(12, v.spec.hash.len)] });
            return v;
        }
        std.log.err("no captured Triton variant of {s} for this launch:", .{function});
        for (args) |a| switch (a.value) {
            .ptr => |p| std.log.err("  {s}: {s} at {x} (16-aligned {})", .{ a.name, p.ty, p.addr, p.addr % 16 == 0 }),
            .i32 => |x| std.log.err("  {s}: i32 {d}", .{ a.name, x }),
            .f32 => |x| std.log.err("  {s}: fp32 {d}", .{ a.name, x }),
            .u64 => |x| std.log.err("  {s}: u64 {d}", .{ a.name, x }),
        };
        return error.MissingTritonVariant;
    }

    /// The smallest value of constexpr `name` at or above `at_least` among `function`'s variants.
    pub fn smallestConst(self: *const Set, function: []const u8, name: []const u8, at_least: i64) ?i64 {
        var best: ?i64 = null;
        for (self.variants) |v| {
            if (!std.mem.eql(u8, v.spec.@"fn", function)) continue;
            const c = (v.spec.consts.map.get(name) orelse continue).int orelse continue;
            if (c >= at_least and (best == null or c < best.?)) best = c;
        }
        return best;
    }

    /// Launches `function` on `grid` as Triton's launcher would: runtime arguments in the variant's order, scratch null.
    pub fn run(self: *const Set, stream: Stream, function: []const u8, grid: [3]u32, args: []const Arg, consts: []const Const) !void {
        return self.runRelaxed(stream, function, grid, args, consts, .{});
    }

    /// `run` with the variant from `findRelaxed`.
    pub fn runRelaxed(self: *const Set, stream: Stream, function: []const u8, grid: [3]u32, args: []const Arg, consts: []const Const, relax: Relax) !void {
        const v = try self.findRelaxed(function, args, consts, relax);
        var packed_args: launch.Args = .{};
        for (v.spec.params) |p| {
            const a = lookup(args, p.name).?;
            switch (a.value) {
                .ptr => |x| packed_args.add(x.addr),
                .i32 => |x| packed_args.add(x),
                .f32 => |x| packed_args.add(x),
                .u64 => |x| packed_args.add(x),
            }
        }
        try v.kernel.launchOn(.{ .x = grid[0], .y = grid[1], .z = grid[2] }, stream, &packed_args, .{}, &.{});
    }
};

/// The variant `Set.findRelaxed` takes among `kernels` (an aot.json's, in its order; Set.variants is loaded in the same
/// order), by index: the one captured for this launch exactly (unless `weakest`), else the fitting one that holds the
/// most of the launch's specializations (the fewest with `weakest`; ties: the first). Null: none fits (the launch is
/// refused with error.MissingTritonVariant). Host only: what a launch would run, without loading a cubin.
pub fn pick(kernels: []const KernelJson, function: []const u8, args: []const Arg, consts: []const Const, relax: Relax, weakest: bool) ?struct { index: usize, exact: bool } {
    if (!weakest) for (kernels, 0..) |k, i| {
        if (!std.mem.eql(u8, k.@"fn", function)) continue;
        if (matches(k, args, consts)) return .{ .index = i, .exact = true };
    };
    // the call site's own relaxation first (as before relax.proven), then the proven pointers too
    var r = relax;
    r.proven = &.{};
    if (fitting(kernels, function, args, consts, r, weakest)) |i| return .{ .index = i, .exact = false };
    if (relax.proven.len == 0) return null;
    if (fitting(kernels, function, args, consts, relax, weakest)) |i| return .{ .index = i, .exact = false };
    return null;
}

fn fitting(kernels: []const KernelJson, function: []const u8, args: []const Arg, consts: []const Const, relax: Relax, weakest: bool) ?usize {
    var best: ?usize = null;
    var best_held: usize = 0;
    for (kernels, 0..) |k, i| {
        if (!std.mem.eql(u8, k.@"fn", function)) continue;
        const held = fits(k, args, consts, relax) orelse continue;
        const better = if (weakest) held < best_held else held > best_held;
        if (best == null or better) {
            best = i;
            best_held = held;
        }
    }
    return best;
}

/// An aot.json's text parsed as Set.load parses it (host tests of `pick` against a captured kit).
pub fn parseSet(gpa: std.mem.Allocator, text: []const u8) !std.json.Parsed(SetJson) {
    return std.json.parseFromSlice(SetJson, gpa, text, .{ .ignore_unknown_fields = true, .allocate = .alloc_always });
}

fn lookup(args: []const Arg, name: []const u8) ?Arg {
    for (args) |a| if (std.mem.eql(u8, a.name, name)) return a;
    return null;
}

/// Whether variant `k` may run this launch though compiled for a narrower one: the same constexprs, and every
/// assumption it makes held by the launch (an argument it takes as 16-divisible is; an int it folded to 1 is 1), but a
/// pointer `relax.ptrs8` lists may be 8-aligned where `k` assumed 16. Null when it may not, else how many of the
/// launch's own specializations it shares.
fn fits(k: KernelJson, args: []const Arg, consts: []const Const, relax: Relax) ?usize {
    for (consts) |c| {
        const got = k.consts.map.get(c.name) orelse return null;
        if (c.int) |x| if (got.int == null or got.int.? != x) return null;
        if (c.f32) |x| if (got.f32 == null or got.f32.? != @as(u32, @bitCast(x))) return null;
    }
    var held: usize = 0;
    var runtime: usize = 0;
    for (args) |a| {
        const param = for (k.params) |p| {
            if (std.mem.eql(u8, p.name, a.name)) break p;
        } else null;
        switch (a.value) {
            .i32 => |x| {
                if (param == null) {
                    const got = k.consts.map.get(a.name) orelse return null;
                    if (x != 1 or got.int == null or got.int.? != 1) return null;
                    held += 1;
                    continue;
                }
                const p = param.?;
                if (!std.mem.eql(u8, p.type, "i32")) return null;
                const d16 = !p.nospec and x != 1 and @mod(x, 16) == 0;
                if (p.div16 and !d16) return null;
                if (p.div16 == d16 and (p.nospec or x != 1)) held += 1;
            },
            .ptr => |x| {
                const p = param orelse return null;
                if (!std.mem.eql(u8, p.type, x.ty)) return null;
                const d16 = x.addr % 16 == 0;
                if (p.div16 and !d16 and !(x.addr % 8 == 0 and relax.allows(k.hash, a.name))) return null;
                if (p.div16 == d16) held += 1;
            },
            .f32 => {
                const p = param orelse return null;
                if (!std.mem.eql(u8, p.type, "fp32")) return null;
                held += 1;
            },
            .u64 => |x| {
                const p = param orelse return null;
                if (!std.mem.eql(u8, p.type, "u64")) return null;
                const d16 = x % 16 == 0;
                if (p.div16 and !d16) return null;
                if (p.div16 == d16) held += 1;
            },
        }
        runtime += 1;
    }
    if (runtime != k.params.len) return null;
    return held;
}

fn matches(k: KernelJson, args: []const Arg, consts: []const Const) bool {
    for (consts) |c| {
        const got = k.consts.map.get(c.name) orelse return false;
        if (c.int) |x| if (got.int == null or got.int.? != x) return false;
        if (c.f32) |x| if (got.f32 == null or got.f32.? != @as(u32, @bitCast(x))) return false;
    }
    var runtime: usize = 0;
    for (args) |a| {
        const param = for (k.params) |p| {
            if (std.mem.eql(u8, p.name, a.name)) break p;
        } else null;
        switch (a.value) {
            .i32 => |x| {
                if (param == null) {
                    // an int Triton folded: only the value 1 is ever specialized to a constexpr
                    const got = k.consts.map.get(a.name) orelse return false;
                    if (x != 1 or got.int == null or got.int.? != 1) return false;
                    continue;
                }
                const p = param.?;
                if (!std.mem.eql(u8, p.type, "i32")) return false;
                if (!p.nospec and x == 1) return false;
                if (p.div16 != (!p.nospec and @mod(x, 16) == 0)) return false;
            },
            .ptr => |x| {
                const p = param orelse return false;
                if (!std.mem.eql(u8, p.type, x.ty) or p.div16 != (x.addr % 16 == 0)) return false;
            },
            .f32 => {
                const p = param orelse return false;
                if (!std.mem.eql(u8, p.type, "fp32")) return false;
            },
            .u64 => |x| {
                const p = param orelse return false;
                if (!std.mem.eql(u8, p.type, "u64") or p.div16 != (x % 16 == 0)) return false;
            },
        }
        runtime += 1;
    }
    return runtime == k.params.len;
}

test "fallback fit: the most specializations held, assumptions never broken; 8-aligned pointers only as listed" {
    const gpa = std.testing.allocator;
    const expect = std.testing.expect;
    const P = ParamJson;
    // _index_score's captured fp32-score shapes, cut to what matters: 5f4f (n, nt any; VIS 16) and 670f (n div 16, nt = 1)
    var p5 = [_]P{ .{ .name = "VIS", .type = "*i64", .div16 = true, .nospec = false }, .{ .name = "n", .type = "i32", .div16 = false, .nospec = false }, .{ .name = "nt", .type = "i32", .div16 = false, .nospec = false } };
    var p6 = [_]P{ .{ .name = "VIS", .type = "*i64", .div16 = true, .nospec = false }, .{ .name = "n", .type = "i32", .div16 = true, .nospec = false } };
    var k5: KernelJson = .{ .@"fn" = "f", .hash = "5f4f", .name = "f", .num_warps = 4, .params = &p5, .consts = .{} };
    defer k5.consts.map.deinit(gpa);
    try k5.consts.map.put(gpa, "KEYS", .{ .int = 0 });
    var k6: KernelJson = .{ .@"fn" = "f", .hash = "670f", .name = "f", .num_warps = 4, .params = &p6, .consts = .{} };
    defer k6.consts.map.deinit(gpa);
    try k6.consts.map.put(gpa, "KEYS", .{ .int = 0 });
    try k6.consts.map.put(gpa, "nt", .{ .int = 1 });
    const consts = [_]Const{ci("KEYS", 0)};
    const rx: Relax = .{ .ptrs8 = &.{"VIS"} };
    // the captured key: exact, and the fit holds all three
    try expect(matches(k5, &.{ ptr("VIS", "*i64", 0x1000), int("n", 146421), int("nt", 2289) }, &consts));
    try expect(fits(k5, &.{ ptr("VIS", "*i64", 0x1000), int("n", 146421), int("nt", 2289) }, &consts, .{}).? == 3);
    // 146,421 (nt = 2288, div 16): no exact variant; 5f4f fits (nt not held); 670f does not (nt is not 1)
    try expect(!matches(k5, &.{ ptr("VIS", "*i64", 0x1000), int("n", 146421), int("nt", 2288) }, &consts));
    try expect(fits(k5, &.{ ptr("VIS", "*i64", 0x1000), int("n", 146421), int("nt", 2288) }, &consts, .{}).? == 2);
    try expect(fits(k6, &.{ ptr("VIS", "*i64", 0x1000), int("n", 146421), int("nt", 2288) }, &consts, .{}) == null);
    // 145,152 (n div 16, VIS 8 bytes in): upstream's rule refuses VIS; listed in ptrs8, 5f4f fits and VIS is not held
    try expect(fits(k5, &.{ ptr("VIS", "*i64", 0x1008), int("n", 145152), int("nt", 2268) }, &consts, .{}) == null);
    try expect(fits(k5, &.{ ptr("VIS", "*i64", 0x1008), int("n", 145152), int("nt", 2268) }, &consts, rx).? == 1);
    // 4-aligned never; a variant compiled for div-16 n never takes n = 146421; another constexpr never
    try expect(fits(k5, &.{ ptr("VIS", "*i64", 0x1004), int("n", 146421), int("nt", 2289) }, &consts, rx) == null);
    try expect(fits(k6, &.{ ptr("VIS", "*i64", 0x1000), int("n", 146421), int("nt", 1) }, &consts, rx) == null);
    try expect(fits(k5, &.{ ptr("VIS", "*i64", 0x1000), int("n", 146421), int("nt", 2289) }, &.{ci("KEYS", 1)}, rx) == null);
}

test "proven 8-aligned pointers: only on the listed variant, only after every other fit, never under 8" {
    const gpa = std.testing.allocator;
    const expect = std.testing.expect;
    const P = ParamJson;
    var pa = [_]P{ .{ .name = "POS", .type = "*i64", .div16 = true, .nospec = false }, .{ .name = "n", .type = "i32", .div16 = false, .nospec = false } };
    var pb = [_]P{ .{ .name = "POS", .type = "*i64", .div16 = true, .nospec = false }, .{ .name = "n", .type = "i32", .div16 = true, .nospec = false } };
    var ks = [_]KernelJson{
        .{ .@"fn" = "f", .hash = "aaaa", .name = "f", .num_warps = 4, .params = &pa, .consts = .{} },
        .{ .@"fn" = "f", .hash = "bbbb", .name = "f", .num_warps = 4, .params = &pb, .consts = .{} },
    };
    const proven: Relax = .{ .proven = &.{.{ .hash = "aaaa", .ptrs8 = &.{"POS"} }} };
    // exact first
    try expect(pick(&ks, "f", &.{ ptr("POS", "*i64", 0x1000), int("n", 33) }, &.{}, proven, false).?.index == 0);
    // n div 16, POS aligned: bbbb exactly
    const e = pick(&ks, "f", &.{ ptr("POS", "*i64", 0x1000), int("n", 32) }, &.{}, proven, false).?;
    try expect(e.exact and e.index == 1);
    // POS 8 off: only aaaa is proven (n 32 onto its any-n), and only with the table
    try expect(pick(&ks, "f", &.{ ptr("POS", "*i64", 0x1008), int("n", 32) }, &.{}, .{}, false) == null);
    const r = pick(&ks, "f", &.{ ptr("POS", "*i64", 0x1008), int("n", 32) }, &.{}, proven, false).?;
    try expect(!r.exact and r.index == 0);
    // 4 off never
    try expect(pick(&ks, "f", &.{ ptr("POS", "*i64", 0x1004), int("n", 33) }, &.{}, proven, false) == null);
    _ = gpa;
}
