const std = @import("std");
const isox = @import("isox.zig");
const c = @cImport({
    @cInclude("isox_host_bindings.h");
});

test "roundtrip-int" {
    const env = isox.allocEnv();
    defer isox.freeEnv(env);
    const term = try isox.encode(i64, env, 42);
    const value = try isox.decode(i64, env, term);
    try std.testing.expect(value == 42);
}

test "roundtrip-bool" {
    const env = isox.allocEnv();
    defer isox.freeEnv(env);
    const term = try isox.encode(bool, env, true);
    const value = try isox.decode(bool, env, term);
    try std.testing.expect(value == true);
}

test "roundtrip-float" {
    const env = isox.allocEnv();
    defer isox.freeEnv(env);
    const term = try isox.encode(f64, env, 42.42);
    const value = try isox.decode(f64, env, term);
    try std.testing.expect(value == 42.42);
}

test "roundtrip-struct" {
    const TestStruct = struct {
        intField: i64,
        floatField: f64,
        boolField: bool,
    };
    const input = TestStruct{
        .intField = 1,
        .floatField = 1.2,
        .boolField = true,
    };
    const env = isox.allocEnv();
    defer isox.freeEnv(env);
    const term = try isox.encode(TestStruct, env, input);
    const decoded = try isox.decode(TestStruct, env, term);
    try std.testing.expectEqual(input, decoded);
}

test "roundtrip-tuple-struct" {
    const env = isox.allocEnv();
    defer isox.freeEnv(env);
    const term = try isox.encode(struct { i64, bool }, env, .{ 1, true });
    _ = try isox.decode(struct { i64, bool }, env, term);
    // try std.testing.expect(value == .{1, true});
}

test "roundtrip-union" {
    const TestUnion = union(enum) {
        value1: i64,
        Value2: f64,
    };
    const env = isox.allocEnv();
    defer isox.freeEnv(env);
    const input1 = TestUnion{ .value1 = 5 };
    const term1 = try isox.encode(TestUnion, env, input1);
    const decoded1 = try isox.decode(TestUnion, env, term1);
    try std.testing.expectEqual(input1, decoded1);

    const input2 = TestUnion{ .Value2 = 1.2 };
    const term2 = try isox.encode(TestUnion, env, input2);
    const decoded2 = try isox.decode(TestUnion, env, term2);
    try std.testing.expectEqual(input2, decoded2);
}

test "roundtrip-optional" {
    const env = isox.allocEnv();
    defer isox.freeEnv(env);
    const input1: ?i64 = 5;
    const term1 = try isox.encode(?i64, env, input1);
    const decoded1 = try isox.decode(?i64, env, term1);
    try std.testing.expectEqual(input1, decoded1);

    const input2: ?i64 = null;
    const term2 = try isox.encode(?i64, env, input2);
    const decoded2 = try isox.decode(?i64, env, term2);
    try std.testing.expectEqual(input2, decoded2);
}

test "roundtrip-enum" {
    const TestEnum = enum {
        Enum1,
        enum2,
    };
    const env = isox.allocEnv();
    defer isox.freeEnv(env);
    const input1 = TestEnum.Enum1;
    const term1 = try isox.encode(TestEnum, env, input1);
    const decoded1 = try isox.decode(TestEnum, env, term1);
    try std.testing.expectEqual(input1, decoded1);

    const input2 = TestEnum.enum2;
    const term2 = try isox.encode(TestEnum, env, input2);
    const decoded2 = try isox.decode(TestEnum, env, term2);
    try std.testing.expectEqual(input2, decoded2);
}

test "roundtrip-slice" {
    const env = isox.allocEnv();
    defer isox.freeEnv(env);
    const input = [_]i64{ 1, 2, 3, 4, 5 };
    const slice: []i64 = @constCast(&input);
    const term = try isox.encode([]i64, env, slice);
    const decoded = try isox.decode([]i64, env, term);
    defer env.allocator.free(decoded);
    try std.testing.expectEqualSlices(i64, slice, decoded);
}

test "roundtrip-single-pointer" {
    const env = isox.allocEnv();
    defer isox.freeEnv(env);
    const input: i64 = 7;
    const term = try isox.encode(*i64, env, @constCast(&input));
    const decoded = try isox.decode(*i64, env, term);
    defer env.allocator.destroy(decoded);
    try std.testing.expectEqual(input, decoded.*);
}

test "roundtrip-term" {
    const env = isox.allocEnv();
    defer isox.freeEnv(env);
    const input = try isox.encode(i64, env, 7);
    const term = try isox.encode(isox.Term, env, input);
    const decoded = try isox.decode(isox.Term, env, term);
    try std.testing.expectEqual(input, decoded);
}
