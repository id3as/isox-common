const std = @import("std");

const testTargets = [_]std.Target.Query{
    .{}, // native
};

pub fn build(b: *std.Build) void {
    const target = b.resolveTargetQuery(.{ .cpu_model = if (b.graph.host.result.cpu.arch.isX86()) .{ .explicit = &std.Target.x86.cpu.haswell } else .native });
    const optimize = b.standardOptimizeOption(.{});

    // Module (exported from this library)
    const exports = b.addModule("isox-common", .{
        .root_source_file = b.path("src/isox.zig"),
        .target = target,
        .optimize = optimize,
    });
    exports.addIncludePath(.{ .cwd_relative = "../isox_host/c_src" });

    // ISOX tests
    const test_step = b.step("test", "Run unit tests");

    const unit_tests = b.addTest(.{
        .root_source_file = b.path("src/isox_test.zig"),
        .target = target,
        .link_libc = true,
    });
    unit_tests.linkSystemLibrary("isox_host");
    unit_tests.addIncludePath(.{ .cwd_relative = "../isox_host/c_src" });
    unit_tests.addLibraryPath(.{ .cwd_relative = "../priv/isox" });

    const run_unit_tests = b.addRunArtifact(unit_tests);
    run_unit_tests.has_side_effects = true;
    run_unit_tests.step.name = "ISOX Tests";
    test_step.dependOn(&run_unit_tests.step);
}
