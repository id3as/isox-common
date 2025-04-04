const std = @import("std");
const c = @cImport({
    @cInclude("isox_host_bindings.h");
});
const builtin = @import("builtin");

pub const std_options = .{
    .logFn = log,
};

pub const Unit = enum { isoxUnit };

pub const Term = *const c.Term;

pub fn abs(x: i64) i64 {
    return if (x < 0) -x else x;
}

pub fn rationalFromInt(val: i64) !std.math.big.Rational {
    var zero = try std.math.big.Rational.init(std.heap.c_allocator);
    try zero.setInt(val);
    return zero;
}

pub fn rationalFromRatio(num: i64, den: i64) !std.math.big.Rational {
    var val = try std.math.big.Rational.init(std.heap.c_allocator);
    try val.setRatio(num, den);
    return val;
}

pub fn addRational(a: std.math.big.Rational, b: std.math.big.Rational) !std.math.big.Rational {
    var sum = try std.math.big.Rational.init(std.heap.c_allocator);
    try std.math.big.Rational.add(&sum, a, b);
    return sum;
}

pub fn mulRational(a: std.math.big.Rational, b: i64) !std.math.big.Rational {
    var multiplier = try rationalFromInt(b);
    defer multiplier.deinit();

    var product = try std.math.big.Rational.init(std.heap.c_allocator);
    try std.math.big.Rational.mul(&product, a, multiplier);
    return product;
}

// Generate timestamps based on counting samples, allowing for dropped frames (up to user to tell us how many (in total) have been dropped)
// For example on SDI with 48kHz audio and a frame rate of 24fps, you'd create two, one for the video and one for the audio:
//    videoTracker = TimestampTracker.new(24, 1, 24, 1);
//    audioTracker = TimestampTracker.new(48000, 1, 24, 1);
// then as frames flow you'd call nextTimestamp:
//   audioPts = audioTracker.nextTimestamp(nbSamples, nbDroppedFrames);
//   videoPts = videoTracker.nextTimestamp(1, nbDroppedFrames);
pub const TimestampTracker = struct {
    duration: std.math.big.Rational,

    pub fn new() !TimestampTracker {
        return TimestampTracker{
            .duration = try rationalFromInt(0),
        };
    }

    pub fn deinit(self: *@This()) void {
        self.duration.deinit();
    }

    pub fn nextTimestamp(
        self: *@This(),
        sampleRateNum: i64,
        sampleRateDen: i64,
        frameRateNum: i64,
        frameRateDen: i64,
        nbSamples: i64,
        nbDroppedFrames: i64,
    ) !Pts {
        var sampleDuration = try rationalFromRatio(sampleRateDen * nbSamples, sampleRateNum);
        defer sampleDuration.deinit();

        if (nbDroppedFrames > 0) {
            var frameDuration = try rationalFromRatio(frameRateDen, frameRateNum);
            defer frameDuration.deinit();

            var timestampForDrops = try mulRational(frameDuration, nbDroppedFrames);
            defer timestampForDrops.deinit();

            var timestamp = try addRational(self.duration, timestampForDrops);
            defer timestamp.deinit();

            const pts = .{
                .fst = try timestamp.p.to(i64),
                .snd = try timestamp.q.to(i64),
            };

            self.duration.deinit();

            self.duration = try addRational(timestamp, sampleDuration);

            return pts;
        } else {
            const pts = .{
                .fst = try self.duration.p.to(i64),
                .snd = try self.duration.q.to(i64),
            };

            const newDuration = try addRational(self.duration, sampleDuration);
            self.duration.deinit();
            self.duration = newDuration;

            return pts;
        }
    }
};

pub const DropFrameDetector = struct {
    timescale: i64,
    t0: i64,
    totalDuration: std.math.big.Rational,

    pub fn new(timescale: i64) !DropFrameDetector {
        return DropFrameDetector{
            .timescale = timescale,
            .t0 = 0,
            .totalDuration = try rationalFromInt(0),
        };
    }

    pub fn deinit(self: *@This()) void {
        self.totalDuration.deinit();
    }

    // Parameters are:
    // timestamp - timestamp of this frame
    // durationN / durationD - duration of this frame
    pub fn updateTiming(self: *@This(), timestamp: i64, durationN: c_int, durationD: c_int) !i64 {
        if (self.t0 == 0) {
            self.t0 = timestamp;
        }

        // Get this frame duration into a rational
        var frameDuration = try rationalFromRatio(durationN, durationD);
        defer frameDuration.deinit();

        // Timestamp delta should approximately equal duration, math in milliseconds
        const realtime: f64 = @as(f64, @floatFromInt(timestamp - self.t0)) / @as(f64, @floatFromInt(self.timescale));
        const calculated: f64 = try self.totalDuration.toFloat(f64);
        const realtimeToCalculatedDelta = realtime - calculated;
        const halfFrameDuration = try frameDuration.toFloat(f64) / 2;

        // std.log.warn("REAL TS {}, CALC TS {} DUR-N {} DUR-D {}", .{
        //     @divTrunc(timestamp - self.t0, 10000),
        //     @as(i64, @intFromFloat(@trunc(calculated * 1000))),
        //     durationN,
        //     durationD,
        // });

        if (realtimeToCalculatedDelta > (halfFrameDuration + 0.001)) {
            // Realtime has moved forward further than we'd expect, we must have dropped frames
            const missingFrames: i64 = @intFromFloat(@round(realtimeToCalculatedDelta / try frameDuration.toFloat(f64)));

            std.log.warn("{} Dropped frame(s) detected", .{missingFrames});

            // std.log.warn("REALTIME {}, CALC {}, DURATION {}/{}/{}, DELTA {} ", .{
            //     realtime,
            //     calculated,
            //     durationN,
            //     durationD,
            //     try frameDuration.toFloat(f64),
            //     realtimeToCalculatedDelta,
            // });

            // Calculate the duration of these missing frames
            var missingDuration = try mulRational(frameDuration, missingFrames);
            defer missingDuration.deinit();

            // And add it to our current total duration to get the new total duration
            self.totalDuration = try addRational(self.totalDuration, missingDuration);

            // missing frames *includes* this frame - we want it in total duration. So return missingFrames - 1...
            return missingFrames - 1;
        } else {
            self.totalDuration = try addRational(self.totalDuration, frameDuration);
            return 0;
        }
    }
};

pub fn HashMap(
    comptime K: type,
    comptime V: type,
    comptime HashContext: type,
    comptime max_load_percentage: u64,
) type {
    return struct {
        const Self = @This();
        map: std.HashMap(
            K,
            V,
            HashContext,
            max_load_percentage,
        ),
        pub fn init(allocator: std.mem.Allocator) Self {
            return .{
                .map = std.HashMap(K, V, HashContext, max_load_percentage).init(allocator),
            };
        }
        pub fn deinit(self: *Self) void {
            self.map.deinit();
        }
        pub fn put(self: *Self, key: K, value: V) std.mem.Allocator.Error!void {
            try self.map.put(key, value);
        }
        pub fn get(self: *Self, key: K) ?V {
            return self.map.get(key);
        }
        pub fn encode_(self: *const @This(), env: Env) !Term {
            var term = c.isox_make_map(env.env) orelse return IsoxError.NullTerm;
            var iterator = self.map.iterator();
            while (iterator.next()) |entry| {
                const key = try encode(K, env, entry.key_ptr.*);
                const value = try encode(V, env, entry.value_ptr.*);
                term = c.isox_add_map_entry(env.env, key, value, term) orelse return IsoxError.NullTerm;
            }
            return term;
        }
    };
}

pub const IsoxError = error{
    UnsupportedTermType,
    UnsupportedDecodeType,
    UnsupportedEncodeType,
    UnexpectedTermType,
    UnsupportedMapKey,
    InvalidArguments,
    InvalidAnyType,
    KeyNotFound,
    NoStackTraceAvailable,
    SendFailed,
    UnknownUnionTagType,
    UnknownUnionTag,
    UnknownEnumTag,
    UnsupportedPointerType,
    NullTerm,
    InvalidSubBinarySize,
    NotNamedChannel,
};

pub const NoStackTrace: [:0]const u8 = "[no stack trace available]";

pub const ReadError = IsoxError || std.mem.Allocator.Error;

pub const IsoxExtension = c.IsoxFunctionTable;

pub const QueryReturn = ?Term;

pub const CreateInstanceReturn = struct { IsoxResource, ?Term };

pub const QueryInstanceReturn = ?Term;

pub const UpdateInstanceReturn = ?Term;

pub const DestroyInstanceReturn = void;

const QueryResponse = struct { IsoxAtom, Term };

const CreateInstanceResponse = struct { IsoxAtom, IsoxResource, Term };

const QueryInstanceResponse = struct { IsoxAtom, Term };

const UpdateInstanceResponse = struct { IsoxAtom, Term };

const DestroyInstanceResponse = IsoxAtom;

pub const FixedFrameRate = Tagged2Tuple(i64, "fixedFrameRate");

pub const FrameRate = union(enum) {
    fixed: FixedFrameRate,
    variable: Unit,
};

pub const PicStruct = enum {
    progressive,
    tFF,
    bFF,
};

pub const VideoPlane = struct {
    stride: u64,
    binary: IsoxBinary,

    const FfiShape = struct { IsoxAtom, i64, IsoxBinary };
    pub fn encode_(self: *const @This(), env: Env) !Term {
        const ffi = FfiShape{
            IsoxAtom.newLiteral("tuple"),
            @intCast(self.*.stride),
            self.*.binary,
        };
        return encode(FfiShape, env, ffi);
    }

    pub fn decode_(env: Env, term: Term) !@This() {
        const ffi = try decode(FfiShape, env, term);
        defer ffi[0].deinit();
        return VideoPlane{ .stride = @intCast(ffi[1]), .binary = ffi[2] };
    }
};

pub const Pts = Tagged2Tuple(i64, "ratio");

pub const Dts = Tagged2Tuple(i64, "ratio");

pub fn ptsToTimestamp(pts: Pts, timescale: i64) i64 {
    return @divTrunc(pts.fst * timescale, pts.snd);
}

pub const Duration = Tagged2Tuple(i64, "ratio");

pub const EncodeMetadata = struct {};

pub const VideoFrameData = struct {
    planes: []const VideoPlane,
    bufferHeight: u64,
    picStruct: PicStruct,
    originalMetadata: ?EncodeMetadata,

    pub fn deinit(self: *const @This(), allocator: std.mem.Allocator) void {
        allocator.free(self.planes);
    }
};

pub const AudioFrameData = struct {
    planar: bool,
    nbSamples: i64,
    channels: []const IsoxBinary,

    const FfiShape = struct { bool, i64, []const IsoxBinary };
    pub fn encode_(self: *const @This(), env: Env) !Term {
        const ffi = FfiShape{
            self.*.planar,
            self.*.nbSamples,
            self.*.channels,
        };
        return encode(FfiShape, env, ffi);
    }

    pub fn decode_(env: Env, term: Term) !@This() {
        const ffi = try decode(FfiShape, env, term);
        return AudioFrameData{ .planar = ffi[0], .nbSamples = ffi[1], .channels = ffi[2] };
    }
};

pub const StreamKey = struct {
    sourceName: IsoxString,
    programNumber: i64,
    streamId: i64,
    renditionName: IsoxString,

    pub const HashContext = struct {
        pub fn hash(_: HashContext, key: StreamKey) u64 {
            var h = std.hash.Wyhash.init(0);
            h.update(key.sourceName.string);
            h.update(&std.mem.toBytes(key.programNumber));
            h.update(&std.mem.toBytes(key.streamId));
            h.update(key.renditionName.string);
            return h.final();
        }

        pub fn eql(_: HashContext, lhs: StreamKey, rhs: StreamKey) bool {
            return std.mem.eql(u8, lhs.sourceName.string, rhs.sourceName.string) and
                lhs.programNumber == rhs.programNumber and
                lhs.streamId == rhs.streamId and
                std.mem.eql(u8, lhs.renditionName.string, rhs.renditionName.string);
        }
    };

    pub fn copy(self: *const @This(), env: Env) !StreamKey {
        return StreamKey{
            .sourceName = try self.sourceName.copy(env),
            .programNumber = self.programNumber,
            .streamId = self.streamId,
            .renditionName = try self.renditionName.copy(env),
        };
    }

    pub fn deinit(self: *const @This()) void {
        self.sourceName.deinit();
        self.renditionName.deinit();
    }
};

pub const SampleFormat = enum {
    S16,
    S16p,
    Flt,
    Fltp,
};

pub const NamedLayout = enum {
    Mono,
    Stereo,
    Surround,
    FourDotZero,
    FiveDotZero,
    FiveDotOne,
    SevenDotOne,
    FiveDotOneDotFour,
    SevenDotOneDotFour,
    pub fn numChannels(self: NamedLayout) u32 {
        return switch (self) {
            .Mono => 1,
            .Stereo => 2,
            .Surround => 3,
            .FourDotZero => 4,
            .FiveDotZero => 5,
            .FiveDotOne => 6,
            .SevenDotOne => 8,
            .FiveDotOneDotFour => 10,
            .SevenDotOneDotFour => 12,
        };
    }
    pub fn fromNumChannels(nbChannels: u32) !NamedLayout {
        return switch (nbChannels) {
            1 => .Mono,
            2 => .Stereo,
            3 => .Surround,
            4 => .FourDotZero,
            5 => .FiveDotZero,
            6 => .FiveDotOne,
            8 => .SevenDotOne,
            10 => .FiveDotOneDotFour,
            12 => .SevenDotOneDotFour,
            else => IsoxError.NotNamedChannel,
        };
    }
};

pub const ChannelName = enum {
    L,
    R,
    C,
    LFE,
    Ls,
    Rs,
    Lc,
    Rc,
    Lsr,
    Rsr,
    Cs,
    Lsd,
    Rsd,
    Lss,
    Rss,
    Lw,
    Rw,
    Lv,
    Rv,
    Cv,
    Lvr,
    Rvr,
    Cvr,
    Lvss,
    Rvss,
    Ts,
    LFE2,
    Lb,
    Rb,
    Cb,
    Lvs,
    Rvs,
    LFE3,
    Leos,
    Reos,
    Hwbcal,
    Hwbcar,
    Lbs,
    Rbs,
    Unknown,
};

pub const ChannelLayout = union(enum) {
    NamedLayout: NamedLayout,
    SingleChannel: ChannelName,
    MultiChannel: []ChannelName,

    pub fn numChannels(self: ChannelLayout) u32 {
        return switch (self) {
            .NamedLayout => |named| named.numChannels(),
            .SingleChannel => |_| 1,
            .MultiChannel => |list| @intCast(list.len),
        };
    }

    pub fn fromNumChannels(allocator: std.mem.Allocator, nbChannels: u32) !ChannelLayout {
        if (NamedLayout.fromNumChannels(nbChannels)) |namedLayout| {
            return ChannelLayout{ .NamedLayout = namedLayout };
        } else |_| {
            const channels = try allocator.alloc(ChannelName, @intCast(nbChannels));
            for (channels) |*channel| {
                channel.* = ChannelName.Unknown;
            }
            return ChannelLayout{ .MultiChannel = channels };
        }
    }

    pub fn deinit(self: *const ChannelLayout, allocator: std.mem.Allocator) void {
        switch (self.*) {
            .MultiChannel => |list| allocator.free(list),
            else => {},
        }
    }
};

pub const PixelFormat = enum {
    Yuv420p,
    Yuv422p,
    Yuv444p,

    Yuva420p,
    Yuva422p,
    Yuva444p,

    Yuv420p10le,
    Yuv422p10le,
    Yuv444p10le,

    Nv12,
    Uyvy,

    Rgb24,
    Rgba,
    Argb,
    Bgra,

    pub fn chromaInfo(self: PixelFormat) ChromaInfo {
        return switch (self) {
            .Yuv420p => ChromaInfo{
                .componentBitDepth = .EightBit,
                .sizeFactor = 1.5,
                .planes = &[_]ScalingFactor{
                    ScalingFactor{ .widthFactor = 1.0, .heightFactor = 1.0 },
                    ScalingFactor{ .widthFactor = 0.5, .heightFactor = 0.5 },
                    ScalingFactor{ .widthFactor = 0.5, .heightFactor = 0.5 },
                },
            },
            .Yuv422p => ChromaInfo{
                .componentBitDepth = .EightBit,
                .sizeFactor = 2.0,
                .planes = &[_]ScalingFactor{
                    ScalingFactor{ .widthFactor = 1.0, .heightFactor = 1.0 },
                    ScalingFactor{ .widthFactor = 0.5, .heightFactor = 1.0 },
                    ScalingFactor{ .widthFactor = 0.5, .heightFactor = 1.0 },
                },
            },
            .Yuv444p => ChromaInfo{
                .componentBitDepth = .EightBit,
                .sizeFactor = 3.0,
                .planes = &[_]ScalingFactor{
                    ScalingFactor{ .widthFactor = 1.0, .heightFactor = 1.0 },
                    ScalingFactor{ .widthFactor = 1.0, .heightFactor = 1.0 },
                    ScalingFactor{ .widthFactor = 1.0, .heightFactor = 1.0 },
                },
            },
            .Yuva420p => ChromaInfo{
                .componentBitDepth = .EightBit,
                .sizeFactor = 2.5,
                .planes = &[_]ScalingFactor{
                    ScalingFactor{ .widthFactor = 1.0, .heightFactor = 1.0 },
                    ScalingFactor{ .widthFactor = 0.5, .heightFactor = 0.5 },
                    ScalingFactor{ .widthFactor = 0.5, .heightFactor = 0.5 },
                    ScalingFactor{ .widthFactor = 1.0, .heightFactor = 1.0 },
                },
            },
            .Yuva422p => ChromaInfo{
                .componentBitDepth = .EightBit,
                .sizeFactor = 3.0,
                .planes = &[_]ScalingFactor{
                    ScalingFactor{ .widthFactor = 1.0, .heightFactor = 1.0 },
                    ScalingFactor{ .widthFactor = 0.5, .heightFactor = 1.0 },
                    ScalingFactor{ .widthFactor = 0.5, .heightFactor = 1.0 },
                    ScalingFactor{ .widthFactor = 1.0, .heightFactor = 1.0 },
                },
            },
            .Yuva444p => ChromaInfo{
                .componentBitDepth = .EightBit,
                .sizeFactor = 4.0,
                .planes = &[_]ScalingFactor{
                    ScalingFactor{ .widthFactor = 1.0, .heightFactor = 1.0 },
                    ScalingFactor{ .widthFactor = 1.0, .heightFactor = 1.0 },
                    ScalingFactor{ .widthFactor = 1.0, .heightFactor = 1.0 },
                    ScalingFactor{ .widthFactor = 1.0, .heightFactor = 1.0 },
                },
            },
            .Yuv420p10le => ChromaInfo{
                .componentBitDepth = .TenBit,
                .sizeFactor = 1.5,
                .planes = &[_]ScalingFactor{
                    ScalingFactor{ .widthFactor = 1.0, .heightFactor = 1.0 },
                    ScalingFactor{ .widthFactor = 0.5, .heightFactor = 1.0 },
                    ScalingFactor{ .widthFactor = 0.5, .heightFactor = 1.0 },
                },
            },
            .Yuv422p10le => ChromaInfo{
                .componentBitDepth = .TenBit,
                .sizeFactor = 2.0,
                .planes = &[_]ScalingFactor{
                    ScalingFactor{ .widthFactor = 1.0, .heightFactor = 1.0 },
                    ScalingFactor{ .widthFactor = 0.5, .heightFactor = 1.0 },
                    ScalingFactor{ .widthFactor = 0.5, .heightFactor = 1.0 },
                },
            },
            .Yuv444p10le => ChromaInfo{
                .componentBitDepth = .TenBit,
                .sizeFactor = 3.0,
                .planes = &[_]ScalingFactor{
                    ScalingFactor{ .widthFactor = 1.0, .heightFactor = 1.0 },
                    ScalingFactor{ .widthFactor = 1.0, .heightFactor = 1.0 },
                    ScalingFactor{ .widthFactor = 1.0, .heightFactor = 1.0 },
                },
            },
            .Nv12 => ChromaInfo{
                .componentBitDepth = .EightBit,
                .sizeFactor = 1.5,
                .planes = &[_]ScalingFactor{
                    ScalingFactor{ .widthFactor = 1.0, .heightFactor = 1.0 },
                    ScalingFactor{ .widthFactor = 2.0, .heightFactor = 0.25 },
                },
            },
            .Uyvy => ChromaInfo{
                .componentBitDepth = .EightBit,
                .sizeFactor = 2.0,
                .planes = &[_]ScalingFactor{
                    ScalingFactor{ .widthFactor = 2.0, .heightFactor = 1.0 },
                },
            },
            .Rgb24 => ChromaInfo{
                .componentBitDepth = .EightBit,
                .sizeFactor = 3.0,
                .planes = &[_]ScalingFactor{
                    ScalingFactor{ .widthFactor = 3.0, .heightFactor = 1.0 },
                },
            },
            .Rgba => ChromaInfo{
                .componentBitDepth = .EightBit,
                .sizeFactor = 4.0,
                .planes = &[_]ScalingFactor{
                    ScalingFactor{ .widthFactor = 4.0, .heightFactor = 1.0 },
                },
            },
            .Argb => ChromaInfo{
                .componentBitDepth = .EightBit,
                .sizeFactor = 4.0,
                .planes = &[_]ScalingFactor{
                    ScalingFactor{ .widthFactor = 4.0, .heightFactor = 1.0 },
                },
            },
            .Bgra => ChromaInfo{
                .componentBitDepth = .EightBit,
                .sizeFactor = 4.0,
                .planes = &[_]ScalingFactor{
                    ScalingFactor{ .widthFactor = 4.0, .heightFactor = 1.0 },
                },
            },
        };
    }
};

pub const ScalingFactor = struct {
    widthFactor: f32,
    heightFactor: f32,
};

pub const ChromaInfo = struct {
    componentBitDepth: BitDepth,
    sizeFactor: f32,
    planes: []const ScalingFactor,
};

pub const Resolution = struct {
    width: u64,
    height: u64,
};

pub const PixelAspectRatio = Tagged2Tuple(i64, "ratio");

pub const AudioCodec = enum {
    Aac,
    Opus,
};

pub const BitDepth = enum {
    EightBit,
    TenBit,
};

pub const H264Profile = enum {
    Baseline,
    Main,
    Extended,
    High,
    High10,
    High422,
    High444,
};

pub const VideoCodec = enum {
    H264,
    HEVC,
};

pub const RawAudioMetadata = struct {
    sampleRate: i64,
    sampleFormat: SampleFormat,
    channelLayout: ChannelLayout,
};

// Colour information coded via Coding-independant code points (H.273 / ISO/IEC 23091-2)
pub const ColourInfo = struct {
    primaries: i64, // 8.1 ColourPrimaries
    transferCharacteristics: i64, // 8.2 TransferCharacteristics
    matrixCoefficients: i64, // 8.3 MatrixCoefficients
    fullRange: bool, // 8.3 VideoFullRangeFlag
    chromaLocation: i64, // 8.7 Chroma420SampleLocType
};

pub const RawVideoMetadata = struct {
    pixelFormat: PixelFormat,
    resolution: Resolution,
    frameRate: FrameRate,
    pixelAspectRatio: PixelAspectRatio,
    colourInfo: ?ColourInfo,
};

pub const CompressedAudioMetadata = struct {
    codec: AudioCodec,
    sampleRate: i64,
    channelLayout: ChannelLayout,
};

pub const CompressedVideoMetadata = struct {
    codec: VideoCodec,
    bitDepth: BitDepth,
    frameRate: FrameRate,
    resolution: Resolution,
    pixelAspectRatio: PixelAspectRatio,
    bitrate: ?i64,
};

pub const Metadata = union(enum) {
    RawAudio: RawAudioMetadata,
    RawVideo: RawVideoMetadata,
    CompressedAudio: CompressedAudioMetadata,
    CompressedVideo: CompressedVideoMetadata,
};

pub const Context = HashMap(StreamKey, Metadata, StreamKey.HashContext, std.hash_map.default_max_load_percentage);

pub const RawVideoFrame = struct {
    key: StreamKey,
    pts: Pts,
    duration: ?Duration,
    frameData: VideoFrameData,

    pub fn deinit(self: *const @This(), allocator: std.mem.Allocator) void {
        self.key.deinit();
        self.frameData.deinit(allocator);
    }
};

pub const RawAudioFrame = struct {
    key: StreamKey,
    pts: Pts,
    duration: Duration,
    frameData: AudioFrameData,
};

pub const H264SliceType = enum {
    I,
    P,
    B,
    Si,
    Sp,
};

pub const H264SeiMessage = struct {
    payloadType: i64,
    payload: IsoxBinary,
};

pub const H264VideoFrameMetadata = struct {
    sliceTypes: []H264SliceType,
    seiMessages: []H264SeiMessage,
};

pub const HEVCSliceType = enum {
    I,
    P,
    B,
};

pub const HEVCSeiMessage = struct {
    payloadType: i64,
    payload: IsoxBinary,
};

pub const HEVCVideoFrameMetadata = struct {
    isIdr: bool,
    sliceTypes: []HEVCSliceType,
    seiMessages: []HEVCSeiMessage,
};

pub const CompressedVideoFrameMetadataTag = enum {
    H264PayloadMetadata,
    HEVCPayloadMetadata,
};

pub const CompressedVideoFrameMetadata = union(CompressedVideoFrameMetadataTag) {
    H264PayloadMetadata: H264VideoFrameMetadata,
    HEVCPayloadMetadata: HEVCVideoFrameMetadata,
};

pub const FrameTypeHint = enum {
    HintIDR,
    HintAny,
};

pub const CompressedVideoFrame = struct {
    key: StreamKey,
    dts: Dts,
    pts: Pts,
    data: IsoxBinary,
    picStruct: PicStruct,
    frameType: ?FrameTypeHint,
    duration: ?Duration,
    metadata: ?CompressedVideoFrameMetadata,

    pub fn deinit(self: *const @This()) void {
        self.key.deinit();
    }
};

pub const IsoxResourceError = struct {
    errorText: IsoxString,
    errorCode: i64,
    description: IsoxString,
    detail: ?Term,
};

pub const Message = union(enum) {
    ContextChange: Context,
    RawAudioFrame: RawAudioFrame,
    RawVideoFrame: RawVideoFrame,
    CompressedVideoFrame: CompressedVideoFrame,
    Status: IsoxAtom,
    Error: IsoxResourceError,
    Other: Term,
};

pub const ExtensionMessage = struct {
    instanceRef: IsoxResourceRef,
    message: Message,
};

pub const Env = struct {
    const Self = @This();
    env: ?*c.Env,
    allocator: std.mem.Allocator,

    pub fn privData(self: *const Self) ?*anyopaque {
        return c.isox_priv_data(self.env);
    }

    pub fn setPrivData(self: *const Self, data: *anyopaque) void {
        if (!builtin.is_test) @compileError("setPrivData is only available in test mode");
        c.isox_set_priv_data(self.env, data);
    }
};

fn unexpectedTypeWarning(comptime expected: [:0]const u8, actual: Term) void {
    std.log.warn("Expected '{s}', got '{s}'", .{ expected, c.isox_term_type_to_string(c.isox_term_type(actual)) });
}

pub fn Tagged2Tuple(comptime T: type, comptime tag: [:0]const u8) type {
    const FfiTagged2Tuple = struct { IsoxAtom, i64, i64 };

    return struct {
        fst: T,
        snd: T,
        pub fn encode_(self: *const @This(), env: Env) !Term {
            var ratio = try rationalFromRatio(self.*.fst, self.*.snd);
            defer ratio.deinit();

            const ffi = FfiTagged2Tuple{
                IsoxAtom.newLiteral(tag),
                try ratio.p.to(i64),
                try ratio.q.to(i64),
            };
            return encode(FfiTagged2Tuple, env, ffi);
        }
        pub fn decode_(env: Env, term: Term) !@This() {
            const ffi = try decode(FfiTagged2Tuple, env, term);
            defer ffi[0].deinit();
            return .{ .fst = ffi[1], .snd = ffi[2] };
        }
        pub fn toFloat(self: *const @This(), factor: f64) f64 {
            return (@as(f64, @floatFromInt(self.*.fst)) * factor) / @as(f64, @floatFromInt(self.*.snd));
        }
        pub fn toInt(self: *const @This(), factor: f64) i64 {
            return @intFromFloat(@round((@as(f64, @floatFromInt(self.*.fst)) * factor) / @as(f64, @floatFromInt(self.*.snd))));
        }
    };
}

pub fn TaggedValue(comptime T: type) type {
    return struct { IsoxAtom, T };
}

const TaggedEnum =
    struct { IsoxAtom };

pub const IsoxAtom = struct {
    atom: [:0]const u8,
    allocator: ?std.mem.Allocator,

    pub fn newLiteral(comptime atom: [:0]const u8) IsoxAtom {
        return IsoxAtom{
            .atom = atom,
            .allocator = null,
        };
    }

    /// Create a new IsoxAtom that just references the passed in slice.  The caller
    /// must ensure that the slice outlives the atom...
    pub fn newNoAlloc(atom: [:0]const u8) IsoxAtom {
        return IsoxAtom{ .atom = atom, .allocator = null };
    }

    pub fn new(env: Env, atom: [:0]const u8) !IsoxAtom {
        const buffer = try env.allocator.allocSentinel(u8, atom.len, 0);
        @memcpy(buffer, atom);
        return IsoxAtom{ .atom = buffer, .allocator = env.allocator };
    }

    pub fn ok() IsoxAtom {
        return IsoxAtom{ .atom = "ok", .allocator = null };
    }

    pub fn err() IsoxAtom {
        return IsoxAtom{ .atom = "error", .allocator = null };
    }

    pub fn undef() IsoxAtom {
        return IsoxAtom{ .atom = "undefined", .allocator = null };
    }

    pub fn deinit(self: *const @This()) void {
        if (self.*.allocator) |allocator| {
            allocator.free(self.*.atom);
        }
    }

    pub fn encode_(self: *const @This(), env: Env) !Term {
        return c.isox_make_atom(env.env, self.*.atom) orelse return IsoxError.NullTerm;
    }

    pub fn decode_(env: Env, term: Term) !@This() {
        return IsoxAtom{ .atom = try readAtom_(env, term), .allocator = env.allocator };
    }
};

pub const IsoxString = struct {
    string: [:0]const u8,
    allocator: ?std.mem.Allocator,

    pub fn newLiteral(comptime string: [:0]const u8) IsoxString {
        return IsoxString{ .string = string, .allocator = null };
    }

    /// Create a new IsoxString that just references the passed in slice.  The caller
    /// must ensure that the slice outlives the string...
    pub fn newNoAlloc(string: [:0]const u8) IsoxString {
        return IsoxString{ .string = string, .allocator = null };
    }

    pub fn new(env: Env, string: [:0]const u8) !IsoxString {
        const buffer = try env.allocator.allocSentinel(u8, string.len, 0);
        @memcpy(buffer, string);
        return IsoxString{ .string = buffer, .allocator = env.allocator };
    }

    pub fn copy(self: *const @This(), env: Env) !IsoxString {
        if (self.*.allocator) |_| {
            const buffer = try env.allocator.allocSentinel(u8, self.string.len, 0);
            @memcpy(buffer, self.string);

            return IsoxString{ .string = buffer, .allocator = env.allocator };
        } else {
            return IsoxString{ .string = self.string, .allocator = null };
        }
    }

    pub fn deinit(self: *const @This()) void {
        if (self.*.allocator) |allocator| {
            allocator.free(self.*.string);
        }
    }

    pub fn encode_(self: *const @This(), env: Env) !Term {
        return c.isox_make_string(env.env, self.*.string) orelse return IsoxError.NullTerm;
    }

    pub fn decode_(env: Env, term: Term) !@This() {
        return IsoxString{ .string = try readString_(env, term), .allocator = env.allocator };
    }
};

pub const IsoxBinary = struct {
    term: Term,
    data: []u8,

    pub fn t(self: *const @This()) c_uint {
        return c.isox_term_type(self.term);
    }

    pub fn new(env: Env, size: usize) !IsoxBinary {
        var term: ?*const c.Term = undefined;
        const data: [*]u8 = @ptrCast(c.isox_make_binary(@constCast(env.env), size, &term) orelse return IsoxError.NullTerm);
        return IsoxBinary{ .term = term orelse return IsoxError.NullTerm, .data = data[0..size] };
    }

    pub fn newSubBinary(env: Env, binary: IsoxBinary, start: usize, length: usize) !IsoxBinary {
        if (start + length > binary.data.len) {
            return IsoxError.InvalidSubBinarySize;
        }
        const term: *const c.Term = c.isox_make_sub_binary(@constCast(env.env), binary.term, start, length) orelse return IsoxError.NullTerm;
        return IsoxBinary{ .term = term, .data = binary.data[start .. start + length] };
    }

    pub fn encode_(self: *const @This(), _: Env) !Term {
        return self.term;
    }

    pub fn decode_(_: Env, term: Term) !@This() {
        var result: ?[*]u8 = undefined;
        var size: usize = 0;

        if (c.isox_read_binary(term, @ptrCast(&result), &size)) {
            if (result) |res| {
                return IsoxBinary{ .term = term, .data = res[0..size] };
            } else {
                return IsoxError.UnexpectedTermType;
            }
        } else {
            unexpectedTypeWarning("binary", term);
            return IsoxError.UnexpectedTermType;
        }
    }
};

pub const IsoxResourceRef = struct {
    resourceRef: c.ResourceRefData,

    pub fn new(resource: IsoxResource) IsoxResourceRef {
        return IsoxResourceRef{ .resourceRef = c.isox_resource_to_resource_ref(resource.term) };
    }

    pub fn encode_(self: *const @This(), env: Env) !Term {
        return c.isox_make_resource_ref(env.env, self.*.resourceRef) orelse return IsoxError.NullTerm;
    }
};

pub const IsoxResource = struct {
    term: Term,
    resourceType: ResourceType,
    obj: *anyopaque,

    pub fn new(env: Env, resourceType: ResourceType, obj: *anyopaque) !IsoxResource {
        return IsoxResource{ .term = c.isox_make_resource(@constCast(env.env), resourceType, obj) orelse return IsoxError.NullTerm, .resourceType = resourceType, .obj = obj };
    }

    pub fn encode_(self: *const @This(), _: Env) !Term {
        return self.*.term;
    }

    pub fn decode_(env: Env, term: Term) !@This() {
        return readResource_(env, term);
    }
};

pub const IsoxPid = if (builtin.is_test)
    struct {
        var next_id: i64 = 0;
        id: i64,
        term: Term,

        pub fn new(env: Env) !IsoxPid {
            next_id += 1;
            return IsoxPid{ .id = next_id, .term = try makeInt64(env, next_id) };
        }
    }
else
    struct {
        term: Term,

        pub fn encode_(self: *const @This(), _: Env) !Term {
            return self.*.term;
        }

        pub fn decode_(env: Env, term: Term) !@This() {
            return try readPid_(env, term);
        }
    };

pub const IsoxExtensionConfig = struct {
    name: [*c]const u8,
    load: *const fn (Env, Term) anyerror!*anyopaque,
    query: *const fn (Env, Term) anyerror!QueryReturn,
    createInstance: *const fn (Env, Term) anyerror!CreateInstanceReturn,
    queryInstance: *const fn (Env, IsoxResource, Term) anyerror!QueryInstanceReturn,
    updateInstance: *const fn (Env, IsoxResource, Term) anyerror!UpdateInstanceReturn,
    destroyInstance: *const fn (Env, IsoxResource) anyerror!DestroyInstanceReturn,
    unload: *const fn () void,
};

pub const ResourceType = *const c.ResourceType;

fn resourceDestructorWrapper(context: ?*const anyopaque, cEnv: ?*c.Env, obj: ?*anyopaque) callconv(.C) void {
    if (obj) |nonNullObj| {
        const destructor: *const fn (Env, *anyopaque) void = @alignCast(@ptrCast(context));
        const env = Env{ .env = cEnv, .allocator = isoxAllocator() };
        destructor(env, nonNullObj);
    }
}

pub fn allocResourceData(_: Env, comptime T: type) !*T {
    const resource = try isoxAllocator().create(T);
    const dst_cast = @as([*c]u8, @ptrCast(resource));
    @memset(dst_cast[0..@sizeOf(T)], @as(u8, 0));
    return resource;
}

pub fn freeResourceData(ptr: anytype) void {
    isoxAllocator().destroy(ptr);
}

pub fn createResourceType(env: Env, destructor: *const fn (Env, *anyopaque) void) !ResourceType {
    return c.isox_create_resource_type(@constCast(env.env), resourceDestructorWrapper, destructor).?;
}

pub fn makeBool(env: Env, value: bool) !Term {
    return c.isox_make_bool(env.env, value) orelse return IsoxError.NullTerm;
}

pub fn makeInt64(env: Env, value: i64) !Term {
    return c.isox_make_int64(env.env, value) orelse return IsoxError.NullTerm;
}

pub fn makeFloat(env: Env, value: f64) !Term {
    return c.isox_make_float(env.env, value) orelse return IsoxError.NullTerm;
}

pub fn encode(comptime T: type, env: Env, value: T) !Term {
    switch (@typeInfo(T)) {
        .Bool => {
            return makeBool(env, value);
        },
        .Int => {
            return makeInt64(env, @intCast(value));
        },
        .Float => {
            return makeFloat(env, value);
        },
        .Struct => |structInfo| {
            if (@hasDecl(T, "encode_")) {
                return try T.encode_(&value, env);
            } else if (std.mem.eql(u8, @typeName(T), "HashMap")) {
                return IsoxError.NullTerm;
            } else if (structInfo.is_tuple) {
                if (structInfo.fields.len == 0) {
                    return c.isox_make_tuple(env.env, 0, null);
                } else {
                    var terms = try env.allocator.alloc(Term, structInfo.fields.len);
                    defer env.allocator.free(terms);
                    comptime var i = 0;
                    inline for (structInfo.fields) |field| {
                        const child = @field(value, field.name);
                        terms[i] = try encode(field.type, env, child);
                        i = i + 1;
                    }
                    return c.isox_make_tuple(env.env, structInfo.fields.len, &terms[0]) orelse return IsoxError.NullTerm;
                }
            } else {
                var map = c.isox_make_map(env.env) orelse return IsoxError.NullTerm;
                inline for (structInfo.fields) |field| {
                    const key = try encode(IsoxAtom, env, IsoxAtom.newLiteral(field.name));
                    const child = @field(value, field.name);

                    map = c.isox_add_map_entry(env.env, key, try encode(field.type, env, child), map) orelse return IsoxError.NullTerm;
                }
                return map;
            }
        },
        .Enum => |_| {
            if (@hasDecl(T, "encode_")) {
                return try T.encode_(&value, env);
            } else {
                switch (value) {
                    inline else => |tag| {
                        const tagName = @tagName(tag);
                        const lowerCased = try lowercaseFirstChar(tagName, env.allocator);
                        defer env.allocator.free(lowerCased);
                        return encode(TaggedEnum, env, .{IsoxAtom.newNoAlloc(lowerCased)});
                    },
                }
            }
        },
        .Union => |union_info| {
            if (@hasDecl(T, "encode_")) {
                return try T.encode_(&value, env);
            } else if (union_info.tag_type) |tag_type| {
                switch (@as(tag_type, value)) {
                    inline else => |tag| {
                        const tagName = @tagName(tag);
                        const lowerCased = try lowercaseFirstChar(tagName, env.allocator);
                        defer env.allocator.free(lowerCased);
                        const tagValue = @field(value, tagName);
                        return encode(TaggedValue(@TypeOf(tagValue)), env, .{ IsoxAtom.newNoAlloc(lowerCased), tagValue });
                    },
                }
            } else {
                return IsoxError.UnknownUnionTagType;
            }
        },
        .Optional => |optional| {
            if (value) |just| {
                return encode(TaggedValue(optional.child), env, .{ IsoxAtom.newLiteral("just"), just });
            } else {
                return encode(struct { IsoxAtom }, env, .{IsoxAtom.newLiteral("nothing")});
            }
        },
        .Pointer => |pointer| {
            if (pointer.child == c.Term) {
                // todo - one vs many
                return value;
            } else switch (pointer.size) {
                .One => {
                    switch (pointer.child) {
                        c.Term => {
                            return c.isox_make_list(env.env, 1, &value) orelse return IsoxError.NullTerm;
                        },
                        else => {
                            const term = try encode(pointer.child, env, value.*);
                            const terms = [_](Term){term};
                            return c.isox_make_list(env.env, 1, &terms[0]) orelse return IsoxError.NullTerm;
                        },
                    }
                },
                .Slice => {
                    if (value.len > 0) {
                        var terms = try env.allocator.alloc(Term, value.len);
                        defer env.allocator.free(terms);

                        var i: u64 = 0;
                        for (value) |item| {
                            terms[i] = try encode(pointer.child, env, item);
                            i = i + 1;
                        }
                        return c.isox_make_list(env.env, value.len, &terms[0]) orelse return IsoxError.NullTerm;
                    } else {
                        return c.isox_make_list(env.env, 0, null) orelse return IsoxError.NullTerm;
                    }
                },
                .Many,
                .C,
                => {
                    std.log.warn("Unsupported pointer type {s} {any}\n", .{ @typeName(T), pointer });
                    return IsoxError.UnsupportedPointerType;
                },
            }
        },
        .Void => {},
        .Array,
        .Type,
        .NoReturn,
        .Fn,
        .Opaque,
        .ComptimeInt,
        .ComptimeFloat,
        .Undefined,
        .Null,
        .ErrorUnion,
        .ErrorSet,
        .Frame,
        .AnyFrame,
        .EnumLiteral,
        .Vector,
        => {
            std.log.warn("Unsupported encode type {s} {any}\n", .{ @typeName(T), @typeInfo(T) });
            return IsoxError.UnsupportedEncodeType;
        },
    }
}

pub fn decode(comptime T: type, env: Env, term: Term) !T {
    switch (@typeInfo(T)) {
        .Bool => {
            return readBool_(env, term);
        },
        .Int => {
            const i = try readInt64_(env, term);
            return @intCast(i);
        },
        .Float => {
            return readFloat_(env, term);
        },
        .Struct => |structInfo| {
            if (@hasDecl(T, "decode_")) {
                // Catch errors from the decoder and log them
                return T.decode_(env, term) catch |err| {
                    std.log.warn("Error decoding {s}: {s}", .{ @typeName(T), @errorName(err) });
                    return err;
                };
            } else if (structInfo.is_tuple) {
                var val: T = undefined;
                comptime var i = 0;
                inline for (structInfo.fields) |field| {
                    const child = readTupleItem_(env, term, i) catch |err| {
                        std.log.warn("Error reading tuple item {d} for {s}: {s}", .{ i, @typeName(T), @errorName(err) });
                        return err;
                    };
                    @field(val, field.name) = decode(field.type, env, child) catch |err| {
                        std.log.warn("Error decoding field {s} in tuple for {s}: {s}", .{ field.name, @typeName(T), @errorName(err) });
                        return err;
                    };
                    i = i + 1;
                }
                return val;
            } else {
                var val: T = undefined;
                inline for (structInfo.fields) |field| {
                    const child = readMapEntry_(env, term, field.name) catch |err| {
                        std.log.warn("Error reading map entry {s} for {s}: {s}", .{ field.name, @typeName(T), @errorName(err) });
                        return err;
                    };
                    @field(val, field.name) = decode(field.type, env, child) catch |err| {
                        std.log.warn("Error decoding field {s} in map for {s}: {s}", .{ field.name, @typeName(T), @errorName(err) });
                        return err;
                    };
                }
                return val;
            }
        },
        .Enum => |enumInfo| {
            if (@hasDecl(T, "decode_")) {
                return T.decode_(env, term) catch |err| {
                    std.log.warn("Error decoding enum {s}: {s}", .{ @typeName(T), @errorName(err) });
                    return err;
                };
            } else {
                // We have a tuple of Atom and Value
                const decoded = decode(TaggedEnum, env, term) catch |err| {
                    std.log.warn("Error decoding TaggedEnum for enum {s}: {s}", .{ @typeName(T), @errorName(err) });
                    return err;
                };
                defer decoded[0].deinit();
                const upperCased = uppercaseFirstChar(decoded[0].atom, env.allocator) catch |err| {
                    std.log.warn("Error uppercasing atom for enum {s}: {s}", .{ @typeName(T), @errorName(err) });
                    return err;
                };
                defer env.allocator.free(upperCased);
                inline for (enumInfo.fields) |field| {
                    if (std.mem.eql(u8, field.name, upperCased)) {
                        return @enumFromInt(field.value);
                    }
                    if (std.mem.eql(u8, field.name, decoded[0].atom)) {
                        return @enumFromInt(field.value);
                    }
                }
                std.log.warn("Unknown enum tag '{s}' / '{s}' for {s}", .{ decoded[0].atom, upperCased, @typeName(T) });
                return IsoxError.UnknownEnumTag;
            }
        },
        .Union => |union_info| {
            if (@hasDecl(T, "decode_")) {
                return T.decode_(env, term) catch |err| {
                    std.log.warn("Error decoding union {s}: {s}", .{ @typeName(T), @errorName(err) });
                    return err;
                };
            } else if (union_info.tag_type) |_| {
                // We have a tuple of Atom and Value
                const decoded = decode(TaggedValue(Term), env, term) catch |err| {
                    std.log.warn("Error decoding TaggedValue for union {s}: {s}", .{ @typeName(T), @errorName(err) });
                    return err;
                };
                defer decoded[0].deinit();
                const upperCased = uppercaseFirstChar(decoded[0].atom, env.allocator) catch |err| {
                    std.log.warn("Error uppercasing atom for union {s}: {s}", .{ @typeName(T), @errorName(err) });
                    return err;
                };
                defer env.allocator.free(upperCased);
                inline for (union_info.fields) |field| {
                    if (std.mem.eql(u8, field.name, upperCased)) {
                        return @unionInit(T, field.name, try decode(field.type, env, decoded[1]));
                    }
                    if (std.mem.eql(u8, field.name, decoded[0].atom)) {
                        return @unionInit(T, field.name, try decode(field.type, env, decoded[1]));
                    }
                }
                std.log.warn("Unknown union tag '{s}' for {s}", .{ decoded[0].atom, @typeName(T) });
                return IsoxError.UnknownUnionTag;
            } else {
                std.log.warn("Unknown union tag type for {s}", .{@typeName(T)});
                return IsoxError.UnknownUnionTagType;
            }
        },
        .Optional => |optional| {
            const len = readTupleLen_(env, term) catch |err| {
                std.log.warn("Error reading tuple length for optional {s}: {s}", .{ @typeName(T), @errorName(err) });
                return err;
            };
            if (len == 1) {
                // nothing
                return null;
            } else {
                const child = readTupleItem_(env, term, 1) catch |err| {
                    std.log.warn("Error reading tuple item for optional {s}: {s}", .{ @typeName(T), @errorName(err) });
                    return err;
                };
                return decode(optional.child, env, child) catch |err| {
                    std.log.warn("Error decoding optional child for {s}: {s}", .{ @typeName(T), @errorName(err) });
                    return err;
                };
            }
        },
        .Pointer => |pointer| {
            if (pointer.child == c.Term) {
                // todo - one vs many
                return term;
            } else switch (pointer.size) {
                .One => {
                    const item = env.allocator.create(pointer.child) catch |err| {
                        std.log.warn("Error allocating pointer for {s}: {s}", .{ @typeName(T), @errorName(err) });
                        return err;
                    };
                    errdefer env.allocator.destroy(item);

                    const listItem = readListItem_(env, term, 0) catch |err| {
                        std.log.warn("Error reading list item for pointer {s}: {s}", .{ @typeName(T), @errorName(err) });
                        return err;
                    };

                    item.* = decode(pointer.child, env, listItem) catch |err| {
                        std.log.warn("Error decoding pointer child for {s}: {s}", .{ @typeName(T), @errorName(err) });
                        return err;
                    };

                    return item;
                },
                .Slice => {
                    const len: u64 = readListLen_(env, term) catch |err| {
                        std.log.warn("Error reading list length for slice {s}: {s}", .{ @typeName(T), @errorName(err) });
                        return err;
                    };

                    var list = env.allocator.alloc(pointer.child, len) catch |err| {
                        std.log.warn("Error allocating slice for {s}: {s}", .{ @typeName(T), @errorName(err) });
                        return err;
                    };
                    errdefer env.allocator.free(list);

                    for (0..len) |i| {
                        const listItem = readListItem_(env, term, i) catch |err| {
                            std.log.warn("Error reading list item {d} for slice {s}: {s}", .{ i, @typeName(T), @errorName(err) });
                            return err;
                        };

                        list[i] = decode(pointer.child, env, listItem) catch |err| {
                            std.log.warn("Error decoding slice item {d} for {s}: {s}", .{ i, @typeName(T), @errorName(err) });
                            return err;
                        };
                    }
                    return list;
                },
                .Many,
                .C,
                => {
                    std.log.warn("Unsupported pointer type for {s}", .{@typeName(T)});
                    return IsoxError.UnsupportedPointerType;
                },
            }
        },
        .Void => {},
        .Array,
        .Type,
        .NoReturn,
        .Fn,
        .Opaque,
        .ComptimeInt,
        .ComptimeFloat,
        .Undefined,
        .Null,
        .ErrorUnion,
        .ErrorSet,
        .Frame,
        .AnyFrame,
        .EnumLiteral,
        .Vector,
        => {
            std.log.warn("Unsupported decode type {s}", .{@typeName(T)});
            return IsoxError.UnsupportedDecodeType;
        },
    }
}

pub fn readBool_(_: Env, term: Term) !bool {
    var val: bool = false;
    if (c.isox_read_bool(term, &val)) {
        return val;
    } else {
        unexpectedTypeWarning("bool", term);
        return IsoxError.UnexpectedTermType;
    }
}

pub fn readInt64_(_: Env, term: Term) !i64 {
    var val: i64 = 0;
    if (c.isox_read_int64(term, &val)) {
        return val;
    } else {
        unexpectedTypeWarning("int64", term);
        return IsoxError.UnexpectedTermType;
    }
}

pub fn readFloat_(_: Env, term: Term) !f64 {
    var val: f64 = 0;
    if (c.isox_read_float(term, &val)) {
        return val;
    } else {
        unexpectedTypeWarning("float", term);
        return IsoxError.UnexpectedTermType;
    }
}

pub fn readPid_(_: Env, term: Term) !IsoxPid {
    if (c.isox_term_type(term) == c.Pid) {
        return IsoxPid{ .term = term };
    } else {
        unexpectedTypeWarning("pid", term);
        return IsoxError.UnexpectedTermType;
    }
}

pub fn readResource_(_: Env, term: Term) !IsoxResource {
    var resourceType: ?*c.ResourceType = undefined;
    var obj: ?*anyopaque = undefined;
    if (c.isox_read_resource(term, &resourceType, &obj)) {
        return IsoxResource{
            .resourceType = resourceType orelse return IsoxError.NullTerm,
            .term = term,
            .obj = obj orelse return IsoxError.NullTerm,
        };
    } else {
        unexpectedTypeWarning("resource", term);
        return IsoxError.UnexpectedTermType;
    }
}

pub fn readAtom_(env: Env, term: Term) ![:0]const u8 {
    var len: u64 = 0;
    if (c.isox_read_atom_len(term, &len)) {
        var val = try env.allocator.allocSentinel(u8, len - 1, 0);
        _ = c.isox_read_atom(term, &val[0], len);
        return val;
    } else {
        unexpectedTypeWarning("atom", term);
        return IsoxError.UnexpectedTermType;
    }
}

pub fn readString_(env: Env, term: Term) ![:0]const u8 {
    var len: u64 = 0;
    if (c.isox_read_string_len(term, &len)) {
        var val = try env.allocator.allocSentinel(u8, len - 1, 0);
        _ = c.isox_read_string(term, &val[0], len);
        return val;
    } else {
        unexpectedTypeWarning("string", term);
        return IsoxError.UnexpectedTermType;
    }
}

pub fn readMapEntry_(env: Env, term: Term, key: [:0]const u8) !Term {
    var entry: ?*const c.Term = undefined;
    const keyAtom = try IsoxAtom.new(env, key);
    defer keyAtom.deinit();
    if (c.isox_read_map_entry(term, try encode(IsoxAtom, env, keyAtom), &entry)) {
        return entry orelse return IsoxError.NullTerm;
    } else {
        if (c.isox_term_type(term) != c.Map) {
            unexpectedTypeWarning("map", term);
            return IsoxError.UnexpectedTermType;
        } else {
            std.log.warn("Failed to find key {s} in map. Existing keys are {{", .{key});
            var it: ?*c.MapIterator = undefined;
            var mapKey: ?Term = undefined;
            var mapValue: ?Term = undefined;
            _ = c.isox_get_map_iterator(term, &it);
            while (c.isox_read_map_iterator_next(it, &mapKey, &mapValue)) {
                if (mapKey) |mapKey2| {
                    const atom = try readAtom_(env, mapKey2);
                    std.log.warn("\t\t{s}", .{atom});
                }
            }
            std.log.warn("}}", .{});
            return IsoxError.KeyNotFound;
        }
    }
}

pub fn readListLen_(_: Env, term: Term) !u64 {
    var len: u64 = 0;
    if (c.isox_read_list_len(term, &len)) {
        return len;
    } else {
        return IsoxError.UnexpectedTermType;
    }
}

pub fn readListItem_(_: Env, term: Term, i: u64) !Term {
    var item: ?*const c.Term = undefined;
    if (c.isox_read_list_item(term, i, &item)) {
        return item orelse return IsoxError.NullTerm;
    } else {
        unexpectedTypeWarning("list", term);
        return IsoxError.UnexpectedTermType;
    }
}

pub fn readTupleItem_(_: Env, term: Term, i: u64) !Term {
    var item: ?*const c.Term = undefined;
    if (c.isox_read_tuple_item(term, i, &item)) {
        return item orelse return IsoxError.NullTerm;
    } else {
        unexpectedTypeWarning("tuple", term);
        return IsoxError.UnexpectedTermType;
    }
}

pub fn readTupleLen_(_: Env, term: Term) !u64 {
    var len: u64 = 0;
    if (c.isox_read_tuple_len(term, &len)) {
        return len;
    } else {
        unexpectedTypeWarning("tuple", term);
        return IsoxError.UnexpectedTermType;
    }
}

pub fn allocEnv() Env {
    return Env{ .env = c.isox_alloc_env(), .allocator = isoxAllocator() };
}

pub fn freeEnv(env: Env) void {
    c.isox_free_env(env.env);
}

pub fn copyTerm(env: Env, term: anytype) @TypeOf(term) {
    _ = c.isox_copy_term(env.env, term.term);
    return term;
}

pub fn copyEnv(env: Env) Env {
    return Env{
        .env = c.isox_copy_env(env.env),
        .allocator = env.allocator,
    };
}

const MessageRecord = struct {
    typeName: []const u8,
    ptr: *const anyopaque,
};

var testMessages = if (builtin.is_test)
    std.AutoHashMap(i64, std.ArrayList(MessageRecord)).init(std.heap.page_allocator)
else {};

pub fn storeTestMessage(pid: IsoxPid, msg: anytype) void {
    if (!builtin.is_test) @compileError("storeTestMessage is test-only");

    var list = testMessages.getOrPut(pid.id) catch unreachable;
    if (!list.found_existing) {
        list.value_ptr.* = std.ArrayList(MessageRecord).init(std.heap.page_allocator);
    }

    const msg_copy = std.heap.page_allocator.create(@TypeOf(msg)) catch unreachable;
    msg_copy.* = msg;

    list.value_ptr.append(.{
        .typeName = @typeName(@TypeOf(msg)),
        .ptr = msg_copy,
    }) catch unreachable;
}

pub fn getTestMessages(pid: IsoxPid) []const MessageRecord {
    if (!builtin.is_test) @compileError("getTestMessages is test-only");

    if (testMessages.get(pid.id)) |list| {
        return list.items;
    }
    return &[_]MessageRecord{};
}

pub fn sendMsg(comptime T: type, env: Env, pid: IsoxPid, msg: T) !void {
    if (builtin.is_test) {
        storeTestMessage(pid, msg);
    } else {
        const encoded = try encode(T, env, msg);
        const res = c.isox_send_msg(@constCast(env.env), pid.term, encoded);
        if (!res) {
            return IsoxError.SendFailed;
        }
    }
}

fn formatStack(allocator: std.mem.Allocator, stack: ?*const std.builtin.StackTrace) ![:0]u8 {
    if (stack) |stack2| {
        const len = 1024;
        const buf = try allocator.allocSentinel(u8, len, 0);
        errdefer allocator.free(buf);

        var stream = std.io.fixedBufferStream(buf[0..len]);
        const writer = stream.writer();
        try stack2.format("", .{}, writer);
        const written = try stream.getPos();
        if (written < len) {
            buf[written] = 0;
        } else {
            buf[written - 1] = 0;
        }
        return buf;
    }
    return IsoxError.NoStackTraceAvailable;
}

fn errorTerm(env: Env, errorType: [:0]const u8, err: anyerror, stack: ?*std.builtin.StackTrace) Term {
    const Shape = struct { IsoxAtom, struct { IsoxAtom, IsoxAtom, IsoxString } };
    const formattedStack = formatStack(env.allocator, stack) catch @constCast(NoStackTrace);
    defer env.allocator.free(formattedStack);

    return encode(Shape, env, .{
        IsoxAtom.err(), .{
            IsoxAtom.newNoAlloc(errorType),
            IsoxAtom.newNoAlloc(@errorName(err)),
            IsoxString.newNoAlloc(formattedStack),
        },
    }) catch unreachable;
}

fn null_to_undef(env: Env, term: ?Term) !Term {
    return term orelse try encode(IsoxAtom, env, IsoxAtom.undef());
}

fn loadWrapper(context: ?*const anyopaque, cEnv: ?*c.Env, args: ?*const c.Term, priv_data: [*c]?*const anyopaque) callconv(.C) ?*const c.Term {
    const extension: *const IsoxExtensionConfig = contextToConfig(context);
    const env = Env{ .env = cEnv, .allocator = isoxAllocator() };
    if (extension.load(env, args.?)) |data| {
        priv_data.* = data;
        return null;
    } else |err| {
        return errorTerm(env, "extensionCallError", err, @errorReturnTrace());
    }
}

fn callQuery(env: Env, extension: *const IsoxExtensionConfig, args: ?*const c.Term) !*const c.Term {
    const response = try extension.query(env, args.?);
    return try encode(QueryResponse, env, .{ IsoxAtom.ok(), try null_to_undef(env, response) });
}

fn queryWrapper(context: ?*const anyopaque, cEnv: ?*c.Env, args: ?*const c.Term) callconv(.C) ?*const c.Term {
    const extension: *const IsoxExtensionConfig = contextToConfig(context);
    const env = Env{ .env = cEnv, .allocator = isoxAllocator() };
    if (callQuery(env, extension, args)) |response| {
        return response;
    } else |err| {
        return errorTerm(env, "extensionCallError", err, @errorReturnTrace());
    }
}

fn callCreateInstance(env: Env, extension: *const IsoxExtensionConfig, args: ?*const c.Term) !*const c.Term {
    const response = try extension.createInstance(env, args.?);
    return try encode(CreateInstanceResponse, env, .{ IsoxAtom.ok(), response[0], try null_to_undef(env, response[1]) });
}

fn createInstanceWrapper(context: ?*const anyopaque, cEnv: ?*c.Env, args: ?*const c.Term) callconv(.C) ?*const c.Term {
    const extension: *const IsoxExtensionConfig = contextToConfig(context);
    const env = Env{ .env = cEnv, .allocator = isoxAllocator() };
    if (callCreateInstance(env, extension, args)) |response| {
        return response;
    } else |err| {
        return errorTerm(env, "extensionCallError", err, @errorReturnTrace());
    }
}

fn callQueryInstance(env: Env, extension: *const IsoxExtensionConfig, instanceId: ?*const c.Term, args: ?*const c.Term) !*const c.Term {
    const response = try extension.queryInstance(env, try decode(IsoxResource, env, instanceId.?), args.?);
    return try encode(QueryInstanceResponse, env, .{ IsoxAtom.ok(), try null_to_undef(env, response) });
}

fn queryInstanceWrapper(context: ?*const anyopaque, cEnv: ?*c.Env, instanceId: ?*const c.Term, args: ?*const c.Term) callconv(.C) ?*const c.Term {
    const extension: *const IsoxExtensionConfig = contextToConfig(context);
    const env = Env{ .env = cEnv, .allocator = isoxAllocator() };
    if (callQueryInstance(env, extension, instanceId, args)) |response| {
        return response;
    } else |err| {
        return errorTerm(env, "extensionCallError", err, @errorReturnTrace());
    }
}

fn callUpdateInstance(env: Env, extension: *const IsoxExtensionConfig, instanceId: ?*const c.Term, args: ?*const c.Term) !*const c.Term {
    const response = try extension.updateInstance(env, try decode(IsoxResource, env, instanceId.?), args.?);
    const ret = try encode(UpdateInstanceResponse, env, .{ IsoxAtom.ok(), try null_to_undef(env, response) });
    return ret;
}

fn updateInstanceWrapper(context: ?*const anyopaque, cEnv: ?*c.Env, instanceId: ?*const c.Term, args: ?*const c.Term) callconv(.C) ?*const c.Term {
    const extension: *const IsoxExtensionConfig = contextToConfig(context);
    const env = Env{ .env = cEnv, .allocator = isoxAllocator() };
    if (callUpdateInstance(env, extension, instanceId, args)) |response| {
        return response;
    } else |err| {
        return errorTerm(env, "extensionCallError", err, @errorReturnTrace());
    }
}

fn callDestroyInstance(env: Env, extension: *const IsoxExtensionConfig, instanceId: ?*const c.Term) !*const c.Term {
    try extension.destroyInstance(env, try decode(IsoxResource, env, instanceId.?));
    return try encode(DestroyInstanceResponse, env, IsoxAtom.ok());
}

fn destroyInstanceWrapper(context: ?*const anyopaque, cEnv: ?*c.Env, instanceId: ?*const c.Term) callconv(.C) ?*const c.Term {
    const extension: *const IsoxExtensionConfig = contextToConfig(context);
    const env = Env{ .env = cEnv, .allocator = isoxAllocator() };
    if (callDestroyInstance(env, extension, instanceId)) |response| {
        return response;
    } else |err| {
        return errorTerm(env, "extensionCallError", err, @errorReturnTrace());
    }
}

fn unloadWrapper(context: ?*const anyopaque) callconv(.C) void {
    // todo
    // extension.unload();
    _ = context;
}

pub fn init(comptime extension: *const IsoxExtensionConfig) *const c.IsoxFunctionTable {
    const p = comptime c.IsoxFunctionTable{
        .name = extension.name,
        .load = loadWrapper,
        .query = queryWrapper,
        .create_instance = createInstanceWrapper,
        .query_instance = queryInstanceWrapper,
        .update_instance = updateInstanceWrapper,
        .destroy_instance = destroyInstanceWrapper,
        .unload = unloadWrapper,
        .context = @ptrCast(extension),
    };

    return &p;
}

fn lowercaseFirstChar(orig: [:0]const u8, allocator: std.mem.Allocator) ![:0]u8 {
    const len = orig.len;
    var copy = try allocator.allocSentinel(u8, len, 0);
    @memcpy(copy, orig);

    if (len > 0) {
        copy[0] = std.ascii.toLower(copy[0]);
    }

    return copy;
}

fn uppercaseFirstChar(orig: [:0]const u8, allocator: std.mem.Allocator) ![:0]u8 {
    const len = orig.len;
    var copy = try allocator.allocSentinel(u8, len, 0);
    @memcpy(copy, orig);

    if (len > 0) {
        copy[0] = std.ascii.toUpper(copy[0]);
    }

    return copy;
}

fn contextToConfig(context: ?*const anyopaque) *const IsoxExtensionConfig {
    const aligned: ?*align(@alignOf(IsoxExtensionConfig)) const anyopaque = @alignCast(context);
    const extension: *const IsoxExtensionConfig = @ptrCast(aligned);

    return extension;
}

pub fn log(
    comptime level: std.log.Level,
    comptime scope: @Type(.EnumLiteral),
    comptime format: []const u8,
    args: anytype,
) void {
    const allocator = isoxAllocator();

    const formatted = std.fmt.allocPrintZ(allocator, format, args) catch |err| {
        std.debug.print("Failed to format log message with args: {}\n", .{err});
        return;
    };
    defer allocator.free(formatted);

    const levelInt: c_int = switch (level) {
        std.log.Level.err => c.Error,
        std.log.Level.warn => c.Warning,
        std.log.Level.info => c.Info,
        std.log.Level.debug => c.Debug,
    };

    if (!c.isox_log(levelInt, @tagName(scope), formatted)) {
        log_to_file(level, scope, formatted);
    }
}

pub fn log_to_filef(
    comptime level: std.log.Level,
    comptime scope: @Type(.EnumLiteral),
    comptime format: []const u8,
    args: anytype,
) void {
    const allocator = isoxAllocator();

    const formatted = std.fmt.allocPrintZ(allocator, format, args) catch |err| {
        std.debug.print("Failed to format log message with args: {}\n", .{err});
        return;
    };
    defer allocator.free(formatted);

    log_to_file(level, scope, formatted);
}

pub fn log_to_file(
    comptime level: std.log.Level,
    comptime scope: @Type(.EnumLiteral),
    logMessage: []const u8,
) void {
    const allocator = isoxAllocator();

    const prefix = "[" ++ comptime level.asText() ++ "] " ++ "(" ++ @tagName(scope) ++ ") ";
    const message = std.fmt.allocPrint(allocator, "{s}{s}\n", .{ prefix, logMessage }) catch |err| {
        std.debug.print("Failed to format log message with args: {}\n", .{err});
        return;
    };
    defer allocator.free(message);

    const home = std.posix.getenv("TMP") orelse "/tmp";
    const path = std.fmt.allocPrint(allocator, "{s}/{s}", .{ home, "isox.log" }) catch |err| {
        std.debug.print("Failed to create log file path: {}\n", .{err});
        return;
    };
    defer allocator.free(path);

    const file = std.fs.createFileAbsolute(path, .{ .truncate = false }) catch |err| {
        std.debug.print("Failed to open log file: {s} / {}\n", .{ path, err });
        return;
    };
    defer file.close();

    const stat = file.stat() catch |err| {
        std.debug.print("Failed to get stat of log file: {}\n", .{err});
        return;
    };

    file.seekTo(stat.size) catch |err| {
        std.debug.print("Failed to seek log file: {}\n", .{err});
        return;
    };

    file.writeAll(message) catch |err| {
        std.debug.print("Failed to write to log file: {}\n", .{err});
    };
}

pub fn isoxAllocator() std.mem.Allocator {
    if (builtin.is_test) {
        return std.testing.allocator;
    } else {
        return std.heap.c_allocator;
    }
}
