// SPDX-License-Identifier: GPL-3.0-only

@group(1) @binding(0) var ray_allocation: texture_2d<u32>;

struct GuidedSeed {
    @location(0) color: vec4<f32>,
    @location(1) geometry: vec4<f32>,
}

struct GuidedSeedWithDenoise {
    @location(0) color: vec4<f32>,
    @location(1) geometry: vec4<f32>,
    @location(2) albedo: vec4<f32>,
    @location(3) normal: vec4<f32>,
}

fn seed_geometry(guide: SurfaceGuide, validity: f32) -> vec4<f32> {
    return vec4<f32>(guide.normal, select(-2.0, guide.depth, validity > 0.5));
}

@fragment fn fs_guided_seed(in: VertexOut) -> GuidedSeed {
    var guide: SurfaceGuide;
    let color = viewport_sample_range(in, 0.0, 1u, true, &guide);
    return GuidedSeed(color, seed_geometry(guide, color.a));
}

@fragment fn fs_guided_seed_denoise(in: VertexOut) -> GuidedSeedWithDenoise {
    var guide: SurfaceGuide;
    let color = viewport_sample_range(in, 0.0, 1u, true, &guide);
    return GuidedSeedWithDenoise(color, seed_geometry(guide, color.a), vec4<f32>(guide.albedo, 1.0), vec4<f32>(guide.normal, 1.0));
}

fn guided_sample(in: VertexOut, guide: ptr<function, SurfaceGuide>) -> vec4<f32> {
    let sample_pass = u32(camera.filtering.z);
    let assigned = textureLoad(ray_allocation, vec2<i32>(in.position.xy), 0).x;
    // Two uniform pilot passes precede the alternating guided and uniform
    // passes. The fixed allocation lets each pixel recover its sample count.
    let previous = 1u + sample_pass / 2u + ((sample_pass - 1u) / 2u) * assigned;
    let count = select(1u, assigned, sample_pass % 2u == 0u);
    if count == 0u { discard; }
    let color = viewport_sample_range(in, f32(previous), count, true, guide);
    return vec4<f32>(color.rgb, f32(count) / f32(previous + count));
}

@fragment fn fs_guided(in: VertexOut) -> @location(0) vec4<f32> {
    var guide: SurfaceGuide;
    return guided_sample(in, &guide);
}

@fragment fn fs_guided_denoise(in: VertexOut) -> ExportPasses {
    var guide: SurfaceGuide;
    let color = guided_sample(in, &guide);
    return ExportPasses(color, vec4<f32>(guide.albedo, color.a), vec4<f32>(guide.normal, color.a));
}
