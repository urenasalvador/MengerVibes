// SPDX-License-Identifier: GPL-3.0-only

@group(1) @binding(0) var<storage, read> converged_pixels: array<u32>;

struct NoisePass {
    @location(0) color: vec4<f32>,
    @location(1) half_pass: vec4<f32>,
}

struct NoiseDenoisePass {
    @location(0) color: vec4<f32>,
    @location(1) half_pass: vec4<f32>,
    @location(2) albedo: vec4<f32>,
    @location(3) normal: vec4<f32>,
}

fn keep_sampling(position: vec2<f32>) {
    let pixel = vec2<u32>(position);
    // Fragment work uses local attachment coordinates. The host stores the
    // physical attachment width in the otherwise unused export/viewport slot.
    let index = pixel.y * u32(camera.noise_sampling.x) + pixel.x;
    if (converged_pixels[index / 32u] & (1u << (index % 32u))) != 0u { discard; }
}

fn half_sample(color: vec3<f32>) -> vec4<f32> {
    let index = u32(camera.filtering.z);
    // Each adjacent pair contributes one sample to the auxiliary estimate.
    // Alternate which member is chosen without dropping one Sobol stratum.
    let selected = (countOneBits(index) & 1u) == 0u;
    return vec4<f32>(color, select(0.0, 1.0 / f32(index / 2u + 1u), selected));
}

fn noise_color(color: vec3<f32>, depth: f32, valid: f32) -> vec4<f32> {
    // Accumulation alpha retains evidence of an uncertain surface under sampled lighting.
    let sampled_lighting = camera.path_settings.x > 0.5 || u32(camera.path_settings.y) == 10u;
    let uncertain = sampled_lighting && (depth >= 0.0 || valid < 1.0);
    return vec4<f32>(color, select(0.0, 1.0, uncertain));
}

fn noise_denoise(color: vec4<f32>, guide: SurfaceGuide) -> NoiseDenoisePass {
    return NoiseDenoisePass(color, half_sample(color.rgb), vec4<f32>(guide.albedo, 1.0), vec4<f32>(guide.normal, 1.0));
}

@fragment fn fs_noise_viewport(in: VertexOut) -> NoisePass {
    keep_sampling(in.position.xy);
    var guide: SurfaceGuide;
    let color = viewport_samples(in, camera.filtering.z, true, &guide);
    return NoisePass(noise_color(color.rgb, guide.depth, color.a), half_sample(color.rgb));
}

@fragment fn fs_noise_viewport_denoise(in: VertexOut) -> NoiseDenoisePass {
    keep_sampling(in.position.xy);
    var guide: SurfaceGuide;
    let color = viewport_samples(in, camera.filtering.z, true, &guide);
    return noise_denoise(noise_color(color.rgb, guide.depth, color.a), guide);
}

@fragment fn fs_noise_export(in: VertexOut) -> NoisePass {
    keep_sampling(in.position.xy);
    var guide: SurfaceGuide;
    let color = export_sample(in, &guide);
    return NoisePass(noise_color(color.rgb, guide.depth, guide.valid), half_sample(color.rgb));
}

@fragment fn fs_noise_export_denoise(in: VertexOut) -> NoiseDenoisePass {
    keep_sampling(in.position.xy);
    var guide: SurfaceGuide;
    let color = export_sample(in, &guide);
    return noise_denoise(noise_color(color.rgb, guide.depth, guide.valid), guide);
}
