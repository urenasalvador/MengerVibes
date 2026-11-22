// SPDX-License-Identifier: GPL-3.0-only

@group(0) @binding(2) var blue_noise: texture_2d_array<u32>;

fn r2_offset(index: f32) -> vec2<f32> {
    return fract(vec2<f32>(0.5) + index * vec2<f32>(0.754877666, 0.569840296)) - 0.5;
}

fn sobol_offset(pixel: vec2<u32>, index: u32) -> vec2<f32> {
    var directions = vec2<u32>(0x80000000u);
    var bits = index ^ (index >> 1u);
    var point = vec2<u32>(0u);
    while bits != 0u {
        if (bits & 1u) != 0u { point ^= directions; }
        directions = vec2<u32>(directions.x >> 1u, directions.y ^ (directions.y >> 1u));
        bits >>= 1u;
    }
    let seed = pixel.x + 13763u * pixel.y;
    let shift = vec2<u32>(pixel_hash(seed + 6143u), pixel_hash(seed + 67391u));
    return (vec2<f32>((point ^ shift) >> vec2<u32>(8u)) + 0.5) / 16777216.0 - 0.5;
}

fn stbn_offset(pixel: vec2<u32>, index: u32) -> vec2<f32> {
    let cycle = index >> 5u;
    let position = (pixel + vec2<u32>(cycle * 5u, cycle * 9u)) & vec2<u32>(15u);
    let packed = textureLoad(blue_noise, vec2<i32>(position), i32(index & 31u), 0).x;
    let ranks = vec2<u32>(packed & 65535u, packed >> 16u);
    // Shift and stratify later cycles to avoid replaying identical subpixel rays.
    let subrank = (vec2<u32>(cycle) * vec2<u32>(13u, 23u)) & vec2<u32>(31u);
    let within_rank = select((vec2<f32>(subrank) + 0.5) / 32.0, vec2<f32>(0.5), cycle == 0u);
    return (vec2<f32>(ranks) + within_rank) / 8192.0 - 0.5;
}

fn primary_offset(pixel: vec2<u32>, index: f32, temporal_start: f32) -> vec2<f32> {
    if camera.parameters.w > 0.5 { return vec2<f32>(0.0); }
    let mode = u32(camera.four_branch_twist.z);
    let sample = u32(index + temporal_start);
    if mode == 1u { return sobol_offset(pixel, sample); }
    if mode == 2u { return stbn_offset(pixel, sample); }
    return r2_offset(index);
}
