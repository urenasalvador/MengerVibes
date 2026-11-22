// SPDX-License-Identifier: GPL-3.0-only

struct Settings {
    width: u32,
    height: u32,
    threshold: f32,
    tile_columns: u32,
    tile_width: u32,
    tile_height: u32,
    completed: u32,
}

@group(0) @binding(0) var accumulated: texture_2d<f32>;
@group(0) @binding(1) var half_estimate: texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> mask: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> remaining: array<atomic<u32>>;
@group(0) @binding(4) var<uniform> settings: Settings;

var<workgroup> active_count: atomic<u32>;

fn noisy(p: vec2<i32>) -> bool {
    let sample_mean = textureLoad(accumulated, p, 0);
    let full = sample_mean.rgb;
    let half_color = textureLoad(half_estimate, p, 0).rgb;
    let magnitude = max(max(full.r, full.g), full.b);
    let delta = 2.0 * abs(full - half_color);
    let dark_surface = settings.completed < 32u && sample_mean.a > 0.5 && magnitude < 0.002;
    return dark_surface || max(max(delta.r, delta.g), delta.b) > settings.threshold * (0.1 + magnitude);
}

@compute @workgroup_size(8, 8)
fn check_noise(@builtin(global_invocation_id) id: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    if lane == 0u { atomicStore(&active_count, 0u); }
    workgroupBarrier();
    let pixel = id.xy;
    let size = vec2<u32>(settings.width, settings.height);
    if all(pixel < size) {
        let index = pixel.y * textureDimensions(accumulated).x + pixel.x;
        let bit = 1u << (index % 32u);
        if (atomicLoad(&mask[index / 32u]) & bit) == 0u {
            let p = vec2<i32>(pixel);
            let edge = vec2<i32>(size - 1u);
            // Keep a one-pixel margin around unstable pixels and silhouettes.
            if noisy(p) || noisy(max(p - vec2<i32>(1, 0), vec2<i32>(0)))
                || noisy(min(p + vec2<i32>(1, 0), edge))
                || noisy(max(p - vec2<i32>(0, 1), vec2<i32>(0)))
                || noisy(min(p + vec2<i32>(0, 1), edge)) {
                atomicAdd(&active_count, 1u);
            } else {
                atomicOr(&mask[index / 32u], bit);
            }
        }
    }
    workgroupBarrier();
    if lane == 0u {
        let active_pixels = atomicLoad(&active_count);
        if active_pixels > 0u {
            atomicAdd(&remaining[0], active_pixels);
            if settings.tile_columns > 0u {
                let tile_index = pixel.y / settings.tile_height * settings.tile_columns + pixel.x / settings.tile_width;
                atomicAdd(&remaining[1u + tile_index], active_pixels);
            }
        }
    }
}
