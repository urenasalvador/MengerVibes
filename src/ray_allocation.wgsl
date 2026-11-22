// SPDX-License-Identifier: GPL-3.0-only

@group(0) @binding(0) var geometry: texture_2d<f32>;
@group(0) @binding(1) var allocation: texture_storage_2d<r32uint, write>;
@group(0) @binding(2) var<uniform> camera: array<vec4<f32>, 14>;
@group(0) @binding(3) var first_sample: texture_2d<f32>;
@group(0) @binding(4) var two_sample_mean: texture_2d<f32>;

fn edge_score(center: vec4<f32>, neighbor: vec4<f32>) -> f32 {
    if neighbor.w < -1.5 { return 0.65; }
    if (center.w < 0.0) != (neighbor.w < 0.0) { return 1.0; }
    if center.w < 0.0 { return 0.0; }
    let depth = abs(center.w - neighbor.w) / max(min(center.w, neighbor.w), 0.002);
    let normal = 1.0 - clamp(dot(center.xyz, neighbor.xyz), -1.0, 1.0);
    return clamp(8.0 * depth + 0.7 * normal, 0.0, 1.0);
}

fn importance(p: vec2<i32>, limit: vec2<i32>) -> f32 {
    let center = textureLoad(geometry, p, 0);
    if center.w < -1.5 { return 1.25; }
    var score = 0.0;
    for (var axis = 0u; axis < 2u; axis++) {
        let offset = select(vec2<i32>(1, 0), vec2<i32>(0, 1), axis == 1u);
        let left = textureLoad(geometry, max(p - offset, vec2<i32>(0)), 0);
        let right = textureLoad(geometry, min(p + offset, limit), 0);
        score += edge_score(center, left) + edge_score(center, right);
    }
    // The difference between independently traced subpixel samples catches
    // thin features and lighting variation even on a nearly planar surface.
    let first = textureLoad(first_sample, p, 0).rgb;
    let mean = textureLoad(two_sample_mean, p, 0).rgb;
    let difference = 2.0 * abs(mean - first);
    let variation = max(difference.r, max(difference.g, difference.b))
        / max(0.12 + max(mean.r, max(mean.g, mean.b)), 0.12);
    return max(score * 0.25, clamp(variation, 0.0, 1.0));
}

@compute @workgroup_size(8, 8)
fn allocate(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = select(textureDimensions(geometry), vec2<u32>(camera[13].yz), camera[13].y > 0.0);
    let pixel = id.xy;
    if any(pixel >= size) { return; }
    // Pair nearby pixels in alternating directions. Both members choose the
    // same axis, including at tile boundaries and with odd viewport sizes.
    let block = pixel / 4u;
    let vertical = (block.x ^ block.y) % 2u == 1u;
    let partner = pixel ^ select(vec2<u32>(2, 0), vec2<u32>(0, 2), vertical);
    var rays = 1u;
    if all(partner < size) {
        let limit = vec2<i32>(size - 1u);
        let own_score = importance(vec2<i32>(pixel), limit);
        let partner_score = importance(vec2<i32>(partner), limit);
        // Only take a ray from a locally stable pixel. A relative difference
        // alone can move samples between two pixels that both need them.
        if min(own_score, partner_score) < 0.025 && max(own_score, partner_score) > 0.15 {
            rays = select(0u, 2u, own_score > partner_score);
        }
    }
    textureStore(allocation, vec2<i32>(pixel), vec4<u32>(rays, 0, 0, 0));
}
