// SPDX-License-Identifier: GPL-3.0-only

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}
struct MutationField {
    blend: vec4<f32>,
    settings: vec4<f32>,
    rotation: mat3x3<f32>,
}
struct Camera {
    position_aspect: vec4<f32>,
    orientation: vec4<f32>,
    parameters: vec4<f32>,
    appearance: vec4<f32>,
    display: vec4<f32>,
    filtering: vec4<f32>,
    surface: vec4<f32>,
    light_color_metallic: vec4<f32>,
    light_settings: vec4<f32>,
    render_region: vec4<f32>,
    lens: vec4<f32>,
    fog: vec4<f32>,
    viewport: vec4<f32>,
    matcap: vec4<f32>,
    mandelbox: vec4<f32>,
    mandelbulb: vec4<f32>,
    menger: vec4<f32>,
    menger_rotation: mat3x3<f32>,
    background_color: vec4<f32>,
    kifs: vec4<f32>,
    amazing_surface: vec4<f32>,
    pseudo_kleinian: vec4<f32>,
    mutation_stack: vec4<f32>,
    extra_mutations: array<MutationField, 8>,
}
@group(0) @binding(0) var<uniform> camera: Camera;
@vertex fn vs_main(@builtin(vertex_index) i: u32) -> VertexOut {
    var positions = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
    var out: VertexOut;
    out.position = vec4<f32>(positions[i], 0.0, 1.0);
    out.uv = positions[i];
    return out;
}
fn bulb_direction(z: vec3<f32>, r: f32, power: f32, settings: vec3<f32>) -> vec3<f32> {
    let axis = clamp(z.z / r, -1.0, 1.0);
    let polar_offset = settings.x * sqrt(max(0.0, 1.0 - axis * axis));
    let polar = acos(axis) * power + polar_offset;
    let azimuth = atan2(z.y, z.x) * power + settings.y * r;
    return vec3<f32>(sin(polar) * cos(azimuth), sin(polar) * sin(azimuth), cos(polar));
}

fn bulb_seed(p: vec3<f32>) -> vec3<f32> {
    return p + vec3<f32>(camera.mandelbulb.z, 0.0, 0.0);
}

fn bulb_escape(p: vec3<f32>, iterations: i32) -> i32 {
    let seed = bulb_seed(p);
    let power = camera.parameters.x;
    let seed_r2 = dot(seed, seed);
    var z = p;
    for (var i = 0; i < iterations; i++) {
        let r2 = dot(z, z);
        if (r2 > 16.0) { return i; }
        // For power >= 2, r <= 0.5 maps to r <= 0.25 before adding the seed.
        if (r2 <= 0.25 && seed_r2 <= 0.0625) { return iterations; }
        let r = max(sqrt(r2), 0.00001);
        z = bulb_direction(z, r, power, camera.mandelbulb.xyz) * pow(r, power) + seed;
    }
    return iterations;
}

fn menger_distance(p: vec3<f32>, settings: vec4<f32>, rotation: mat3x3<f32>) -> f32 {
    let box = abs(p) - vec3<f32>(1.0);
    var distance = length(max(box, vec3<f32>(0.0))) + min(max(box.x, max(box.y, box.z)), 0.0);
    var point = p;
    var cell_size = 2.0;
    for (var level = 0; level < i32(settings.x); level++) {
        let tunnel_radius = cell_size * settings.y / 6.0;
        // A finer tunnel can't contribute a distance larger than its radius.
        if (distance >= tunnel_radius) { break; }
        // A centered grid at each scale cuts the central cross out of every cell.
        let offset = cell_size * abs(fract(point / cell_size + vec3<f32>(0.5)) - vec3<f32>(0.5));
        let q = offset - vec3<f32>(tunnel_radius);
        let cross_distance = min(max(q.x, q.y), min(max(q.x, q.z), max(q.y, q.z)));
        distance = max(distance, -cross_distance);
        point = rotation * point;
        cell_size /= 3.0;
    }
    return distance;
}

fn box_step(orbit: vec4<f32>, seed: vec3<f32>, settings: vec4<f32>) -> vec4<f32> {
    let limit = vec3<f32>(settings.y);
    let folded = 2.0 * clamp(orbit.xyz, -limit, limit) - orbit.xyz;
    let factor = clamp(settings.z / max(dot(folded, folded), 0.000001), 1.0, settings.w);
    return vec4<f32>(folded * factor * settings.x + seed, orbit.w * factor * abs(settings.x) + 1.0);
}

fn bulb_step(orbit: vec4<f32>, seed: vec3<f32>, r: f32, settings: vec4<f32>) -> vec4<f32> {
    let power = settings.w;
    let radial_power = pow(r, power - 1.0);
    let angular_stretch = power + abs(settings.x) + abs(settings.y) * r;
    let derivative = radial_power * angular_stretch * orbit.w + 1.0;
    let next = bulb_direction(orbit.xyz, r, power, settings.xyz) * (radial_power * r) + seed;
    return vec4<f32>(next, derivative);
}

fn box_distance(p: vec3<f32>, iterations: i32, settings: vec4<f32>) -> f32 {
    var state = vec4<f32>(p, 1.0);
    // Keep the Mandelbox orbit three levels deeper than the Mandelbulb orbit.
    let count = min(iterations + 3, 25);
    for (var i = 0; i < count; i++) {
        state = box_step(state, p, settings);
    }
    return length(state.xyz) / abs(state.w);
}

fn bulb_distance(p: vec3<f32>, iterations: i32, settings: vec4<f32>) -> f32 {
    var state = vec4<f32>(p, 1.0);
    var r = 0.0;
    let seed = p + vec3<f32>(settings.z, 0.0, 0.0);
    for (var i = 0; i < 22; i++) {
        if (i >= iterations) { break; }
        r = length(state.xyz);
        if (r > 4.0) { break; }
        let safe_r = max(r, 0.00001);
        state = bulb_step(state, seed, safe_r, settings);
    }
    return 0.5 * log(max(r, 0.00001)) * r / state.w;
}

// KIFS folding method by Knighty, described in Tim McGraw's
// "Interactive Procedural Building Generation", Algorithm 1.
// This octahedral distance estimator was written in WGSL for SDF_Fractal.
// https://web.ics.purdue.edu/~tmcgraw/papers/kifs_mcgraw_2015.pdf
fn kifs_distance(p: vec3<f32>, settings: vec4<f32>) -> f32 {
    let scale = settings.y;
    let offset = (scale - 1.0) * vec3<f32>(1.0, 1.0, 0.85);
    let c = cos(settings.z);
    let s = sin(settings.z);
    var z = p;
    var inverse_scale = 1.0;
    for (var i = 0; i < i32(settings.x); i++) {
        if (dot(z, z) > 16.0) {
            return (length(z) - 2.0) * inverse_scale;
        }
        let q = abs(vec3<f32>(c * z.x + s * z.z, z.y, c * z.z - s * z.x));
        let largest = max(q.x, max(q.y, q.z));
        let smallest = min(q.x, min(q.y, q.z));
        let middle = max(min(q.x, q.y), min(max(q.x, q.y), q.z));
        let scaled_z = scale * smallest;
        z = vec3<f32>(scale * largest - offset.x, scale * middle - offset.y,
                      abs(scaled_z - 0.5 * offset.z) - 0.5 * offset.z);
        inverse_scale /= scale;
    }
    return (length(z) - settings.w) * inverse_scale;
}

// Amazing Surface by Kali. This XY box and cylinder fold distance estimator
// was written in WGSL for SDF_Fractal using the formula description and the
// Mandelbulber project's GPLv3 implementation as references.
// https://www.fractalforums.com/index.php?action=downloads;sa=view;down=32
// https://github.com/buddhi1980/mandelbulber2/blob/master/mandelbulber2/formula/opencl/amazing_surf.cl
fn amazing_surface_distance(p: vec3<f32>, iterations: i32, settings: vec4<f32>) -> f32 {
    var z = p;
    var derivative = 1.0;
    var scale = settings.x;
    let limit = vec2<f32>(settings.y);
    for (var i = 0; i < min(iterations + 2, 22); i++) {
        if (dot(z, z) > 256.0) { break; }
        let folded = 2.0 * clamp(z.xy, -limit, limit) - z.xy;
        let radius_squared = dot(folded, folded);
        let gain = scale / clamp(radius_squared, settings.z, 1.0);
        // The cylindrical inversion also scales Z. Account for its shear
        // when the gain varies with the XY radius.
        var stretch = abs(gain);
        if (radius_squared > settings.z && radius_squared < 1.0) {
            stretch *= 1.0 + 2.0 * abs(z.z) * inverseSqrt(radius_squared);
        }
        derivative = min(derivative * stretch + 1.0, 1e30);
        z = gain * vec3<f32>(folded, z.z) + p.yxz;
        scale = settings.x + settings.w * (abs(scale) - 1.0);
    }
    return 0.65 * length(z) / derivative;
}

const MANDELBULB: f32 = 0.0;
const MANDELBOX: f32 = 1.0;
const MENGER: f32 = 2.0;
const KIFS: f32 = 3.0;
const AMAZING_SURFACE: f32 = 4.0;
const PSEUDO_KLEINIAN: f32 = 5.0;

// Pseudo-Kleinian by Knighty and Theli-at.
// Box fold and sphere inversion adapted from Mandelbulber v2's
// PseudoKleinianIteration (Copyright (C) 2017 Mandelbulber Team, GPLv3).
// https://github.com/buddhi1980/mandelbulber2
// https://sources.debian.org/src/mandelbulber2/2.13.2-5/mandelbulber2/formula/opencl/pseudo_kleinian.cl
// Adapted to WGSL for SDF_Fractal on 2026-11-19. This variant uses a signed
// shell field with adjustable thickness to support tracing from either side.
fn pseudo_kleinian_distance(p: vec3<f32>, iterations: i32, settings: vec4<f32>) -> f32 {
    let fold_limit = vec3<f32>(settings.x);
    let inversion_radius_squared = settings.y * settings.y;
    let offset = vec3<f32>(0.0, 0.0, settings.w);
    // Keep the shared default camera off the inversion axis.
    var z = p + settings.x * vec3<f32>(0.8125, 0.5625, 0.0);
    var inverse_derivative = 1.0;
    for (var i = 0; i < min(iterations, 22); i++) {
        z = 2.0 * clamp(z, -fold_limit, fold_limit) - z;
        let gain = max(inversion_radius_squared / max(dot(z, z), 1e-8), 1.0);
        z = z * gain + offset;
        inverse_derivative /= gain;
    }
    let radial = length(z.xy);
    let projected_height = abs(z.z) * radial / max(length(z), 0.00001);
    let shell = max(radial - 0.65 * settings.y, projected_height - settings.z);
    return 0.7 * shell * inverse_derivative;
}

fn base_orbit_distance(p: vec3<f32>, iterations: i32) -> f32 {
    if (camera.parameters.y == MANDELBOX) {
        return box_distance(p, iterations, camera.mandelbox);
    }
    return bulb_distance(p, iterations, vec4<f32>(camera.mandelbulb.xyz, camera.parameters.x));
}
// Compute the adaptive level independently so a normal can use one consistent
// level for all of its neighbouring samples.
fn detail_level(p: vec3<f32>, iterations: f32) -> f32 {
    if (camera.mutation_stack.y == 0.0 && camera.parameters.y == MENGER) { return 1.0; }
    let distance_to_camera = max(length(p - camera.position_aspect.xyz), 0.0001);
    let extra_detail = max(-log2(distance_to_camera) - 2.0, 0.0);
    // Reduced previews evaluate a single nearby orbit level per sample.
    // Still frames and export blend adjacent levels at the full detail range.
    if (camera.matcap.y > 0.0) {
        return iterations + round(min(extra_detail, 4.0));
    }
    return iterations + min(extra_detail, 11.0);
}
fn primary_distance_at_level(p: vec3<f32>, level: f32) -> f32 {
    if (camera.parameters.y == MENGER) {
        return menger_distance(p, camera.menger, camera.menger_rotation);
    }
    let lower = i32(floor(level));
    let fraction = fract(level);
    let a = base_orbit_distance(p, lower);
    if (fraction < 0.0001) { return a; }
    let b = base_orbit_distance(p, lower + 1);
    return mix(a, b, smoothstep(0.0, 1.0, fraction));
}
fn distance_at_level(p: vec3<f32>, level: f32) -> f32 {
    let count = i32(camera.mutation_stack.y);
    let primary_coefficient = select(camera.mutation_stack.x, 1.0, count == 0);
    let folded = camera.parameters.y >= KIFS;
    var result = 0.0;
    if (primary_coefficient > 0.0 && !folded) {
        result = primary_coefficient * primary_distance_at_level(p, level);
    }

    let folded_layers = select(0, 1, folded && primary_coefficient > 0.0);
    let iterations = i32(round(level));
    // Share one call site per field for both primary and additional mutations.
    for (var i = 0; i < count + folded_layers; i++) {
        var coefficient = 0.0;
        var formula = MANDELBULB;
        var settings = vec4<f32>(0.0);
        var rotation = camera.menger_rotation;
        if (i < folded_layers) {
            coefficient = primary_coefficient;
            formula = camera.parameters.y;
            if (formula == KIFS) {
                settings = camera.kifs;
            } else if (formula == AMAZING_SURFACE) {
                settings = camera.amazing_surface;
            } else {
                settings = camera.pseudo_kleinian;
            }
        } else {
            let stage = camera.extra_mutations[i - folded_layers];
            coefficient = stage.blend.x;
            formula = stage.blend.y;
            settings = stage.settings;
            rotation = stage.rotation;
        }
        var distance = 0.0;
        if (formula == KIFS) {
            distance = kifs_distance(p, settings);
        } else if (formula == AMAZING_SURFACE) {
            distance = amazing_surface_distance(p, iterations, settings);
        } else if (formula == PSEUDO_KLEINIAN) {
            distance = pseudo_kleinian_distance(p, iterations, settings);
        } else if (formula == MENGER) {
            distance = menger_distance(p, settings, rotation);
        } else if (formula == MANDELBOX) {
            distance = box_distance(p, iterations, settings);
        } else {
            distance = bulb_distance(p, iterations, settings);
        }
        result += coefficient * distance;
    }
    return result;
}
// Give nearby samples more detail and blend adjacent levels continuously.
fn distance_estimate(p: vec3<f32>, iterations: f32) -> f32 {
    return distance_at_level(p, detail_level(p, iterations));
}
// The sliders specify close-range values; distant samples use a larger
// surface tolerance and a broader normal filter.
fn distance_factor(travel: f32) -> f32 {
    return 1.0 + min(max(travel, 0.0), 10.0) * 0.5;
}
fn adaptive_normal_step(travel: f32) -> f32 {
    let smoothness = camera.display.y * distance_factor(travel);
    let pixel_footprint = camera.filtering.x * travel / max(camera.parameters.z * focal_scale(), 1.0);
    return max(smoothness, pixel_footprint);
}
fn normal(p: vec3<f32>, e: f32, iterations: f32, fallback: vec3<f32>) -> vec3<f32> {
    let level = detail_level(p, iterations);
    var gradient = vec3<f32>(0.0);
    // A single call site avoids replicating the full distance pipeline per sample.
    for (var sample = 0u; sample < 6u; sample++) {
        let axis = sample / 2u;
        let side = select(1.0, -1.0, sample % 2u != 0u);
        var offset = vec3<f32>(0.0);
        offset[axis] = side * e;
        gradient[axis] += side * distance_at_level(p + offset, level);
    }
    let magnitude2 = dot(gradient, gradient);
    if (!(magnitude2 > 1e-24) || magnitude2 > 1e30) { return fallback; }
    return gradient * inverseSqrt(magnitude2);
}
// Approximate occlusion from nearby distance samples; this is not a path-traced result.
fn ambient_occlusion(p: vec3<f32>, n: vec3<f32>, iterations: f32, side: f32) -> f32 {
    var sum = 0.0;
    for (var i = 1; i <= 4; i++) {
        let h = camera.appearance.w * f32(i) * 0.25;
        let d = max(side * distance_estimate(p + n * h, iterations), 0.0);
        sum += max(h - d, 0.0) / h;
    }
    return clamp(1.0 - sum * 0.18 * camera.appearance.z, 0.05, 1.0);
}
// Procedural camera-oriented matcap-like shading, without image textures.
fn appearance(n: vec3<f32>, forward: vec3<f32>, right: vec3<f32>, up: vec3<f32>, view_direction: vec3<f32>) -> vec3<f32> {
    let mode = camera.appearance.x;
    if (mode > 3.5 && mode < 4.5) {
        let azimuth = camera.light_settings.x;
        let elevation = camera.light_settings.y;
        let light_direction = normalize(vec3<f32>(sin(azimuth) * cos(elevation), sin(elevation), cos(azimuth) * cos(elevation)));
        let light_color = camera.light_color_metallic.rgb;
        let metallic = camera.light_color_metallic.w;
        let base = camera.surface.rgb;
        let roughness = clamp(camera.surface.w, 0.05, 1.0);
        let ndotl = max(dot(n, light_direction), 0.0);
        let half_vector = normalize(light_direction + view_direction + vec3<f32>(0.000001));
        let ndoth = max(dot(n, half_vector), 0.0);
        let exponent = mix(256.0, 2.0, roughness * roughness);
        let specular = pow(ndoth, exponent) * (exponent + 2.0) / 8.0;
        let specular_color = mix(vec3<f32>(0.04), base, metallic);
        let diffuse = base * (1.0 - metallic) * ndotl;
        let direct = (diffuse + specular_color * specular * ndotl) * light_color * camera.light_settings.z;
        let ambient = base * camera.light_settings.w;
        return ambient + direct;
    }
    if (mode < 0.5) {
        let light = max(dot(n, normalize(vec3<f32>(0.5, 0.8, 1.0))), 0.0);
        return vec3<f32>(0.18, 0.48, 0.62) * (0.2 + 0.8 * light);
    }
    // Sample the inverse rotation so positive angles turn the matcap counterclockwise.
    let angle = camera.matcap.x;
    let c = cos(angle);
    let s = sin(angle);
    let x = dot(n, right);
    let y = dot(n, up);
    let view_n = vec3<f32>(c * x + s * y, -s * x + c * y, dot(n, -forward));
    let diffuse = max(dot(view_n, normalize(vec3<f32>(-0.4, 0.7, 0.8))), 0.0);
    let rim = pow(1.0 - clamp(abs(view_n.z), 0.0, 1.0), 2.0);
    let spec = pow(max(dot(view_n, normalize(vec3<f32>(-0.35, 0.55, 0.85))), 0.0), 24.0);
    if (mode < 1.5) { return vec3<f32>(0.48, 0.53, 0.6) * (0.3 + 0.7 * diffuse) + vec3<f32>(0.5) * spec + vec3<f32>(0.1) * rim; }
    if (mode < 2.5) { return vec3<f32>(0.47, 0.19, 0.07) * (0.25 + 0.8 * diffuse) + vec3<f32>(1.0, 0.65, 0.32) * spec + vec3<f32>(0.16, 0.06, 0.02) * rim; }
    if (mode < 3.5) { return vec3<f32>(0.15, 0.08, 0.32) * (0.3 + 0.8 * diffuse) + vec3<f32>(0.08, 0.5, 0.7) * rim + vec3<f32>(0.75, 0.45, 0.9) * spec; }
    let broad_spec = pow(max(dot(view_n, normalize(vec3<f32>(-0.35, 0.55, 0.85))), 0.0), 8.0);
    let sharp_spec = pow(max(dot(view_n, normalize(vec3<f32>(-0.35, 0.55, 0.85))), 0.0), 64.0);
    if (mode < 5.5) { return vec3<f32>(0.50, 0.33, 0.10) * (0.25 + 0.85 * diffuse) + vec3<f32>(1.0, 0.83, 0.42) * (0.5 * broad_spec + 0.6 * sharp_spec); }
    if (mode < 6.5) { return vec3<f32>(0.22, 0.27, 0.34) * (0.22 + 0.8 * diffuse) + vec3<f32>(0.7, 0.8, 1.0) * (0.45 * broad_spec + sharp_spec) + vec3<f32>(0.15, 0.25, 0.4) * rim; }
    if (mode < 7.5) { return vec3<f32>(0.54, 0.27, 0.18) * (0.35 + 0.65 * diffuse) + vec3<f32>(0.18, 0.10, 0.07) * broad_spec; }
    if (mode < 8.5) { return vec3<f32>(0.04, 0.27, 0.22) * (0.3 + 0.8 * diffuse) + vec3<f32>(0.20, 0.95, 0.70) * (0.32 * broad_spec + 0.5 * sharp_spec) + vec3<f32>(0.04, 0.18, 0.12) * rim; }
    return vec3<f32>(0.12, 0.06, 0.18) * (0.3 + 0.7 * diffuse) + vec3<f32>(0.70, 0.20, 0.60) * broad_spec + vec3<f32>(0.16, 0.50, 0.95) * rim + vec3<f32>(0.9, 0.8, 1.0) * sharp_spec;
}
const BULB_INTERIOR_ITERATIONS: i32 = 12;

fn inside_escape_fractal(p: vec3<f32>) -> bool {
    return camera.mutation_stack.y == 0.0 && camera.parameters.y == MANDELBULB
        && bulb_escape(p, BULB_INTERIOR_ITERATIONS) == BULB_INTERIOR_ITERATIONS;
}
fn focal_scale() -> f32 {
    return select(1.8, camera.lens.x, camera.lens.x > 0.0);
}

fn fogged(color: vec3<f32>, distance: f32) -> vec4<f32> {
    let visibility = exp(-camera.lens.w * max(distance - camera.fog.w, 0.0));
    return vec4<f32>(mix(camera.fog.rgb, color, visibility), 1.0);
}

fn trace(uv: vec2<f32>, lens_offset: vec2<f32>, focus_only: bool) -> vec4<f32> {
    let background = select(camera.background_color.rgb, vec3<f32>(0.015, 0.02, 0.04), camera.parameters.w > 0.5);
    if (camera.orientation.z > 0.5) {
        let p = vec3<f32>(uv.x * camera.position_aspect.w * 1.6, uv.y * 1.6, camera.orientation.w);
        var escaped = 0;
        if (camera.mutation_stack.y > 0.0) {
            escaped = select(0, 9, distance_at_level(p, 9.0) < 0.0);
        } else {
            escaped = bulb_escape(p, 9);
        }
        if (escaped == 9) { return vec4<f32>(0.1, 0.7, 0.85, 1.0); }
        let shade = f32(escaped) / 9.0;
        return vec4<f32>(background + shade * vec3<f32>(0.2, 0.12, 0.04), 1.0);
    }
    // All distance, normal and AO samples use the same spatial detail rule.
    let iterations = 11.0;
    let yaw = camera.orientation.x;
    let pitch = camera.orientation.y;
    let forward = vec3<f32>(sin(yaw) * cos(pitch), sin(pitch), -cos(yaw) * cos(pitch));
    let base_right = vec3<f32>(cos(yaw), 0.0, sin(yaw));
    let base_up = cross(base_right, forward);
    let roll = camera.filtering.y;
    let right = base_right * cos(roll) + base_up * sin(roll);
    let up = base_up * cos(roll) - base_right * sin(roll);
    let primary_ray = forward * focal_scale() + right * uv.x * camera.position_aspect.w + up * uv.y;
    let lens_shift = right * lens_offset.x + up * lens_offset.y;
    let origin = camera.position_aspect.xyz + lens_shift;
    var ray = normalize(primary_ray);
    if (dot(lens_offset, lens_offset) > 0.0) {
        ray = normalize(primary_ray * (camera.lens.y / focal_scale()) - lens_shift);
    }
    var travel = 0.0;
    var hit = false;
    let quality = clamp(camera.display.z, 0.5, 8.0);
    if (inside_escape_fractal(origin)) {
        let refined = focus_only || camera.filtering.w > 0.5;
        let scan_budget = select(64, 112, refined);
        let min_step = select(0.0015, 0.00065, refined) / sqrt(quality);
        let max_step = select(0.16, 0.07, refined);
        let growth = select(0.24, 0.16, refined);
        var previous_travel = 0.0;
        // Use escape-time membership instead of two distance evaluations per step.
        // Nearby openings receive the smallest steps, with a bounded search farther out.
        for (var i = 0; i < scan_budget; i++) {
            previous_travel = travel;
            travel = min(5.12, travel + clamp(travel * growth, min_step, max_step));
            if (!inside_escape_fractal(origin + ray * travel)) { hit = true; break; }
            if (travel >= 5.12) { break; }
        }
        if (!hit) {
            if (camera.parameters.w > 0.5) { return vec4<f32>(1.0, 0.1, 0.05, 1.0); }
            if (focus_only) { return vec4<f32>(0.0); }
            return fogged(background, 5.12);
        }
        if (!focus_only && camera.parameters.w > 0.5) { return vec4<f32>(0.25, 0.65, 0.75, 1.0); }
        // Refine the transition. Features thinner than the step may still be missed.
        var low = previous_travel;
        var high = travel;
        for (var j = 0; j < 6; j++) {
            let mid = (low + high) * 0.5;
            if (!inside_escape_fractal(origin + ray * mid)) { high = mid; }
            else { low = mid; }
        }
        travel = high;
        if (focus_only) { return vec4<f32>(max(travel * dot(ray, forward), 0.0001)); }
        let n = -normal(origin + ray * travel, adaptive_normal_step(travel), iterations, ray);
        var color = appearance(n, forward, right, up, -ray);
        if (camera.appearance.y > 0.5) { color *= ambient_occlusion(origin + ray * travel, n, iterations, 1.0); }
        return fogged(color * exp2(camera.display.x), travel);
    }
    var closest_distance = 1e20;
    var closest_travel = 0.0;
    var exhausted = true;
    var approximate_hit = false;
    var extended_hit = false;
    // March to the next boundary when starting inside a signed field.
    let signed_field = camera.mutation_stack.y > 0.0 || camera.parameters.y == MENGER || camera.parameters.y == KIFS || camera.parameters.y == PSEUDO_KLEINIAN;
    let inside_field = signed_field && distance_estimate(origin, iterations) < 0.0;
    let side = select(1.0, -1.0, inside_field);
    // Only rays which have not converged continue beyond the regular budget.
    // The extra work is limited to still-frame rendering and export.
    // A mixed signed field can need more steps to resolve narrow surfaces.
    let preview_budget = select(512, 256, camera.matcap.y > 0.0);
    let step_budget = select(preview_budget, 3072, camera.filtering.w > 0.5);
    for (var i = 0; i < step_budget; i++) {
        let d = side * distance_estimate(origin + ray * travel, iterations);
        let local_quality = quality / distance_factor(travel);
        let epsilon = max(0.00008, travel * 0.00008) / local_quality;
        if (d < closest_distance) { closest_distance = d; closest_travel = travel; }
        if (d < epsilon) {
            hit = true;
            extended_hit = i >= 768;
            exhausted = false;
            break;
        }
        travel += d * 0.8;
        // Extend the search distance without changing the step budget, so
        // distant openings can be distinguished from the old range limit.
        if (travel > 30.0) { exhausted = false; break; }
    }
    // Approximate a hit only when the step budget runs out very near a surface.
    // This reduces false background pixels but is not an exact intersection.
    if (!hit && exhausted && closest_distance < max(0.00016, closest_travel * 0.00016) * distance_factor(closest_travel) / quality) {
        travel = closest_travel;
        hit = true;
        approximate_hit = true;
    }
    if (!hit) {
        if (focus_only) { return vec4<f32>(0.0); }
        if (camera.parameters.w > 0.5 && exhausted) {
            return vec4<f32>(1.0, 0.1, 0.05, 1.0);
        }
        if (camera.parameters.w > 0.5) { return vec4<f32>(background, 1.0); }
        return fogged(background, 30.0);
    }
    if (focus_only) { return vec4<f32>(max(travel * dot(ray, forward), 0.0001)); }
    if (camera.parameters.w > 0.5) {
        if (approximate_hit) { return vec4<f32>(1.0, 0.75, 0.0, 1.0); }
        if (extended_hit) { return vec4<f32>(0.1, 1.0, 0.2, 1.0); }
        return vec4<f32>(0.25, 0.65, 0.75, 1.0);
    }
    // Sample the normal over roughly a pixel footprint to reduce unstable
    // high-frequency shading when the fractal has subpixel details.
    let normal_step = adaptive_normal_step(travel);
    let n = side * normal(origin + ray * travel, normal_step, iterations, -side * ray);
    var color = appearance(n, forward, right, up, -ray);
    if (camera.appearance.y > 0.5) { color *= ambient_occlusion(origin + ray * travel, n, iterations, side); }
    return fogged(color * exp2(camera.display.x), travel);
}

fn viewport_uv(uv: vec2<f32>) -> vec2<f32> {
    let scale = select(1.0, camera.viewport.z, camera.viewport.z > 0.0);
    return (uv - camera.viewport.xy) * scale;
}

fn sample_lens(index: f32) -> vec2<f32> {
    if (camera.lens.z <= 0.0 || index == 0.0) { return vec2<f32>(0.0); }
    let sample = fract(vec2<f32>(0.5) + index * vec2<f32>(0.43857902, 0.3281734));
    let angle = 6.2831853 * sample.y;
    return vec2<f32>(cos(angle), sin(angle)) * (sqrt(sample.x) * camera.lens.z);
}

@fragment fn fs_focus() -> @location(0) f32 {
    return trace(camera.render_region.xy, vec2<f32>(0.0), true).x;
}

fn sample_viewport(uv: vec2<f32>, jitter: vec2<f32>, lens_offset: vec2<f32>) -> vec4<f32> {
    var columns = 1;
    if (camera.parameters.w <= 0.5 && camera.display.w >= 0.5) {
        // Use one interactive sample inside the bulb.
        if (camera.orientation.z > 0.5 || !inside_escape_fractal(camera.position_aspect.xyz)) {
            columns = select(2, 4, camera.display.w > 1.5);
        }
    }
    let rows = select(1, 2, columns > 1 && camera.display.w >= 1.0);
    let height = max(camera.parameters.z * max(camera.viewport.z, 1.0), 1.0);
    let pixel = vec2<f32>(2.0 / (height * camera.position_aspect.w), 2.0 / height);
    let count = rows * columns;
    var sum = vec4<f32>(0.0);
    // One call site keeps the native shader compiler from duplicating the ray marcher.
    for (var sample = 0; sample < count; sample++) {
        let cell = vec2<f32>(f32(sample % columns), f32(sample / columns));
        var offset = (cell + vec2<f32>(0.5)) / vec2<f32>(f32(columns), f32(rows)) - vec2<f32>(0.5);
        if (columns == 2 && rows == 1) { offset.y = offset.x; }
        sum += trace(viewport_uv(uv + pixel * (offset + jitter)), lens_offset, false);
    }
    return sum / f32(count);
}

@fragment fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    return apply_contrast(sample_viewport(in.uv, vec2<f32>(0.0), vec2<f32>(0.0)), camera.viewport.w);
}

// Each still-frame pass contributes one distinct subpixel sample.
@fragment fn fs_progress(in: VertexOut) -> @location(0) vec4<f32> {
    let index = select(camera.filtering.z, 0.0, camera.parameters.w > 0.5);
    let offset = fract(vec2<f32>(0.5) + index * vec2<f32>(0.754877666, 0.569840296)) - vec2<f32>(0.5);
    return sample_viewport(in.uv, offset, sample_lens(index));
}

// The export renderer renders independent tiles. Pixel coordinates remain
// relative to the full output image, so neighbouring tiles line up exactly.
@fragment fn fs_export(in: VertexOut) -> @location(0) vec4<f32> {
    let full_size = camera.render_region.zw;
    let pixel_position = camera.render_region.xy + in.position.xy;
    let uv = vec2<f32>(2.0 * pixel_position.x / full_size.x - 1.0,
                       1.0 - 2.0 * pixel_position.y / full_size.y);
    let index = camera.filtering.z;
    let offset = fract(vec2<f32>(0.5) + index * vec2<f32>(0.754877666, 0.569840296)) - vec2<f32>(0.5);
    let jitter = vec2<f32>(2.0 * offset.x / full_size.x, 2.0 * offset.y / full_size.y);
    return trace(uv + jitter, sample_lens(index), false);
}
