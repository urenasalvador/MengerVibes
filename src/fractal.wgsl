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
    preview: vec4<f32>,
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
    environment_and_light_radius: vec4<f32>,
    path_settings: vec4<f32>,
    path_display: vec4<f32>,
    surface_finish: vec4<f32>,
    surface_emission: vec4<f32>,
}
@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var panorama: texture_2d<f32>;
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
    if (camera.preview.y > 0.0) {
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
fn appearance(point: vec3<f32>, n: vec3<f32>, view_direction: vec3<f32>, iterations: f32, side: f32, camera_distance: f32, light_sample: vec2<f32>, material: PathMaterial) -> vec3<f32> {
    let azimuth = camera.light_settings.x;
    let elevation = camera.light_settings.y;
    var light_direction = normalize(vec3<f32>(sin(azimuth) * cos(elevation), sin(elevation), cos(azimuth) * cos(elevation)));
    if (light_sample.x >= 0.0) {
        let axis = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0, 0.0, 1.0), abs(light_direction.y) > 0.95);
        let tangent = normalize(cross(axis, light_direction));
        let bitangent = cross(light_direction, tangent);
        let delta = light_sample.x * camera.environment_and_light_radius.w;
        let radial = sqrt(delta * (2.0 - delta));
        let angle = light_sample.y * 6.2831853;
        light_direction = (1.0 - delta) * light_direction + radial * (cos(angle) * tangent + sin(angle) * bitangent);
    }
    let light_color = camera.light_color_metallic.rgb;
    let environment = camera.environment_and_light_radius.rgb;
    var ndotl = max(dot(n, light_direction), 0.0);
    if (ndotl > 0.0 && camera.light_settings.z > 0.0 && camera.display.w > 0.5) {
        if (directional_shadow(point, n, light_direction, iterations, side, camera_distance)) {
            ndotl = 0.0;
        }
    }
    let ambient = material.base * environment * coat_transmission(material.coat, dot(n, view_direction));
    if (ndotl <= 0.0 || camera.light_settings.z <= 0.0) { return ambient; }
    let direct = path_bsdf(material, n, view_direction, light_direction).value
        * light_color * (camera.light_settings.z * PATH_PI * ndotl);
    return ambient + direct;
}
const BULB_INTERIOR_ITERATIONS: i32 = 12;

fn inside_escape_fractal(p: vec3<f32>) -> bool {
    return camera.mutation_stack.y == 0.0 && camera.parameters.y == MANDELBULB
        && bulb_escape(p, BULB_INTERIOR_ITERATIONS) == BULB_INTERIOR_ITERATIONS;
}

fn directional_shadow(point: vec3<f32>, n: vec3<f32>, direction: vec3<f32>, iterations: f32, side: f32, camera_distance: f32) -> bool {
    let quality = clamp(camera.display.z, 0.5, 8.0);
    let tolerance = max(0.00008, camera_distance * 0.00008) * distance_factor(camera_distance) / quality;
    // Start on the visible side of the surface to avoid hitting the originating boundary.
    let bias = 2.0 * tolerance;
    let origin = point + n * bias;
    var travel = bias;
    let refined = camera.filtering.w > 0.5;
    if (side == 0.0) {
        // The Mandelbulb interior is traced with escape-time membership.
        let budget = select(64, 112, refined);
        let min_step = select(0.0015, 0.00065, refined) / sqrt(quality);
        let max_step = select(0.16, 0.07, refined);
        for (var i = 0; i < budget; i++) {
            if (!inside_escape_fractal(origin + direction * travel)) { return true; }
            if (travel >= 5.12) { break; }
            travel = min(5.12, travel + clamp(travel * 0.16, min_step, max_step));
        }
        return false;
    }
    let budget = select(96, 160, refined);
    for (var i = 0; i < budget && travel < 30.0; i++) {
        let distance = side * distance_estimate(origin + direction * travel, iterations);
        if (distance <= tolerance * 0.5) { return true; }
        travel += distance * 0.8;
    }
    return false;
}

fn pixel_random(seed: u32) -> f32 {
    var bits = seed;
    bits = (bits ^ (bits >> 16u)) * 0x7feb352du;
    bits = (bits ^ (bits >> 15u)) * 0x846ca68bu;
    return f32((bits ^ (bits >> 16u)) >> 8u) / 16777216.0;
}

fn ao_ray_sample(pixel: vec2<u32>, index: f32) -> vec2<f32> {
    if (camera.appearance.y <= 0.5 || camera.appearance.z <= 0.0) { return vec2<f32>(-1.0); }
    return pixel_sample(pixel, index, 0u);
}

fn light_ray_sample(pixel: vec2<u32>, index: f32) -> vec2<f32> {
    if (camera.display.w <= 0.5 || camera.environment_and_light_radius.w <= 0.0) { return vec2<f32>(-1.0); }
    return pixel_sample(pixel, index, 672827u);
}

fn pixel_sample(pixel: vec2<u32>, index: f32, seed_offset: u32) -> vec2<f32> {
    let seed = pixel.x * 1973u + pixel.y * 9277u + seed_offset;
    let scramble = vec2<f32>(pixel_random(seed), pixel_random(seed + 26699u));
    return fract(scramble + index * vec2<f32>(0.754877666, 0.569840296));
}

fn ray_traced_occlusion(point: vec3<f32>, n: vec3<f32>, iterations: f32, side: f32, camera_distance: f32, sample: vec2<f32>) -> f32 {
    let radius = camera.appearance.w;
    if (sample.x < 0.0 || radius <= 0.0) { return 1.0; }

    let reference = select(vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(0.0, 1.0, 0.0), abs(n.z) > 0.95);
    let tangent = normalize(cross(reference, n));
    let bitangent = cross(n, tangent);
    let angle = sample.y * 6.2831853;
    let radial = sqrt(sample.x);
    let ray = radial * (tangent * cos(angle) + bitangent * sin(angle)) + n * sqrt(1.0 - sample.x);
    let hit_tolerance = max(0.00008, camera_distance * 0.00008) * distance_factor(camera_distance)
        / clamp(camera.display.z, 0.5, 8.0);
    let bias = min(2.0 * hit_tolerance, radius * 0.125);
    let origin = point + n * bias;
    var distance = bias;
    var hit_distance = radius;
    if (side == 0.0) {
        // The Mandelbulb interior uses the same membership query as the primary ray.
        for (var i = 1; i <= 32; i++) {
            let fraction = f32(i) / 32.0;
            distance = mix(bias, radius, fraction * fraction);
            if (!inside_escape_fractal(origin + ray * distance)) {
                hit_distance = distance;
                break;
            }
        }
    } else {
        for (var i = 0; i < 64; i++) {
            if (distance >= radius) { break; }
            let step = side * distance_estimate(origin + ray * distance, iterations);
            if (step <= bias * 0.25) {
                hit_distance = distance;
                break;
            }
            distance += step * 0.8;
        }
    }
    // Let nearby blockers contribute fully, then soften the edge of the AO radius.
    let occlusion = 1.0 - smoothstep(0.5 * radius, radius, hit_distance);
    return max(0.05, 1.0 - camera.appearance.z * occlusion);
}
fn focal_scale() -> f32 {
    return select(1.8, camera.lens.x, camera.lens.x > 0.0);
}

fn fogged(color: vec3<f32>, distance: f32) -> vec4<f32> {
    let visibility = exp(-camera.lens.w * max(distance - camera.fog.w, 0.0));
    return vec4<f32>(mix(camera.fog.rgb, color, visibility), 1.0);
}

struct SurfaceGuide {
    albedo: vec3<f32>,
    normal: vec3<f32>,
}

fn shade_surface(point: vec3<f32>, n: vec3<f32>, view: vec3<f32>, iterations: f32, side: f32, travel: f32,
    world_to_view: mat3x3<f32>, occlusion_sample: vec2<f32>, light_sample: vec2<f32>, path_seed: u32,
    guide: ptr<function, SurfaceGuide>) -> vec4<f32> {
    let material = surface_material(surface_base_color(n, world_to_view));
    *guide = SurfaceGuide(material.base, n);
    var color: vec3<f32>;
    if (camera.path_settings.x > 0.5) {
        color = trace_light_path(point, n, view, side, travel, path_seed, material, world_to_view);
    } else {
        color = appearance(point, n, view, iterations, side, travel, light_sample, material)
            * ray_traced_occlusion(point, n, iterations, side, travel, occlusion_sample) + material_emission(n, view);
    }
    return fogged(color * exp2(camera.display.x), travel);
}

fn trace(uv: vec2<f32>, lens_offset: vec2<f32>, focus_only: bool, occlusion_sample: vec2<f32>, light_sample: vec2<f32>, path_seed: u32, guide: ptr<function, SurfaceGuide>) -> vec4<f32> {
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
    let roll = camera.filtering.y;
    let forward = vec3<f32>(sin(yaw) * cos(pitch), sin(pitch), -cos(yaw) * cos(pitch));
    let base_right = vec3<f32>(cos(yaw), 0.0, sin(yaw));
    let base_up = cross(base_right, forward);
    let right = base_right * cos(roll) + base_up * sin(roll);
    let up = base_up * cos(roll) - base_right * sin(roll);
    let world_to_view = transpose(mat3x3<f32>(right, up, -forward));
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
            return fogged(camera_background(background, ray), 5.12);
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
        let point = origin + ray * travel;
        let n = -normal(point, adaptive_normal_step(travel), iterations, ray);
        return shade_surface(point, n, -ray, iterations, 0.0, travel, world_to_view, occlusion_sample, light_sample, path_seed, guide);
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
    let preview_budget = select(512, 256, camera.preview.y > 0.0);
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
        return fogged(camera_background(background, ray), 30.0);
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
    let point = origin + ray * travel;
    let n = side * normal(point, normal_step, iterations, -side * ray);
    return shade_surface(point, n, -ray, iterations, side, travel, world_to_view, occlusion_sample, light_sample, path_seed, guide);
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
    var guide: SurfaceGuide;
    return trace(camera.render_region.xy, vec2<f32>(0.0), true, vec2<f32>(-1.0), vec2<f32>(-1.0), 0u, &guide).x;
}

@fragment fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    var guide: SurfaceGuide;
    return apply_contrast(viewport_samples(in, 0.0, false, &guide), camera.viewport.w);
}

fn viewport_samples(in: VertexOut, first_index: f32, progressive: bool, guide_out: ptr<function, SurfaceGuide>) -> vec4<f32> {
    let height = max(camera.parameters.z * max(camera.viewport.z, 1.0), 1.0);
    let pixel = vec2<f32>(2.0 / (height * camera.position_aspect.w), 2.0 / height);
    let position = vec2<u32>(in.position.xy);
    let count = u32(clamp(camera.path_display.z, 1.0, 4.0));
    var color = vec4<f32>(0.0);
    for (var sample = 0u; sample < count; sample++) {
        var guide: SurfaceGuide;
        let index = first_index + f32(sample);
        let offset = fract(vec2<f32>(0.5) + index * vec2<f32>(0.754877666, 0.569840296)) - vec2<f32>(0.5);
        var ao_sample = vec2<f32>(-1.0);
        var light_sample = vec2<f32>(-1.0);
        if (progressive || count > 1u) {
            ao_sample = ao_ray_sample(position, index);
            light_sample = light_ray_sample(position, index);
        }
        color += trace(viewport_uv(in.uv + pixel * offset), sample_lens(index), false,
            ao_sample, light_sample, path_pixel_seed(position, index), &guide);
        (*guide_out).albedo += guide.albedo / f32(count);
        (*guide_out).normal += guide.normal / f32(count);
    }
    return color / f32(count);
}

@fragment fn fs_progress(in: VertexOut) -> @location(0) vec4<f32> {
    let index = select(camera.filtering.z, 0.0, camera.parameters.w > 0.5);
    var guide: SurfaceGuide;
    return viewport_samples(in, index, true, &guide);
}

// The export renderer renders independent tiles. Pixel coordinates remain
// relative to the full output image, so neighbouring tiles line up exactly.
fn export_sample(in: VertexOut, guide: ptr<function, SurfaceGuide>) -> vec4<f32> {
    let full_size = camera.render_region.zw;
    let pixel_position = camera.render_region.xy + in.position.xy;
    let uv = vec2<f32>(2.0 * pixel_position.x / full_size.x - 1.0,
                       1.0 - 2.0 * pixel_position.y / full_size.y);
    let index = camera.filtering.z;
    let offset = fract(vec2<f32>(0.5) + index * vec2<f32>(0.754877666, 0.569840296)) - vec2<f32>(0.5);
    let jitter = vec2<f32>(2.0 * offset.x / full_size.x, 2.0 * offset.y / full_size.y);
    let position = vec2<u32>(pixel_position);
    return trace(uv + jitter, sample_lens(index), false, ao_ray_sample(position, index), light_ray_sample(position, index), path_pixel_seed(position, index), guide);
}

@fragment fn fs_export(in: VertexOut) -> @location(0) vec4<f32> {
    var guide: SurfaceGuide;
    return export_sample(in, &guide);
}

struct ExportPasses {
    @location(0) color: vec4<f32>,
    @location(1) albedo: vec4<f32>,
    @location(2) normal: vec4<f32>,
}

@fragment fn fs_progress_denoise(in: VertexOut) -> ExportPasses {
    var guide: SurfaceGuide;
    let color = viewport_samples(in, camera.filtering.z, true, &guide);
    return ExportPasses(color, vec4<f32>(guide.albedo, 1.0), vec4<f32>(guide.normal, 1.0));
}

@fragment fn fs_export_denoise(in: VertexOut) -> ExportPasses {
    var guide: SurfaceGuide;
    let color = export_sample(in, &guide);
    return ExportPasses(color, vec4<f32>(guide.albedo, 1.0), vec4<f32>(guide.normal, 1.0));
}
