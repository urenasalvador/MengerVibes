// SPDX-License-Identifier: GPL-3.0-only

const PATH_PI: f32 = 3.14159265359;
const PATH_ITERATIONS: f32 = 11.0;

fn path_pixel_seed(pixel: vec2<u32>, sample_index: f32) -> u32 {
    return pixel.x * 1973u + pixel.y * 9277u + u32(sample_index) * 26699u;
}

fn path_random(seed: ptr<function, u32>) -> f32 {
    *seed += 0x9e3779b9u;
    return pixel_random(*seed);
}

fn path_basis(axis: vec3<f32>) -> mat3x3<f32> {
    let up = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0, 0.0, 1.0), abs(axis.y) > 0.95);
    let tangent = normalize(cross(up, axis));
    return mat3x3<f32>(tangent, cross(axis, tangent), axis);
}

fn path_direction(axis: vec3<f32>, cosine: f32, azimuth: f32) -> vec3<f32> {
    let radial = sqrt(max(0.0, 1.0 - cosine * cosine));
    let angle = 2.0 * PATH_PI * azimuth;
    return path_basis(axis) * vec3<f32>(radial * cos(angle), radial * sin(angle), cosine);
}

fn world_radiance(direction: vec3<f32>) -> vec3<f32> {
    let style = u32(camera.path_settings.y);
    let tint = camera.environment_and_light_radius.rgb;
    if (style == 0u) { return tint; }
    let angle = camera.path_settings.z;
    let c = cos(angle);
    let s = sin(angle);
    let ray = vec3<f32>(direction.x * c - direction.z * s, direction.y, direction.z * c + direction.x * s);
    if (style == 10u) {
        let dims = vec2<i32>(textureDimensions(panorama));
        let uv = vec2<f32>(fract(atan2(ray.x, -ray.z) / (2.0 * PATH_PI) + 0.5),
            acos(clamp(ray.y, -1.0, 1.0)) / PATH_PI);
        let texel = uv * vec2<f32>(dims) - 0.5;
        let base = vec2<i32>(floor(texel));
        let fraction = fract(texel);
        let x0 = (base.x + dims.x) % dims.x;
        let x1 = (x0 + 1) % dims.x;
        let y0 = clamp(base.y, 0, dims.y - 1);
        let y1 = clamp(base.y + 1, 0, dims.y - 1);
        let upper = mix(textureLoad(panorama, vec2<i32>(x0, y0), 0).rgb,
            textureLoad(panorama, vec2<i32>(x1, y0), 0).rgb, fraction.x);
        let lower = mix(textureLoad(panorama, vec2<i32>(x0, y1), 0).rgb,
            textureLoad(panorama, vec2<i32>(x1, y1), 0).rgb, fraction.x);
        return tint * mix(upper, lower, fraction.y);
    }
    var key = vec3<f32>(1.0, 0.94, 0.83);
    var fill = vec3<f32>(0.47, 0.64, 1.0);
    var rim = vec3<f32>(0.65, 0.82, 1.0);
    switch (style) {
        case 2u: { key = vec3<f32>(1.0, 0.98, 0.95); fill = vec3<f32>(0.77, 0.87, 1.0); }
        case 3u: { key = vec3<f32>(1.0, 0.5, 0.27); fill = vec3<f32>(0.68, 0.25, 0.13); rim = vec3<f32>(1.0, 0.81, 0.58); }
        case 4u: { key = vec3<f32>(0.69, 0.31, 1.0); fill = vec3<f32>(0.13, 0.65, 1.0); rim = vec3<f32>(1.0, 0.31, 0.69); }
        case 5u: { key = vec3<f32>(1.0, 0.74, 0.27); fill = vec3<f32>(0.75, 0.46, 0.12); rim = vec3<f32>(1.0, 0.94, 0.7); }
        case 6u: { key = vec3<f32>(0.82, 0.89, 1.0); fill = vec3<f32>(0.22, 0.37, 0.65); }
        case 7u: { key = vec3<f32>(1.0, 0.74, 0.62); fill = vec3<f32>(0.67, 0.42, 0.29); rim = key; }
        case 8u: { key = vec3<f32>(0.66, 1.0, 0.82); fill = vec3<f32>(0.19, 0.74, 0.47); rim = vec3<f32>(0.73, 0.97, 1.0); }
        case 9u: { key = vec3<f32>(0.72, 0.4, 1.0); fill = vec3<f32>(0.08, 0.81, 0.68); rim = vec3<f32>(0.96, 0.46, 0.74); }
        default: {}
    }
    let key_lobe = pow(max(dot(ray, vec3<f32>(-0.6, 0.8, 0.0)), 0.0), 9.0);
    let fill_lobe = pow(max(dot(ray, vec3<f32>(0.8, 0.0, 0.6)), 0.0), 3.0);
    let rim_lobe = pow(max(dot(ray, vec3<f32>(0.0, 0.6, -0.8)), 0.0), 16.0);
    return tint * (0.18 * fill + 4.0 * key * key_lobe + 0.9 * fill * fill_lobe + 2.0 * rim * rim_lobe);
}

fn camera_background(background: vec3<f32>, direction: vec3<f32>) -> vec3<f32> {
    if (camera.path_settings.x > 0.5 && camera.path_display.x > 0.5) {
        return world_radiance(direction) * exp2(camera.display.x);
    }
    return background;
}

struct PathMaterial {
    base: vec3<f32>,
    alpha: f32,
    metallic: f32,
    f0: vec3<f32>,
    specular_probability: f32,
    coat: f32,
    coat_alpha: f32,
}

struct PathBsdf {
    value: vec3<f32>,
    pdf: f32,
}

fn surface_base_color(normal: vec3<f32>, world_to_view: mat3x3<f32>) -> vec3<f32> {
    if (camera.appearance.x == 0.0) { return camera.surface.rgb; }
    let style = u32(camera.appearance.x) - 1u;
    let view_normal = rotated_matcap_normal(world_to_view * normal);
    var color = matcap_color(f32(style), normal, view_normal);
    if (style != 9u) {
        let direction = select(view_normal, normal, style == 0u);
        color = tint_matcap(color, direction, style);
    }
    return clamp(color, vec3<f32>(0.0), vec3<f32>(1.0));
}

fn tint_matcap(color: vec3<f32>, n: vec3<f32>, style: u32) -> vec3<f32> {
    // Columns give the accent colors for the X, Y and Z faces.
    var positive = mat3x3<f32>(vec3<f32>(0.10, 0.70, 0.90), vec3<f32>(0.32, 0.85, 0.64), vec3<f32>(0.29, 0.48, 0.94));
    var negative = mat3x3<f32>(vec3<f32>(0.34, 0.30, 0.78), vec3<f32>(0.15, 0.45, 0.90), vec3<f32>(0.14, 0.76, 0.77));
    switch (style) {
        case 1u: { // Pearl
            positive = mat3x3<f32>(vec3<f32>(0.79, 0.67, 0.96), vec3<f32>(0.97, 0.87, 0.69), vec3<f32>(0.67, 0.90, 0.82));
            negative = mat3x3<f32>(vec3<f32>(0.96, 0.69, 0.80), vec3<f32>(0.62, 0.73, 0.95), vec3<f32>(0.65, 0.85, 0.98));
        }
        case 2u: { // Copper
            positive = mat3x3<f32>(vec3<f32>(0.93, 0.32, 0.17), vec3<f32>(0.86, 0.60, 0.27), vec3<f32>(0.36, 0.68, 0.55));
            negative = mat3x3<f32>(vec3<f32>(0.76, 0.45, 0.32), vec3<f32>(0.52, 0.28, 0.26), vec3<f32>(0.91, 0.49, 0.27));
        }
        case 3u: { // Nebula
            positive = mat3x3<f32>(vec3<f32>(0.71, 0.23, 0.93), vec3<f32>(0.16, 0.78, 0.92), vec3<f32>(0.39, 0.29, 0.96));
            negative = mat3x3<f32>(vec3<f32>(0.87, 0.19, 0.58), vec3<f32>(0.19, 0.37, 0.81), vec3<f32>(0.54, 0.28, 0.73));
        }
        case 5u: { // Gold
            positive = mat3x3<f32>(vec3<f32>(0.93, 0.55, 0.17), vec3<f32>(0.96, 0.83, 0.44), vec3<f32>(0.75, 0.73, 0.27));
            negative = mat3x3<f32>(vec3<f32>(0.84, 0.39, 0.16), vec3<f32>(0.67, 0.55, 0.22), vec3<f32>(0.97, 0.70, 0.32));
        }
        case 6u: { // Steel
            positive = mat3x3<f32>(vec3<f32>(0.46, 0.61, 0.86), vec3<f32>(0.83, 0.77, 0.63), vec3<f32>(0.49, 0.74, 0.72));
            negative = mat3x3<f32>(vec3<f32>(0.67, 0.61, 0.79), vec3<f32>(0.36, 0.49, 0.70), vec3<f32>(0.62, 0.72, 0.85));
        }
        case 7u: { // Clay
            positive = mat3x3<f32>(vec3<f32>(0.79, 0.34, 0.24), vec3<f32>(0.91, 0.64, 0.39), vec3<f32>(0.62, 0.43, 0.39));
            negative = mat3x3<f32>(vec3<f32>(0.69, 0.48, 0.28), vec3<f32>(0.52, 0.30, 0.24), vec3<f32>(0.82, 0.45, 0.32));
        }
        case 8u: { // Jade
            positive = mat3x3<f32>(vec3<f32>(0.15, 0.76, 0.57), vec3<f32>(0.62, 0.85, 0.31), vec3<f32>(0.19, 0.69, 0.83));
            negative = mat3x3<f32>(vec3<f32>(0.22, 0.56, 0.42), vec3<f32>(0.15, 0.45, 0.62), vec3<f32>(0.27, 0.75, 0.38));
        }
        default: {} // Original
    }
    let weights = n * n;
    let accent = weights.x * select(negative[0], positive[0], n.x >= 0.0)
        + weights.y * select(negative[1], positive[1], n.y >= 0.0)
        + weights.z * select(negative[2], positive[2], n.z >= 0.0);
    let luma = vec3<f32>(0.2126, 0.7152, 0.0722);
    return mix(color, accent * (dot(color, luma) / dot(accent, luma)), 0.65);
}

fn rotated_matcap_normal(n: vec3<f32>) -> vec3<f32> {
    // Sample the inverse rotation so positive angles turn the matcap counterclockwise.
    let angle = camera.preview.x;
    let c = cos(angle);
    let s = sin(angle);
    return vec3<f32>(c * n.x + s * n.y, -s * n.x + c * n.y, n.z);
}

// The original procedural matcaps supply the Principled base color.
fn matcap_color(mode: f32, world_normal: vec3<f32>, view_n: vec3<f32>) -> vec3<f32> {
    if (mode < 0.5) {
        let light = max(dot(world_normal, normalize(vec3<f32>(0.5, 0.8, 1.0))), 0.0);
        return vec3<f32>(0.18, 0.48, 0.62) * (0.2 + 0.8 * light);
    }
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

fn surface_material(base: vec3<f32>) -> PathMaterial {
    let roughness = clamp(camera.surface.w, 0.05, 1.0);
    let metal = camera.light_color_metallic.w;
    let ior = camera.surface_finish.x;
    let ratio = (ior - 1.0) / max(ior + 1.0, 1.0);
    let dielectric = select(0.04, clamp(2.0 * camera.surface_finish.y * ratio * ratio, 0.0, 1.0), ior >= 1.0);
    let f0 = mix(vec3<f32>(dielectric), base, metal);
    let specular_probability = select(0.0, mix(0.35, 1.0, metal), dielectric > 0.0 || metal > 0.0);
    let coat_roughness = clamp(camera.surface_finish.w, 0.05, 1.0);
    return PathMaterial(base, roughness * roughness, metal, f0, specular_probability,
        camera.surface_finish.z, coat_roughness * coat_roughness);
}

fn fresnel_schlick(f0: vec3<f32>, cosine: f32) -> vec3<f32> {
    return f0 + (vec3<f32>(1.0) - f0) * pow(1.0 - clamp(cosine, 0.0, 1.0), 5.0);
}

fn coat_transmission(weight: f32, cosine: f32) -> f32 {
    return 1.0 - weight * fresnel_schlick(vec3<f32>(0.04), cosine).r;
}

fn material_emission(n: vec3<f32>, view: vec3<f32>) -> vec3<f32> {
    return camera.surface_emission.rgb * coat_transmission(camera.surface_finish.z, dot(n, view));
}

fn microfacet(alpha: f32, fresnel: vec3<f32>, no_l: f32, no_v: f32, no_h: f32, vo_h: f32) -> PathBsdf {
    let alpha2 = alpha * alpha;
    let denominator = (1.0 - no_h * no_h) + alpha2 * no_h * no_h;
    let distribution = alpha2 / (PATH_PI * denominator * denominator);
    let smith_l = no_l + sqrt(alpha2 + (1.0 - alpha2) * no_l * no_l);
    let smith_v = no_v + sqrt(alpha2 + (1.0 - alpha2) * no_v * no_v);
    return PathBsdf(fresnel * distribution / (smith_v * smith_l), distribution * no_h / (4.0 * vo_h));
}

fn path_bsdf(material: PathMaterial, n: vec3<f32>, view: vec3<f32>, light: vec3<f32>) -> PathBsdf {
    let no_l = max(dot(n, light), 0.0);
    let no_v = max(dot(n, view), 0.0);
    if (min(no_l, no_v) <= 0.0) { return PathBsdf(vec3<f32>(0.0), 0.0); }
    let half_vector = normalize(view + light);
    let no_h = clamp(dot(n, half_vector), 0.0, 1.0);
    let vo_h = max(dot(view, half_vector), 0.00001);
    let fresnel = select(vec3<f32>(0.0), fresnel_schlick(material.f0, vo_h), material.specular_probability > 0.0);
    let specular = microfacet(material.alpha, fresnel, no_l, no_v, no_h, vo_h);
    let diffuse = material.base * (1.0 - material.metallic) * (vec3<f32>(1.0) - fresnel) / PATH_PI;
    let base = PathBsdf(diffuse + specular.value, mix(no_l / PATH_PI, specular.pdf, material.specular_probability));
    if (material.coat == 0.0) { return base; }
    let coat = microfacet(material.coat_alpha, fresnel_schlick(vec3<f32>(0.04), vo_h), no_l, no_v, no_h, vo_h);
    let transmission = coat_transmission(material.coat, no_l) * coat_transmission(material.coat, no_v);
    return PathBsdf(base.value * transmission + coat.value * material.coat, mix(base.pdf, coat.pdf, 0.25 * material.coat));
}

// Negative distances distinguish a clear escape (-1) from an inconclusive march (-2).
fn path_intersection(origin: vec3<f32>, ray: vec3<f32>, side: f32, quality: f32, refined: bool) -> f32 {
    var travel = 0.0;
    if (side == 0.0) {
        let budget = select(64, 112, refined);
        let min_step = select(0.0015, 0.00065, refined) / sqrt(quality);
        let max_step = select(0.16, 0.07, refined);
        for (var i = 0; i < budget; i++) {
            let previous = travel;
            travel = min(5.12, travel + clamp(travel * 0.16, min_step, max_step));
            if (!inside_escape_fractal(origin + ray * travel)) {
                var low = previous;
                var high = travel;
                for (var j = 0; j < 6; j++) {
                    let mid = (low + high) * 0.5;
                    if (inside_escape_fractal(origin + ray * mid)) { low = mid; } else { high = mid; }
                }
                return high;
            }
            if (travel >= 5.12) { return -1.0; }
        }
        return -2.0;
    }
    let budget = select(128, 320, refined);
    for (var i = 0; i < budget; i++) {
        let distance = side * distance_estimate(origin + ray * travel, PATH_ITERATIONS);
        let tolerance = max(0.00008, travel * 0.00008) * distance_factor(travel) / quality;
        if (distance < tolerance) { return travel; }
        travel += distance * 0.8;
        if (travel > 30.0) { return -1.0; }
    }
    return -2.0;
}

fn trace_light_path(first_point: vec3<f32>, first_normal: vec3<f32>, first_view: vec3<f32>, side: f32, first_distance: f32, pixel_seed: u32,
    first_material: PathMaterial, world_to_view: mat3x3<f32>) -> vec3<f32> {
    let refined = camera.filtering.w > 0.5;
    let limit = clamp(u32(select(camera.path_display.w, camera.path_settings.w, refined)), 1u, 8u);
    let quality = clamp(camera.display.z, 0.5, 8.0);
    var material = first_material;
    let azimuth = camera.light_settings.x;
    let elevation = camera.light_settings.y;
    let sun = vec3<f32>(sin(azimuth) * cos(elevation), sin(elevation), cos(azimuth) * cos(elevation));
    let sun_radiance = camera.light_color_metallic.rgb * camera.light_settings.z * PATH_PI;
    var seed = pixel_seed;
    var point = first_point;
    var n = first_normal;
    var view = first_view;
    var travel = first_distance;
    var throughput = vec3<f32>(1.0);
    var radiance = vec3<f32>(0.0);

    for (var bounce = 0u; bounce < limit; bounce++) {
        radiance += throughput * material_emission(n, view);
        if (camera.light_settings.z > 0.0) {
            var light = sun;
            if (refined && camera.display.w > 0.5 && camera.environment_and_light_radius.w > 0.0) {
                let delta = camera.environment_and_light_radius.w * path_random(&seed);
                light = path_direction(sun, 1.0 - delta, path_random(&seed));
            }
            let cosine = max(dot(n, light), 0.0);
            if (cosine > 0.0) {
                let blocked = camera.display.w > 0.5 && directional_shadow(point, n, light, PATH_ITERATIONS, side, travel);
                if (!blocked) { radiance += throughput * path_bsdf(material, n, view, light).value * sun_radiance * cosine; }
            }
        }

        let lobe = path_random(&seed);
        let radial_sample = path_random(&seed);
        let angle_sample = path_random(&seed);
        var ray = path_direction(n, sqrt(1.0 - radial_sample), angle_sample);
        let coat_probability = 0.25 * material.coat;
        if (lobe < mix(material.specular_probability, 1.0, coat_probability)) {
            let alpha = select(material.alpha, material.coat_alpha, lobe < coat_probability);
            let alpha2 = alpha * alpha;
            let cosine = sqrt((1.0 - radial_sample) / (1.0 - radial_sample + alpha2 * radial_sample));
            ray = reflect(-view, path_direction(n, cosine, angle_sample));
        }
        let bsdf = path_bsdf(material, n, view, ray);
        if (bsdf.pdf <= 0.0) { break; }
        throughput *= bsdf.value * max(dot(n, ray), 0.0) / bsdf.pdf;
        if (bounce >= 2u) {
            let survival = clamp(max(throughput.r, max(throughput.g, throughput.b)), 0.05, 0.95);
            if (path_random(&seed) >= survival) { break; }
            throughput /= survival;
        }
        let bias = 2.0 * max(0.00008, travel * 0.00008) * distance_factor(travel) / quality;
        let origin = point + n * bias + ray * bias;
        travel = path_intersection(origin, ray, side, quality, refined);
        if (travel < 0.0) {
            if (travel == -1.0) { radiance += throughput * world_radiance(ray); }
            break;
        }
        if (bounce + 1u < limit) {
            point = origin + ray * travel;
            view = -ray;
            let orientation = select(side, -1.0, side == 0.0);
            n = orientation * normal(point, adaptive_normal_step(travel), PATH_ITERATIONS, orientation * view);
            if (camera.appearance.x > 0.0) {
                material = surface_material(surface_base_color(n, world_to_view));
            }
        }
    }
    return radiance;
}
