@group(0) @binding(0) var accumulated: texture_2d<f32>;
@group(0) @binding(1) var<uniform> camera: array<vec4<f32>, 21>;

@vertex fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    var p = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
    return vec4<f32>(p[i], 0.0, 1.0);
}

@fragment fn fs_main(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let preview_size = vec2<f32>(camera[13].w, camera[20].w);
    var color: vec3<f32>;
    if (preview_size.x > 0.0) {
        let size = vec2<f32>(textureDimensions(accumulated));
        let uv = clamp(position.xy * preview_size / size - 0.5, vec2<f32>(0.0), preview_size - 1.0);
        let low = vec2<i32>(floor(uv));
        let high = min(low + 1, vec2<i32>(preview_size) - 1);
        let fraction = fract(uv);
        let top = mix(textureLoad(accumulated, low, 0).rgb,
                      textureLoad(accumulated, vec2<i32>(high.x, low.y), 0).rgb, fraction.x);
        let bottom = mix(textureLoad(accumulated, vec2<i32>(low.x, high.y), 0).rgb,
                         textureLoad(accumulated, high, 0).rgb, fraction.x);
        color = mix(top, bottom, fraction.y);
    } else {
        color = textureLoad(accumulated, vec2<i32>(position.xy), 0).rgb;
    }
    return apply_contrast(vec4<f32>(color, 1.0), camera[12].w);
}
