fn apply_contrast(color: vec4<f32>, contrast: f32) -> vec4<f32> {
    if (contrast <= 0.0 || contrast == 1.0) { return color; }
    let luma = max(dot(color.rgb, vec3<f32>(0.2126, 0.7152, 0.0722)), 0.00001);
    // Adjust tonal separation around middle gray while limiting highlight gain.
    let ratio = mix(clamp(luma / 0.18, 0.0001, 3.0), 1.0, smoothstep(0.35, 1.0, luma));
    return vec4<f32>(color.rgb * pow(ratio, contrast - 1.0), color.a);
}
