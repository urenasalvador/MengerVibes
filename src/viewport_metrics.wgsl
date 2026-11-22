@group(0) @binding(0) var preview: texture_2d<f32>;
@group(0) @binding(1) var<uniform> camera: array<vec4<f32>, 14>;
@group(0) @binding(2) var<storage, read_write> valid: u32;

var<workgroup> sums: array<u32, 256>;

@compute @workgroup_size(16, 16)
fn count_valid(@builtin(local_invocation_id) lane: vec3<u32>, @builtin(local_invocation_index) i: u32) {
    let size = select(vec2<f32>(textureDimensions(preview)), camera[13].yz, camera[13].y > 0.0);
    var sum = 0u;
    // Sample the centers of 32 × 32 strata. Alpha is the fraction of primary
    // samples that resolved a hit or escaped without exhausting their budget.
    for (var x = 0u; x < 2u; x++) {
        for (var y = 0u; y < 2u; y++) {
            let p = (vec2<f32>(lane.xy * 2u + vec2<u32>(x, y)) + 0.5) * size / 32.0;
            sum += u32(round(textureLoad(preview, vec2<i32>(p), 0).a * 4.0));
        }
    }
    sums[i] = sum;
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if (i < stride) { sums[i] += sums[i + stride]; }
        workgroupBarrier();
    }
    if (i == 0u) { valid = sums[0]; }
}
