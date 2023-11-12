@group(0) @binding(0)
var<storage, read> input: array<f32>;

@group(0) @binding(1)
var<storage, read> eps: f32;

@group(0) @binding(2)
var<storage, read_write> output: array<f32>;

var<workgroup> thread_sums: array<f32, 64>;

// each workgroup normalize a single vector

@compute @workgroup_size(64)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    var thread_sum = 0.0f;
    for (var i = 0u; i < 64u; i = i + 1u) {
        let idx = workgroup_id.x * 64u + i;
        let value = input[idx];
        thread_sum += input[idx] * input[idx];
    }
    thread_sums[local_id.x] += thread_sum;
    workgroupBarrier();

    // reduce squared sum
    if local_id.x == 0u {
        var wg_sum = 0.0f;
        for (var i = 1u; i < 64u; i = i + 1u) {
            thread_sums[0] += thread_sums[i];
        }
    }
    workgroupBarrier();

    output[global_id.x] = input[global_id.x] / sqrt(thread_sums[0] + eps);
}