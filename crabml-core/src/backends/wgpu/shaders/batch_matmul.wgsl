// (b, m, k) * (b, k, n) = (b, m, n)
struct Meta {
    B: u32,
    M: u32,
    K: u32,
    N: u32,
    strides_b: vec3<u32>,
};

@group(0) @binding(0)
var<storage, read> bufa: array<f32>;

@group(0) @binding(1)
var<storage, read> bufb: array<f32>;

@group(0) @binding(2)
var<storage, read> bufm: Meta;

@group(0) @binding(3)
var<storage, read_write> output: array<f32>;

@compute @workgroup_size(32)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let gidx = workgroup_id.x * 32u + local_id.x;
    let ni = gidx % bufm.N;
    let mi = ((gidx - ni) / bufm.N) % bufm.M;
    let bi = (gidx - ni - mi * bufm.N) / (bufm.M * bufm.N);

    var sum = 0.0f;
    for (var ki = 0u; ki < bufm.K; ki = ki + 1u) {
        let a = bufa[
            bi * bufm.M * bufm.K +
            mi * bufm.K +
            ki
        ];
        let b = bufb[bufm.strides_b.x * bi + ki * bufm.strides_b.y + ni * bufm.strides_b.z];
        sum += a * b;
    }

    output[bi * bufm.M * bufm.N + mi * bufm.N + ni] = sum;
}