use crate::backends::cpu::buf::buf_f32::exp_f32_cached;
use crate::backends::cpu::buf::CpuTensorBuf;
use crate::backends::cpu::CpuTensorDeviceRef;
use crate::error::Result;

// TODO: support f16
pub fn silu_inplace<'a>(device: CpuTensorDeviceRef<'a>, buf: &mut CpuTensorBuf<'a>) -> Result<()> {
    let exp_cache = &device.exp_cache;
    buf.iter_f32_mut().for_each(|n| {
        // let nexp = exp_f32_cached(-*n, exp_cache);
        // *n = *n / (1.0 + nexp)
        *n = gelu(*n);
    });
    Ok(())
}

fn gelu(x: f32) -> f32 {
    let a: f32 = 0.5;
    let b: f32 = 0.044715;
    let sqrt_2_over_pi: f32 = (2.0 / std::f32::consts::PI).sqrt();

    a * x * (1.0 + ((sqrt_2_over_pi * (x + b * x * x * x)).tanh()))
}
