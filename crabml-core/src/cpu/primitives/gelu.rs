use half::f16;

use crate::cpu::buf::CpuTensorBuf;
use crate::cpu::CpuTensorDeviceRef;
use crate::error::Result;

const COEF_A: f32 = 0.044715;
const SQRT_2_OVER_PI: f64 = 0.797_884_560_802_865_4;

pub fn gelu_inplace<'a>(device: CpuTensorDeviceRef<'a>, buf: &mut CpuTensorBuf<'a>) -> Result<()> {
    let cache = device.gelu_cache();
    buf.iter_f32_mut().for_each(|x| {
        *x = cache[f16::from_f32(*x).to_bits() as usize].to_f32();
    });
    Ok(())
}

#[inline]
pub fn gelu_single(x: f32) -> f32 {
    0.5 * x * (1.0 + ((SQRT_2_OVER_PI as f32) * x * (1.0 + COEF_A * x * x)).tanh())
}
