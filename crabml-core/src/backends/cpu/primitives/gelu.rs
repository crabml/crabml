use super::scalar::unary;
use crate::backends::cpu::buf::CpuTensorBuf;
use crate::backends::cpu::CpuTensorDeviceRef;
use crate::error::Result;

const COEF_A: f32 = 0.044715;
const SQRT_2_OVER_PI: f64 = 0.797_884_560_802_865_4;

pub fn gelu_inplace<'a>(_device: CpuTensorDeviceRef<'a>, buf: &mut CpuTensorBuf<'a>) -> Result<()> {
    unary::<_>(buf, gelu_single)
}

#[inline]
fn gelu_single(x: f32) -> f32 {
    0.5 * x * (1.0 + ((SQRT_2_OVER_PI as f32) * x * (1.0 + COEF_A * x * x)).tanh())
}
