use crate::backends::cpu::buf::buf_f32::exp_f32_cached;
use crate::backends::cpu::buf::CpuTensorBuf;
use crate::backends::cpu::CpuTensorDeviceRef;
use crate::error::Result;

pub fn silu_inplace<'a>(device: CpuTensorDeviceRef<'a>, buf: &mut CpuTensorBuf<'a>) -> Result<()> {
    let exp_cache = &device.exp_cache;
    buf.iter_f32_mut().for_each(|n| {
        let nexp = exp_f32_cached(-*n, exp_cache);
        *n = *n / (1.0 + nexp)
    });
    Ok(())
}
