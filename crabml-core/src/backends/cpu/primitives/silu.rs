use crate::backends::cpu::buf::CpuTensorBuf;
use crate::error::Result;

// TODO: support f16
pub fn silu_inplace<'a>(buf: &mut CpuTensorBuf<'a>) -> Result<()> {
    buf.iter_mut().for_each(|n| *n = *n / (1.0 + (-*n).exp()));
    Ok(())
}
