use std::borrow::Cow;

use crate::backends::cpu::buf::buf_f32::exp_f32_cached;
use crate::backends::cpu::buf::CpuTensorBuf;
use crate::backends::cpu::CpuTensorDeviceRef;
use crate::error::ErrorKind;
use crate::error::Result;
use crate::tensor::TensorStrider;

// TODO: support f16
pub fn softmax_inplace<'a>(
    device: CpuTensorDeviceRef<'a>,
    buf: &mut CpuTensorBuf<'a>,
    strider: TensorStrider,
    axis: usize,
) -> Result<()> {
    assert!(strider.shape().len() == 2);
    assert!(strider.is_contiguous());

    if axis != 1 {
        return Err((ErrorKind::TensorError, "only axis=1 is supported").into());
    }

    let rows = strider.shape()[0];
    let cols = strider.shape()[1];
    let buf = match buf {
        CpuTensorBuf::F32(Cow::Owned(buf)) => buf,
        _ => panic!("only support f32 yet"),
    };

    for row in 0..rows {
        let buf_row = &mut buf[row * cols..(row + 1) * cols];
        let max = buf_row.iter().fold(0.0, |m, val| val.max(m));
        let sum = buf_row.iter_mut().fold(0.0, |mut acc, val| {
            *val = exp_f32_cached(*val - max, &device.exp_cache);
            acc += *val;
            acc
        });
        buf_row.iter_mut().for_each(|val| {
            *val /= sum;
        });
    }

    Ok(())
}
