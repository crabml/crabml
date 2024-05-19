use crate::cpu::buf::buf_f32::exp_f32_cached;
use crate::cpu::buf::CpuTensorBuf;
use crate::cpu::CpuTensorDeviceRef;
use crate::error::ErrorKind;
use crate::error::Result;
use crate::gguf::GGMLType;
use crate::tensor::TensorStrider;

// TODO: support f16
pub fn softmax_inplace<'a>(
    device: CpuTensorDeviceRef<'a>,
    buf: &mut CpuTensorBuf<'a>,
    strider: TensorStrider,
    axis: usize,
) -> Result<()> {
    assert!(strider.dims() == 2 || strider.dims() == 3);
    assert!(strider.is_contiguous());
    assert!(buf.dtype() == GGMLType::F32);

    if axis != strider.dims() - 1 {
        return Err((
            ErrorKind::TensorError,
            format!(
                "only axis={} is supported on a {} dimensions tensor",
                strider.dims() - 1,
                strider.dims()
            ),
        )
            .into());
    }

    let (depths, rows, cols) = match strider.dims() {
        2 => (1, strider.shape()[0], strider.shape()[1]),
        3 => (strider.shape()[0], strider.shape()[1], strider.shape()[2]),
        _ => unreachable!(),
    };
    let (stride_0, stride_1, _) = (rows * cols, cols, 1);

    let buf = buf.as_f32_mut();

    for depth in 0..depths {
        for row in 0..rows {
            let buf_offset = depth * stride_0 + row * stride_1;
            let buf_row = &mut buf[buf_offset..buf_offset + cols];
            let max = buf_row.iter().fold(f32::NEG_INFINITY, |m, val| val.max(m));
            let sum = buf_row.iter_mut().fold(0.0, |mut acc, val| {
                *val = exp_f32_cached(*val - max, &device.exp_cache);
                acc += *val;
                acc
            });
            assert!(sum > 0.0);
            buf_row.iter_mut().for_each(|val| {
                *val /= sum;
            });
        }
    }

    Ok(())
}
