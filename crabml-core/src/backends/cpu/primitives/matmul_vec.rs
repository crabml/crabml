use std::sync::LazyLock;
use std::sync::Mutex;

use rayon::prelude::*;

use crate::backends::cpu::buf::CpuTensorBuf;
use crate::backends::cpu::thread_pool::ThreadPool;
use crate::backends::cpu::CpuTensorDeviceRef;
use crate::tensor::TensorStrider;

static POOL: LazyLock<Mutex<ThreadPool>> = LazyLock::new(|| Mutex::new(ThreadPool::new(2)));

/// only dense GEMV is supported
/// (m, k) @ k -> (m, )
/// (m, k) @ (b, k) -> (b, m)
pub fn matmul_vec<'a>(
    device: &CpuTensorDeviceRef<'a>,
    bufa: &CpuTensorBuf<'a>,
    bufb: &CpuTensorBuf<'a>,
    bufc: &mut CpuTensorBuf<'a>,
    strider1: &TensorStrider,
    strider2: &TensorStrider,
) {
    assert!(strider1.is_contiguous());
    assert!(strider2.is_contiguous());
    assert!(strider1.shape().last() == strider2.shape().last());

    let (m, k) = (strider1.shape()[0], strider1.shape()[1]);
    gemv_dense_2d_2d(device, bufa, bufb, bufc, m, k);
}

#[allow(clippy::too_many_arguments)]
fn gemv_dense_2d_2d(
    device: &CpuTensorDeviceRef,
    bufa: &CpuTensorBuf,
    bufb: &CpuTensorBuf,
    bufc: &mut CpuTensorBuf,
    m: usize,
    k: usize,
) {
    let bufc = bufc.as_f32_mut();
    let bufb = &bufb.quantize(bufa.vec_dot_rhs_dtype()).unwrap();
    let split_size = bufc.len() / 2;
    let chunk_size = 16;
    assert!(split_size % chunk_size == 0);

    POOL.lock().unwrap().scoped(|s| {
        bufc.chunks_exact_mut(split_size)
            .enumerate()
            .for_each(|(cn, cp)| {
                s.spawn(move || {
                    cp.chunks_exact_mut(chunk_size)
                        .enumerate()
                        .for_each(|(i, cpp)| {
                            let mi = (cn * split_size + i * chunk_size) % m;
                            let bi = ((cn * split_size + i * chunk_size) - mi) / m;
                            for (j, cval) in cpp.iter_mut().enumerate() {
                                *cval = bufa.vec_dot((mi + j) * k, bufb, bi * k, k);
                            }
                        });
                });
            });
    });
}
