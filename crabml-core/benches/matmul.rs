use bencher::benchmark_group;
use bencher::benchmark_main;

use bencher::Bencher;
use crabml::tensor::arithmetic::tensor_matmul_2d;
use crabml::tensor::arithmetic::tensor_matmul_specialized_2d_1d;
use crabml::tensor::CpuTensor;
use rayon::prelude::*;

fn simple_matmul(xout: &mut [f32], x: &[f32], w: &[f32]) {
    // w (d,n) @ x (n,) -> xout (d,)
    let x_dim = x.len();
    xout.iter_mut().enumerate().for_each(|(i, xo)| {
        *xo = 0.0;
        for j in 0..x.len() {
            *xo += w[i * x_dim + j] * x[j];
        }
    });
}

fn parallel_simple_matmul(xout: &mut [f32], x: &[f32], w: &[f32]) {
    // w (d,n) @ x (n,) -> xout (d,)
    let x_dim = x.len();
    xout.par_iter_mut().enumerate().for_each(|(i, xo)| {
        *xo = 0.0;
        for j in 0..x.len() {
            *xo += w[i * x_dim + j] * x[j];
        }
    });
}

fn iter_simple_matmul(xout: &mut [f32], x: &[f32], w: &[f32]) {
    // w (d,n) @ x (n,) -> xout (d,)
    let x_dim = x.len();
    xout.iter_mut().enumerate().for_each(|(i, xo)| {
        *xo = 0.0;
        let w_iter = w[i * x_dim..(i + 1) * x_dim].iter();
        let x_iter = x.iter();
        *xo = w_iter.zip(x_iter).map(|(w, x)| w * x).sum::<f32>();
    });
}

pub fn tensor_matmul_specialized_2d_1d_v2<'a>(
    w: &CpuTensor<'a>,
    x: &CpuTensor<'a>,
) -> CpuTensor<'a> {
    let mut xout = CpuTensor::zeros(vec![w.shape()[0]]).unwrap();
    xout.iter_mut().unwrap().enumerate().for_each(|(w_row, xo)| {
        let w_row_iter = w.iter_axis_contigous(&[w_row, 0], 1).unwrap(); // (w_cols, )
        let x_col_iter = x.iter_axis_contigous(&[0], 0).unwrap(); // (w_cols, )
        *xo = w_row_iter.zip(x_col_iter).map(|(w, x)| w * x).sum::<f32>();
    });
    xout
}

fn benchmark_tensor_matmul(bench: &mut Bencher) {
    let w = CpuTensor::new(vec![1.0; 50000], vec![100, 500]).unwrap();
    let b = CpuTensor::new(vec![1.0; 500], vec![500]).unwrap();
    bench.iter(|| {
        tensor_matmul_2d(&w, &b).unwrap();
    })
}

fn benchmark_tensor_matmul_specialized_2d_1d(bench: &mut Bencher) {
    let w = CpuTensor::new(vec![1.0; 50000], vec![100, 500]).unwrap();
    let b = CpuTensor::new(vec![1.0; 500], vec![500]).unwrap();
    bench.iter(|| {
        tensor_matmul_specialized_2d_1d(&w, &b).unwrap();
    })
}

fn benchmark_tensor_matmul_specialized_2d_1d_v2(bench: &mut Bencher) {
    let w = CpuTensor::new(vec![1.0; 50000], vec![100, 500]).unwrap();
    let b = CpuTensor::new(vec![1.0; 500], vec![500]).unwrap();
    bench.iter(|| {
        tensor_matmul_specialized_2d_1d_v2(&w, &b);
    })
}

fn benchmark_simple_matmul(bench: &mut Bencher) {
    let w = vec![1.0; 50000];
    let b = vec![1.0; 500];
    let mut xout = vec![0.0; 100];
    bench.iter(|| {
        simple_matmul(&mut xout, &b, &w);
    })
}

fn benchmark_par_simple_matmul(bench: &mut Bencher) {
    let w = vec![1.0; 50000];
    let b = vec![1.0; 500];
    let mut xout = vec![0.0; 100];
    bench.iter(|| {
        parallel_simple_matmul(&mut xout, &b, &w);
    })
}

fn benchmark_iter_simple_matmul(bench: &mut Bencher) {
    let w = vec![1.0; 50000];
    let b = vec![1.0; 500];
    let mut xout = vec![0.0; 100];
    bench.iter(|| {
        iter_simple_matmul(&mut xout, &b, &w);
    })
}

benchmark_group!(
    benches,
    benchmark_tensor_matmul,
    benchmark_simple_matmul,
    benchmark_par_simple_matmul,
    benchmark_iter_simple_matmul,
    benchmark_tensor_matmul_specialized_2d_1d,
    benchmark_tensor_matmul_specialized_2d_1d_v2,
);
benchmark_main!(benches);
