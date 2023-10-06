use bencher::benchmark_group;
use bencher::benchmark_main;

use bencher::Bencher;
use crabml::tensor::arithmetic::matmul;
use crabml::tensor::arithmetic::matmul_specialized_2d_1d;
use crabml::tensor::CpuTensor;
use rayon::prelude::*;

/// test benchmark_iter_simple_matmul                 ... bench:      35,691 ns/iter (+/- 143)
/// test benchmark_par_simple_matmul                  ... bench:      40,225 ns/iter (+/- 2,400)
/// test benchmark_simple_matmul                      ... bench:      35,747 ns/iter (+/- 402)
/// test benchmark_tensor_matmul                      ... bench:      35,687 ns/iter (+/- 1,554)
/// test benchmark_tensor_matmul_specialized_2d_1d    ... bench:      35,694 ns/iter (+/- 138)
/// test benchmark_tensor_matmul_specialized_2d_1d_v2 ... bench:      36,740 ns/iter (+/- 8,254)
///
/// Some note about the benchmark:
/// - chained iter like .iter().enumerate.map(|i| pos[i]) is dog slow, 30 nearly times slower
/// - `iter()` on a memory contigous slice is not slow
/// - Boxed `iter()` is 3 times slower
/// - having memory allocation did not actually make it slower
/// - C-like speicialized matmul is still the fastest

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

fn benchmark_tensor_matmul(bench: &mut Bencher) {
    let w = CpuTensor::new(vec![1.0; 50000], vec![100, 500]).unwrap();
    let b = CpuTensor::new(vec![1.0; 500], vec![500]).unwrap();
    bench.iter(|| {
        matmul(&w, &b).unwrap();
    })
}

fn benchmark_tensor_matmul_specialized_2d_1d(bench: &mut Bencher) {
    let w = CpuTensor::new(vec![1.0; 50000], vec![100, 500]).unwrap();
    let b = CpuTensor::new(vec![1.0; 500], vec![500]).unwrap();
    bench.iter(|| {
        matmul_specialized_2d_1d(&w, &b).unwrap();
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
);
benchmark_main!(benches);
