use bencher::benchmark_group;
use bencher::benchmark_main;

use bencher::Bencher;
use rayon::prelude::*;
use crabml::tensor::CpuTensor;
use crabml::tensor::arithmetic::tensor_matmul_2d;

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

fn benchmark_tensor_matmul(bench: &mut Bencher) {
    let w = CpuTensor::new(vec![1.0; 50000], vec![100, 500]).unwrap();
    let b = CpuTensor::new(vec![1.0; 500], vec![500]).unwrap();
    bench.iter(|| {
        tensor_matmul_2d(&w, &b).unwrap();
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

benchmark_group!(benches, benchmark_tensor_matmul, benchmark_simple_matmul, benchmark_par_simple_matmul);
benchmark_main!(benches);