// Keeping the K largest coefficients in the Haar basis and in the Fourier basis, for a spike,
// a step and a row of the micrograph; pictures in out/haar/.
//   cargo run -p wavelets -- haar                      (your code)
//   cargo run -p wavelets --features solution -- haar  (the model solution)
use crate::haar::*;
use imgcourse::{fft, samples, Complex};
use crate::common::{Plot, BLUE, GREY, RED};

/// The coefficients of `x` in the real orthonormal Fourier basis (cosines and sines), from the
/// DFT: `N` numbers, as in the Fourier unit.
fn fourier_coefficients(x: &[f32]) -> Vec<f32> {
    let n = x.len();
    let big = fft::fft(&x.iter().map(|&v| Complex::new(v, 0.0)).collect::<Vec<_>>(), false);
    let (a, b) = ((1.0 / n as f32).sqrt(), (2.0 / n as f32).sqrt());
    let mut c = vec![a * big[0].re, a * big[n / 2].re];
    for k in 1..n / 2 {
        c.push(b * big[k].re);
        c.push(b * big[k].im);
    }
    c
}

/// The signal from its real Fourier coefficients (the inverse of `fourier_coefficients`).
fn fourier_signal(c: &[f32]) -> Vec<f32> {
    let n = c.len();
    let (a, b) = ((n as f32).sqrt(), (n as f32 / 2.0).sqrt());
    let mut big = vec![Complex::new(0.0, 0.0); n];
    big[0] = Complex::new(a * c[0], 0.0);
    big[n / 2] = Complex::new(a * c[1], 0.0);
    for k in 1..n / 2 {
        big[k] = Complex::new(b * c[2 * k], b * c[2 * k + 1]);
        big[n - k] = big[k].conj();
    }
    fft::fft(&big, true).iter().map(|v| v.re).collect()
}

/// Relative error `‖x − x̂‖ / ‖x‖` when keeping the `k` largest of the coefficients `c`
/// (orthonormal: from the dropped energy alone).
fn relative_error(c: &[f32], k: usize) -> f32 {
    let total: f32 = c.iter().map(|v| v * v).sum();
    (keep_largest(c, k).1 / total).sqrt()
}

/// The smallest `k` with a relative error of at most 1 %.
fn k_for_one_percent(c: &[f32]) -> usize {
    (1..=c.len()).find(|&k| relative_error(c, k) <= 0.01).unwrap_or(c.len())
}

pub fn main() {
    let dir = "out/haar";

    // A spike and a step at position 23 of 64 samples.
    let spike: Vec<f32> = (0..64).map(|i| if i == 23 { 1.0 } else { 0.0 }).collect();
    let step: Vec<f32> = (0..64).map(|i| if i >= 23 { 1.0 } else { 0.0 }).collect();
    let row: Vec<f32> = samples::cell().row(160).to_vec();
    println!("signal              nonzero Haar   K = 6: Fourier   Haar    K for 1 %: Fourier  Haar");
    for (name, x) in [("spike at 23 (64)", &spike), ("step at 23 (64)", &step), ("micrograph row 160", &row)] {
        let h = haar_forward(x, true);
        let f = fourier_coefficients(x);
        let nonzero = h.iter().filter(|v| v.abs() > 1e-6).count();
        println!(
            "{name:20} {nonzero:6}        {:6.1} %  {:6.1} %       {:6}  {:6}",
            100.0 * relative_error(&f, 6),
            100.0 * relative_error(&h, 6),
            k_for_one_percent(&f),
            k_for_one_percent(&h)
        );
    }

    // The micrograph row rebuilt from its 16 largest coefficients in each basis.
    let k = 16;
    let haar_hat = haar_inverse(&keep_largest(&haar_forward(&row, true), k).0, true);
    let fourier_hat = fourier_signal(&keep_largest(&fourier_coefficients(&row), k).0);
    println!("micrograph row, K = {k}: relative error Fourier {:.1} %, Haar {:.1} %", 100.0 * relative_error(&fourier_coefficients(&row), k), 100.0 * relative_error(&haar_forward(&row, true), k));
    let mut p = Plot::new(900, 360, (-2.0, 258.0), (-0.05, 1.05));
    let pts = |v: &[f32]| -> Vec<[f32; 2]> { v.iter().enumerate().map(|(i, &y)| [i as f32, y]).collect() };
    p.curve(&pts(&row), GREY);
    p.curve(&pts(&fourier_hat), RED);
    p.curve(&pts(&haar_hat), BLUE);
    p.save(format!("{dir}/row-keep16.png"));
    println!("wrote row-keep16.png to {dir}/");
}
