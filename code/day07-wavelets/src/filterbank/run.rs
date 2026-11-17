// Vanishing moments on a smooth signal with jumps, sparsity with four wavelets, and aliasing
// when one branch is dropped; pictures in out/filterbank/.
//   cargo run -p day07 -- filterbank                      (your code)
//   cargo run -p day07 --features solution -- filterbank  (the model solution)
use crate::filterbank::*;
use crate::common::{Plot, BLACK, BLUE, GREEN, GREY, ORANGE, RED};
use std::f32::consts::PI;

/// Test signals of `n` samples, t = i/n in [0, 1).
fn heavisine(n: usize) -> Vec<f32> {
    let sign = |v: f32| if v > 0.0 { 1.0 } else if v < 0.0 { -1.0 } else { 0.0 };
    (0..n).map(|i| i as f32 / n as f32).map(|t| 4.0 * (4.0 * PI * t).sin() - sign(t - 0.3) - sign(0.72 - t)).collect()
}

fn doppler(n: usize) -> Vec<f32> {
    (0..n).map(|i| i as f32 / n as f32).map(|t| (t * (1.0 - t)).sqrt() * (2.0 * PI * 1.05 / (t + 0.05)).sin()).collect()
}

fn step(n: usize) -> Vec<f32> {
    (0..n).map(|i| if i * 10 >= n * 4 { 1.0 } else { 0.0 }).collect()
}

/// The smallest number of coefficients (the largest ones) that leaves at most 1 % relative error.
fn k_for_one_percent(c: &[f32]) -> usize {
    let mut sq: Vec<f32> = c.iter().map(|v| v * v).collect();
    sq.sort_by(|a, b| b.total_cmp(a));
    let total: f32 = sq.iter().sum();
    let mut kept = 0.0;
    for (k, v) in sq.iter().enumerate() {
        if (total - kept) / total <= 1e-4 {
            return k;
        }
        kept += v;
    }
    c.len()
}

pub fn main() {
    let dir = "out/filterbank";
    let wavelets: [(&str, &[f32]); 4] = [("Haar", &HAAR), ("db2", &DB2), ("db3", &DB3), ("db4", &DB4)];

    // Level-1 details of HeaviSine (256 samples): on the smooth parts and at the jumps.
    let n = 256;
    let x = heavisine(n);
    println!("HeaviSine, 256 samples: largest level-1 detail");
    println!("  wavelet   moments   smooth parts   at the jumps");
    let mut p = Plot::new(800, 440, (-2.0, 130.0), (-4.6, 1.8));
    let colours = [RED, ORANGE, GREEN, BLUE];
    for (w, (name, h)) in wavelets.iter().enumerate() {
        let (_, d) = analysis(&x, h);
        let (mut smooth, mut jump) = (0.0f32, 0.0f32);
        for k in 0..d.len() {
            // Does the window x[2k .. 2k + L − 1] contain a jump (or wrap around the end)?
            let window: Vec<usize> = (0..h.len()).map(|i| 2 * k + i).collect();
            let wraps = window[window.len() - 1] >= n;
            let has_jump = window.windows(2).any(|p| p[1] < n && (x[p[1]] - x[p[0]]).abs() > 1.0);
            if wraps || has_jump {
                jump = jump.max(d[k].abs());
            } else {
                smooth = smooth.max(d[k].abs());
            }
        }
        println!("  {name:8}  {:5}     {smooth:10.6}     {jump:8.3}", vanishing_moments(&highpass(h)));
        let offset = -1.2 * w as f32;
        p.curve(&d.iter().enumerate().map(|(k, &v)| [k as f32, offset + v]).collect::<Vec<_>>(), colours[w]);
    }
    p.save(format!("{dir}/details.png"));

    // Sparsity: coefficients needed for 1 % error, 5 levels, periodic.
    println!("\ncoefficients for 1 % error (256 samples, 5 levels)");
    println!("  signal       Haar   db2   db3   db4");
    for (name, sig) in [("HeaviSine", heavisine(n)), ("step", step(n)), ("Doppler", doppler(n))] {
        let mut line = format!("  {name:10}");
        for (_, h) in wavelets {
            line += &format!("  {:4}", k_for_one_percent(&dwt(&sig, h, 5)));
        }
        println!("{line}");
    }

    // Aliasing: a chirp of 64 samples through db2, with and without the high branch.
    let chirp: Vec<f32> = (0..64).map(|i| i as f32 / 64.0).map(|t| (2.0 * PI * (4.0 * t + 12.0 * t * t)).sin()).collect();
    let (a, d) = analysis(&chirp, &DB2);
    let low = synthesis(&a, &vec![0.0; d.len()], &DB2);
    let both = synthesis(&a, &d, &DB2);
    let err = |y: &[f32]| y.iter().zip(&chirp).map(|(p, q)| (p - q).abs()).fold(0.0f32, f32::max);
    println!("\nchirp through db2: largest error with both branches {:.1e}, low branch alone {:.2}", err(&both), err(&low));
    let mut p = Plot::new(800, 300, (-1.0, 64.0), (-1.6, 1.6));
    let pts = |v: &[f32]| -> Vec<[f32; 2]> { v.iter().enumerate().map(|(i, &y)| [i as f32, y]).collect() };
    p.curve(&pts(&chirp), GREY);
    p.curve(&pts(&low), ORANGE);
    for (i, &v) in both.iter().enumerate() {
        p.dot([i as f32, v], 2.5, BLACK);
    }
    p.save(format!("{dir}/aliasing.png"));

    // The cost of the cascade: samples filtered for a 5-level DWT of 256 samples.
    let filtered: usize = (0..5).map(|j| n >> j).sum();
    println!("5-level cascade on {n} samples: {filtered} samples filtered (2N = {})", 2 * n);
    println!("wrote details.png and aliasing.png to {dir}/");
}
