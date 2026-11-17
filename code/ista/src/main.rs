// Deblurs a signal with five spikes by ISTA, FISTA and OMP; prints how many iterations
// each needs and draws the results in out/ista/.
//   cargo run -p ista                      (your code)
//   cargo run -p ista --features solution  (the model solution)

use imgcourse::*;
use ista::*;

fn main() {
    let dir = "out/ista";
    let n = 128;

    // The truth: five spikes, two of them close together. Blur σ = 2, noise 0.01.
    let mut truth = vec![0.0; n];
    for (i, v) in [(20, 1.0), (52, -0.7), (70, 0.8), (76, 0.6), (105, -0.5)] {
        truth[i] = v;
    }
    let a = blur(n, 2.0);
    let mut rng = Rng::new(3);
    let y: Vec<f64> = mat_vec(&a, &truth).iter().map(|v| v + 0.01 * rng.normal() as f64).collect();

    let l = largest_gain(&a, 200);
    let tau = 1.0 / l;
    let lambda = 0.01;
    println!("largest gain L = {l:.4}, step τ = {tau:.4}, λ = {lambda}");

    // How many iterations to reach the minimum within 1e-6? (The minimum: many FISTA steps.)
    let best = objective(&a, &y, &fista(&a, &y, lambda, tau, 20000), lambda);
    for (name, solver) in [("ISTA", ista as Solver), ("FISTA", fista as Solver)] {
        let mut k = 1;
        while objective(&a, &y, &solver(&a, &y, lambda, tau, k), lambda) - best > 1e-6 && k < 20000 {
            k = if k < 64 { k + 1 } else { k * 21 / 20 }; // count closely at first, then in 5 % steps
        }
        let x = solver(&a, &y, lambda, tau, k);
        println!("{name:5}: about {k} iterations to within 1e-6 of the minimum; {} nonzeros", nonzeros(&x));
    }
    for k in [1, 10, 100, 1000] {
        let x = ista(&a, &y, lambda, tau, k);
        println!("  ISTA after {k:4} iterations: {:3} nonzeros, relative error {:.3}", nonzeros(&x), rel_error(&x, &truth));
    }
    let x_fista = fista(&a, &y, lambda, tau, 1000);
    println!("FISTA, 1000 iterations: relative error {:.3}", rel_error(&x_fista, &truth));

    // OMP with 5 atoms; and with the noise level as the stopping rule.
    let x_omp = omp(&a, &y, 5, 0.0);
    let picked: Vec<usize> = (0..n).filter(|&i| x_omp[i] != 0.0).collect();
    println!("OMP, 5 atoms: positions {picked:?} (truth: 20, 52, 70, 76, 105), relative error {:.3}", rel_error(&x_omp, &truth));

    // Pictures: truth (black stems), data (grey line), and each result (blue dots).
    let grey = [0.6, 0.6, 0.6];
    let blue = [0.1, 0.3, 0.9];
    let black = [0.0, 0.0, 0.0];
    let ista20 = ista(&a, &y, lambda, tau, 20);
    for (name, x) in [("ista-20", &ista20), ("fista-1000", &x_fista), ("omp", &x_omp)] {
        let pic = plot(&[(&y, grey, Style::Line), (&truth, black, Style::Stems), (x, blue, Style::Dots)], 4, 200);
        save_rgb(&pic, format!("{dir}/{name}.png"));
    }
    println!("wrote the results to {dir}/ (black stems: truth, grey: data, blue dots: result)");
}

type Solver = fn(&[Vec<f64>], &[f64], f64, f64, usize) -> Vec<f64>;

fn nonzeros(x: &[f64]) -> usize {
    x.iter().filter(|v| **v != 0.0).count()
}

fn rel_error(x: &[f64], truth: &[f64]) -> f64 {
    let e: f64 = x.iter().zip(truth).map(|(a, b)| (a - b).powi(2)).sum();
    let t: f64 = truth.iter().map(|v| v * v).sum();
    (e / t).sqrt()
}

/// A circular Gaussian blur of standard deviation `sigma` (weights sum to 1), as a matrix.
fn blur(n: usize, sigma: f64) -> Vec<Vec<f64>> {
    let w: Vec<f64> = (0..n).map(|d| (-((d.min(n - d)) as f64).powi(2) / (2.0 * sigma * sigma)).exp()).collect();
    let s: f64 = w.iter().sum();
    (0..n).map(|i| (0..n).map(|j| w[(i + n - j) % n] / s).collect()).collect()
}

/// How `plot` draws a series.
#[derive(Clone, Copy, PartialEq)]
enum Style {
    /// A connected line.
    Line,
    /// A vertical line from 0 to each nonzero value.
    Stems,
    /// A dot (5 × 5 pixels) at each nonzero value.
    Dots,
}

/// Draw signals into a picture, `scale` pixels per sample and `height` pixels high: each
/// series is (values, colour, style), drawn in that order. 0 is the thin grey horizontal line.
fn plot(series: &[(&[f64], [f32; 3], Style)], scale: usize, height: usize) -> Image<[f32; 3]> {
    let n = series[0].0.len();
    let lo = series.iter().flat_map(|s| s.0.iter()).fold(0.0f64, |m, v| m.min(*v));
    let hi = series.iter().flat_map(|s| s.0.iter()).fold(0.0f64, |m, v| m.max(*v));
    let row = |v: f64| ((hi - v) / (hi - lo + 1e-12) * (height - 9) as f64) as usize + 4;
    let mut pic = Image::new(n * scale, height, [1.0f32; 3]);
    for x in 0..n * scale {
        pic[(x, row(0.0))] = [0.85; 3];
    }
    let vline = |pic: &mut Image<[f32; 3]>, x: usize, r0: usize, r1: usize, c: [f32; 3]| {
        for y in r0.min(r1)..=r0.max(r1) {
            pic[(x, y)] = c;
        }
    };
    for &(values, colour, style) in series {
        for i in 0..n {
            let x = i * scale + scale / 2;
            match style {
                Style::Stems if values[i] != 0.0 => vline(&mut pic, x, row(0.0), row(values[i]), colour),
                Style::Dots if values[i] != 0.0 => {
                    for dy in -2..=2isize {
                        for dx in -2..=2isize {
                            let (px, py) = (x as isize + dx, row(values[i]) as isize + dy);
                            if pic.contains(px, py) {
                                pic[(px as usize, py as usize)] = colour;
                            }
                        }
                    }
                }
                Style::Line if i + 1 < n => {
                    for s in 0..scale {
                        let f = |s: usize| values[i] + (values[i + 1] - values[i]) * s as f64 / scale as f64;
                        vline(&mut pic, x + s, row(f(s)), row(f(s + 1)), colour);
                    }
                }
                _ => {}
            }
        }
    }
    pic
}
