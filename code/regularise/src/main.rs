// A blurred, noisy 1D profile (a step and a ramp), restored with your Tikhonov and Landweber
// code for many λ and iteration counts. Prints the error against the truth, the residual and
// the penalty, and draws profiles in out/regularise/.
//   cargo run -p regularise                      (your code)
//   cargo run -p regularise --features solution  (the model solution)

use imgcourse::*;
use regularise::*;

const N: usize = 128;
const SIGMA_BLUR: f64 = 2.0;
const SIGMA_NOISE: f64 = 0.02;

fn rms(a: &[f64], b: &[f64]) -> f64 {
    (a.iter().zip(b).map(|(p, q)| (p - q).powi(2)).sum::<f64>() / a.len() as f64).sqrt()
}

fn norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn mul(a: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    a.iter().map(|row| row.iter().zip(x).map(|(p, q)| p * q).sum()).collect()
}

/// Draw profiles as lines on a white picture, 4 pixels per sample; values from −0.2 to 1.2.
fn plot(curves: &[(&[f64], [f32; 3])]) -> Image<[f32; 3]> {
    let (w, h) = (4 * N, 240);
    let mut img = Image::new(w, h, [1.0f32; 3]);
    let to_y = |v: f64| ((1.2 - v) / 1.4 * (h - 1) as f64).round() as isize;
    for x in 0..w {
        img[(x, to_y(0.0) as usize)] = [0.8; 3]; // the zero line
    }
    for (values, colour) in curves {
        for i in 0..values.len() - 1 {
            // a straight line from sample i to sample i + 1, one dot per pixel row and column
            let (x0, y0, y1) = (4 * i as isize, to_y(values[i]), to_y(values[i + 1]));
            let steps = 4.max((y1 - y0).abs());
            for t in 0..=steps {
                let x = x0 + 4 * t / steps;
                let y = y0 + (y1 - y0) * t / steps;
                if img.contains(x, y) {
                    img[(x as usize, y as usize)] = *colour;
                }
            }
        }
    }
    img
}

fn main() {
    let dir = "out/regularise";
    // The truth: 0.2, a step up to 0.8 and down again, then a ramp from 0.2 to 0.8.
    let truth: Vec<f64> = (0..N)
        .map(|i| match i {
            20..=49 => 0.8,
            70..=109 => 0.2 + 0.6 * (i - 70) as f64 / 39.0,
            _ => 0.2,
        })
        .collect();
    let a = blur_matrix(N, SIGMA_BLUR);
    let mut rng = Rng::new(1);
    let y: Vec<f64> = mul(&a, &truth).iter().map(|v| v + SIGMA_NOISE * rng.normal() as f64).collect();
    println!("{N} samples, Gaussian blur σ = {SIGMA_BLUR}, noise σn = {SIGMA_NOISE}; the data's own error {:.4}", rms(&y, &truth));

    for penalty in [Penalty::Energy, Penalty::Roughness] {
        let d = penalty_matrix(N, penalty);
        println!("\n{penalty:?} penalty:");
        println!("        λ      error   RMS residual   ‖Ax − y‖   ‖Dx‖     mean");
        let mut best = (f64::INFINITY, 0.0);
        for e in -14..=4 {
            let lambda = 10f64.powf(e as f64 / 2.0);
            let x = tikhonov(&a, &y, lambda, penalty);
            let error = rms(&x, &truth);
            if error < best.0 {
                best = (error, lambda);
            }
            let res = rms_residual(&a, &x, &y);
            let mean = x.iter().sum::<f64>() / N as f64;
            println!("  {lambda:9.1e}  {error:8.4}   {res:10.4}   {:8.4}  {:8.4}  {mean:.4}", res * (N as f64).sqrt(), norm(&mul(&d, &x)));
        }
        let ld = discrepancy_lambda(&a, &y, SIGMA_NOISE, penalty);
        println!("  smallest error {:.4} at λ = {:.1e}", best.0, best.1);
        println!("  discrepancy principle: λ = {ld:.2e}, error {:.4}", rms(&tikhonov(&a, &y, ld, penalty), &truth));
        if penalty == Penalty::Roughness {
            let (small, good, large) = (tikhonov(&a, &y, 1e-6, penalty), tikhonov(&a, &y, best.1, penalty), tikhonov(&a, &y, 100.0, penalty));
            let grey = [0.6, 0.6, 0.6];
            save_rgb(&plot(&[(&truth, [0.0; 3]), (&y, grey), (&small, [0.85, 0.1, 0.1])]), format!("{dir}/roughness-small-lambda.png"));
            save_rgb(&plot(&[(&truth, [0.0; 3]), (&y, grey), (&good, [0.85, 0.1, 0.1])]), format!("{dir}/roughness-best-lambda.png"));
            save_rgb(&plot(&[(&truth, [0.0; 3]), (&y, grey), (&large, [0.85, 0.1, 0.1])]), format!("{dir}/roughness-large-lambda.png"));
        }
    }

    // Landweber: the number of iterations instead of λ.
    println!("\nLandweber (step ω = 1, from zero):");
    println!("       k      error   RMS residual");
    let mut first_below = None;
    for k in [1, 3, 10, 32, 100, 316, 1000, 3162] {
        let more = landweber(&a, &y, k, 1.0);
        let res = rms_residual(&a, &more, &y);
        println!("  {k:6}  {:8.4}   {res:10.4}", rms(&more, &truth));
        if res <= SIGMA_NOISE && first_below.is_none() {
            first_below = Some(k);
        }
        if k == 32 {
            save_rgb(&plot(&[(&truth, [0.0; 3]), (&y, [0.6; 3]), (&more, [0.1, 0.3, 0.85])]), format!("{dir}/landweber-32.png"));
        }
        if k == 3162 {
            save_rgb(&plot(&[(&truth, [0.0; 3]), (&y, [0.6; 3]), (&more, [0.1, 0.3, 0.85])]), format!("{dir}/landweber-3162.png"));
        }
    }
    if let Some(k) = first_below {
        println!("  the residual first reaches σn at k = {k} (of the values tried)");
    }
    println!("\nwrote the pictures to {dir}/ (black: truth, grey: data, colour: estimate)");
}
