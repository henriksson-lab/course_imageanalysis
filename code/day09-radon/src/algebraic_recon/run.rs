// ART and SIRT on the Shepp–Logan phantom: SIRT's first step, semi-convergence with noisy
// data, and few views with and without non-negativity. Pictures go to out/algebraic-recon/.
//   cargo run -p day09 -- algebraic-recon                      (your code)
//   cargo run -p day09 --features solution -- algebraic-recon  (the model solution)

use crate::algebraic_recon::tomo::*;
use crate::algebraic_recon::*;
use imgcourse::*;

fn rmse(x: &[f32], truth: &[f32]) -> f32 {
    (x.iter().zip(truth).map(|(p, q)| (p - q).powi(2)).sum::<f32>() / x.len() as f32).sqrt()
}

/// ‖A x − b‖ / ‖b‖
fn residual(a: &[Row], x: &[f32], b: &[f32]) -> f32 {
    let ax = forward(a, x);
    let r: f32 = ax.iter().zip(b).map(|(p, q)| (p - q).powi(2)).sum();
    (r / b.iter().map(|q| q * q).sum::<f32>()).sqrt()
}

/// For display: values 0 to 0.5 as black to white (the phantom's soft tissue is 0.2).
fn picture(x: &[f32], n: usize) -> Image<f32> {
    Image::from_vec(n, n, x.iter().map(|v| v / 0.5).collect())
}

pub fn main() {
    let dir = "out/algebraic-recon";
    let n = 96;
    let truth = shepp_logan(n);
    save_gray(&picture(truth.pixels(), n), format!("{dir}/phantom.png"));
    let t = truth.pixels();

    // 1. 60 angles, exact data. SIRT: step 1 (a scaled back-projection), then 100 steps.
    let thetas = angles(60);
    let a = ray_matrix(n, &thetas);
    let b = forward(&a, t);
    let mut x = vec![0.0; n * n];
    sirt_step(&a, &b, &mut x, 1.0);
    println!("SIRT, step 1: RMSE {:.3}", rmse(&x, t));
    save_gray(&picture(&x, n), format!("{dir}/sirt-step1.png"));
    for _ in 1..100 {
        sirt_step(&a, &b, &mut x, 1.0);
    }
    println!("SIRT, 100 steps: RMSE {:.3}, residual {:.4}", rmse(&x, t), residual(&a, &x, &b));
    save_gray(&picture(&x, n), format!("{dir}/sirt-100.png"));

    // 2. Noisy data: ART with ω = 0.5. The error first falls, then rises; the residual falls.
    let mut rng = Rng::new(11);
    let noisy: Vec<f32> = b.iter().map(|v| v + 0.5 * rng.normal()).collect();
    let noise: f32 = noisy.iter().zip(&b).map(|(p, q)| (p - q).powi(2)).sum();
    println!("noise in the data: ‖noise‖ / ‖b‖ = {:.4}", (noise / noisy.iter().map(|q| q * q).sum::<f32>()).sqrt());
    let mut x = vec![0.0; n * n];
    let mut best = (f32::INFINITY, 0);
    println!("ART on noisy data, ω = 0.5:");
    for sweep in 1..=30 {
        art_sweep(&a, &noisy, &mut x, 0.5);
        let e = rmse(&x, t);
        if e < best.0 {
            best = (e, sweep);
            save_gray(&picture(&x, n), format!("{dir}/art-noisy-best.png"));
        }
        if [1, 2, 3, 5, 10, 20, 30].contains(&sweep) {
            println!("  sweep {sweep:>2}: RMSE {e:.4}, residual {:.4}", residual(&a, &x, &noisy));
        }
    }
    println!("  lowest RMSE {:.4} after sweep {}", best.0, best.1);
    save_gray(&picture(&x, n), format!("{dir}/art-noisy-30.png"));

    // 3. Few views (20 angles), exact data: ART, without and with non-negativity.
    let few = angles(20);
    let a = ray_matrix(n, &few);
    let b = forward(&a, t);
    for (name, clip) in [("plain", false), ("non-negative", true)] {
        let mut x = vec![0.0; n * n];
        for _ in 0..20 {
            art_sweep(&a, &b, &mut x, 1.0);
            if clip {
                x.iter_mut().for_each(|v| *v = v.max(0.0));
            }
        }
        println!("20 views, ART 20 sweeps, {name}: RMSE {:.3}", rmse(&x, t));
        save_gray(&picture(&x, n), format!("{dir}/few-views-{name}.png"));
    }
    println!("wrote pictures to {dir}/");
}
