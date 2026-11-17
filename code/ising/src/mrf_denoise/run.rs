// Denoises a binary test image with flip noise by ICM, annealing, sampling (MPM) and mean
// field, and counts the wrong pixels. Pictures in out/mrf-denoise/ (object white).
//   cargo run -p ising -- mrf-denoise                      (your code)
//   cargo run -p ising --features solution -- mrf-denoise  (the model solution)

use imgcourse::*;
use crate::mrf_denoise::*;

/// The test image of the lecture, 128 × 128: a disc, a square, a 1-pixel and a 3-pixel line,
/// and a small square.
fn test_shapes() -> Labels {
    Image::from_fn(128, 128, |x, y| {
        let (fx, fy) = (x as f32, y as f32);
        let disc = (fx - 36.0).powi(2) + (fy - 40.0).powi(2) <= 22.0f32.powi(2);
        let square = (72..112).contains(&x) && (16..56).contains(&y);
        let thin = x == 20 && (72..120).contains(&y);
        let thick = (34..37).contains(&x) && (72..120).contains(&y);
        let small = (70..82).contains(&x) && (90..102).contains(&y);
        (disc || square || thin || thick || small) as u32
    })
}

fn wrong(a: &Labels, truth: &Labels) -> usize {
    a.pixels().iter().zip(truth.pixels()).filter(|(p, q)| p != q).count()
}

fn to_gray(l: &Labels) -> Image<f32> {
    l.map(|v| v as f32)
}

/// Errors in colour: red = false object, blue = missed object, grey = correct object.
fn error_picture(l: &Labels, truth: &Labels) -> Image<[f32; 3]> {
    l.zip_map(truth, |a, t| match (a, t) {
        (1, 0) => [0.9, 0.15, 0.1],
        (0, 1) => [0.15, 0.35, 0.95],
        (1, 1) => [0.6, 0.6, 0.6],
        _ => [0.0, 0.0, 0.0],
    })
}

pub fn main() {
    let dir = "out/mrf-denoise";
    let truth = test_shapes();
    let eps = 0.2;
    let mut rng = Rng::new(1);
    let noisy = Image::from_fn(128, 128, |x, y| if rng.uniform() < eps { 1 - truth[(x, y)] } else { truth[(x, y)] });
    save_gray(&to_gray(&truth), format!("{dir}/truth.png"));
    save_gray(&to_gray(&noisy), format!("{dir}/noisy.png"));
    let costs = flip_noise_costs(&noisy, eps);
    let order: Vec<(usize, usize)> = noisy.coords().collect();
    println!("flip noise ε = {eps}: the noisy image has {} wrong pixels", wrong(&noisy, &truth));

    for lambda in [0.8, 1.2, 1.6] {
        println!("\nλ = {lambda}:              energy   wrong pixels");
        let e0 = energy(&noisy, &costs, lambda);
        println!("  noisy image        {e0:9.1}   {:5}", wrong(&noisy, &truth));

        let (icm_result, sweeps) = icm(&noisy, &costs, lambda, &order);
        println!("  ICM ({sweeps:2} sweeps)    {:9.1}   {:5}", energy(&icm_result, &costs, lambda), wrong(&icm_result, &truth));

        // Annealing: Gibbs sweeps while T falls from 4 to 0.05, then ICM.
        for (name, k) in [("quench", 10), ("slow cooling", 600)] {
            let mut l = noisy.clone();
            let mut rng = Rng::new(2);
            let r = (0.05f32 / 4.0).powf(1.0 / (k - 1) as f32);
            let mut t = 4.0;
            for _ in 0..k {
                gibbs_sweep(&mut l, &costs, lambda, t, &mut rng);
                t *= r;
            }
            let (l, _) = icm(&l, &costs, lambda, &order);
            println!("  {name:13} {k:4} {:9.1}   {:5}", energy(&l, &costs, lambda), wrong(&l, &truth));
            if k == 600 && (lambda == 0.8 || lambda == 1.6) {
                save_rgb(&error_picture(&l, &truth), format!("{dir}/errors-anneal-{lambda}.png"));
            }
        }

        // MPM from 300 Gibbs sweeps at T = 1, after 20 sweeps of burn-in.
        let p = marginals(&noisy, &costs, lambda, 20, 300, &mut Rng::new(3));
        let mpm = p.map(|v| (v > 0.5) as u32);
        println!("  MPM                {:9.1}   {:5}", energy(&mpm, &costs, lambda), wrong(&mpm, &truth));

        let q = mean_field(&costs, lambda, 50);
        let mf = q.map(|v| (v > 0.5) as u32);
        println!("  mean field         {:9.1}   {:5}", energy(&mf, &costs, lambda), wrong(&mf, &truth));

        if lambda == 0.8 {
            save_rgb(&error_picture(&icm_result, &truth), format!("{dir}/errors-icm.png"));
        }
        if lambda == 1.2 {
            save_rgb(&error_picture(&mpm, &truth), format!("{dir}/errors-mpm.png"));
            save_gray(&p, format!("{dir}/marginal.png"));
        }
        if lambda == 1.6 {
            save_rgb(&error_picture(&mf, &truth), format!("{dir}/errors-mean-field.png"));
        }
    }
    println!("\nwrote the pictures to {dir}/ (errors: red false object, blue missed, grey correct object)");
}
