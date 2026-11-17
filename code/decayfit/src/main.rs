// Fits simulated decays with your code: a Monte Carlo of the three costs at 100 to 10 000
// photons, and global analysis of a synthetic FLIM image of two species; pictures in out/decayfit/.
//   cargo run -p decayfit                      (your code)
//   cargo run -p decayfit --features solution  (the model solution)

use decayfit::*;
use imgcourse::*;

const PERIOD: f32 = 12.5;
const BINS: usize = 64;

/// A Gaussian IRF of width `sigma` at `delay` (ns), as a histogram summing to 1.
fn gaussian_irf(delay: f32, sigma: f32) -> Vec<f32> {
    let dt = PERIOD / BINS as f32;
    let g: Vec<f32> = (0..BINS)
        .map(|i| {
            let d = (i as f32 + 0.5) * dt - delay;
            if d.abs() > 4.0 * sigma + dt { 0.0 } else { (-d * d / (2.0 * sigma * sigma)).exp() }
        })
        .collect();
    let s: f32 = g.iter().sum();
    g.iter().map(|v| v / s).collect()
}

/// A Poisson random count with mean `mu` (a normal approximation for large means).
fn poisson(mu: f32, rng: &mut Rng) -> f32 {
    if mu > 50.0 {
        return (mu + mu.sqrt() * rng.normal()).round().max(0.0);
    }
    let (limit, mut k, mut p) = ((-mu).exp(), 0.0, 1.0);
    loop {
        p *= rng.uniform();
        if p < limit {
            return k;
        }
        k += 1.0;
    }
}

fn mean_sd(v: &[f32]) -> (f32, f32) {
    let m = v.iter().sum::<f32>() / v.len() as f32;
    (m, (v.iter().map(|t| (t - m) * (t - m)).sum::<f32>() / (v.len() - 1) as f32).sqrt())
}

fn main() {
    let dir = "out/decayfit";
    std::fs::create_dir_all(dir).unwrap();
    let irf = gaussian_irf(1.0, 0.15);
    let mut rng = Rng::new(1);

    // 1. The photon budget: 300 noisy histograms of a 2.5 ns decay per photon count.
    let tau = 2.5;
    println!("τ = 2.5 ns, 64 bins, IRF σ = 0.15 ns at 1 ns, 0.2 background counts per bin, 300 histograms each:");
    println!("fitted τ: mean ± sd (relative RMS error)");
    println!("photons   ideal τ/√N      least squares            χ² with 1/y              Poisson");
    for n in [100.0, 1000.0, 10_000.0] {
        let mu = model(tau, n, 0.2, &irf, PERIOD); // plus 0.2 background counts per bin
        let mut est = [vec![], vec![], vec![]];
        for _ in 0..300 {
            let y: Vec<f32> = mu.iter().map(|&m| poisson(m, &mut rng)).collect();
            for (k, kind) in [Cost::LeastSquares, Cost::Neyman, Cost::Poisson].into_iter().enumerate() {
                est[k].push(fit(&y, &irf, PERIOD, kind, 2.0).tau);
            }
        }
        print!("{n:7.0}   ± {:.3}      ", tau / n.sqrt());
        for e in &est {
            let (m, sd) = mean_sd(e);
            let rms = (e.iter().map(|t| (t - tau) * (t - tau)).sum::<f32>() / e.len() as f32).sqrt() / tau;
            print!("  {m:.2} ± {sd:.2} ({:4.1} %)   ", 100.0 * rms);
        }
        println!();
    }

    // 2. Global analysis of an image: species of 0.6 and 3.5 ns; the photon fraction of the long
    //    one rises from left to right. About 300 photons in a bright pixel, background 0.2 per bin.
    let size = 96;
    let (img, _) = samples::nuclei(size, 10, 3);
    let (tau1, tau2) = (0.6, 3.5);
    let (s1, s2) = (model(tau1, 1.0, 0.0, &irf, PERIOD), model(tau2, 1.0, 0.0, &irf, PERIOD));
    let true_f = Image::from_fn(size, size, |x, _| 0.2 + 0.6 * x as f32 / size as f32);
    let mut data = vec![vec![0.0f32; BINS]; size * size];
    for (x, y) in img.coords() {
        let n = (400.0 * (img[(x, y)] - 0.1)).max(0.0);
        let f = true_f[(x, y)];
        for j in 0..BINS {
            data[y * size + x][j] = poisson(n * ((1.0 - f) * s1[j] + f * s2[j]) + 0.2, &mut rng);
        }
    }
    let photons = Image::from_fn(size, size, |x, y| data[y * size + x].iter().sum::<f32>());
    save_gray_stretched(&photons, format!("{dir}/intensity.png"));
    let used: Vec<usize> = (0..size * size).filter(|&i| photons.pixels()[i] >= 80.0).collect();

    // The lifetimes once, from the summed decay of all pixels with at least 80 photons: for every
    // pair (τ1, τ2) on a grid, the amplitudes are linear (C5); keep the pair of lowest deviance.
    let mut sum = vec![0.0f32; BINS];
    for &i in &used {
        for j in 0..BINS {
            sum[j] += data[i][j];
        }
    }
    let bg_total = 0.2 * used.len() as f32;
    let flat = vec![1.0f32; BINS];
    let mut best = (f32::INFINITY, 0.0, 0.0);
    for a in 0..60 {
        for b in 0..60 {
            let (t1, t2) = (0.2 + 0.02 * a as f32, 2.0 + 0.05 * b as f32);
            let shapes = vec![model(t1, 1.0, 0.0, &irf, PERIOD), model(t2, 1.0, 0.0, &irf, PERIOD), flat.clone()];
            let amp = amplitudes(&sum, &shapes);
            let mu: Vec<f32> = (0..BINS).map(|j| (amp[0] * shapes[0][j] + amp[1] * shapes[1][j] + amp[2]).max(1e-3)).collect();
            let d = cost(&sum, &mu, Cost::Poisson);
            if d < best.0 {
                best = (d, t1, t2);
            }
        }
    }
    let (_, g1, g2) = best;
    println!("\nglobal analysis: {} pixels with ≥ 80 photons, {:.0} photons in all (background about {bg_total:.0} per bin)", used.len(), sum.iter().sum::<f32>());
    println!("global lifetimes: {g1:.2} and {g2:.2} ns (true 0.60 and 3.50)");

    // Then every pixel: its amplitudes by linear least squares, with the background known.
    let shapes = vec![model(g1, 1.0, 0.0, &irf, PERIOD), model(g2, 1.0, 0.0, &irf, PERIOD)];
    let mut fraction = Image::new(size, size, 0.0f32);
    let mut err = vec![];
    for &i in &used {
        let y: Vec<f32> = data[i].iter().map(|v| v - 0.2).collect();
        let a = amplitudes(&y, &shapes);
        let (a1, a2) = (a[0].max(0.0), a[1].max(0.0));
        let f = if a1 + a2 > 0.0 { a2 / (a1 + a2) } else { 0.0 };
        fraction.pixels_mut()[i] = f;
        err.push(f - true_f.pixels()[i]);
    }
    let (m, sd) = mean_sd(&err);
    println!("photon fraction of the long species per pixel: error {m:+.3} ± {sd:.3}");
    save_gray(&fraction, format!("{dir}/fraction-long.png"));
    save_gray(&Image::from_fn(size, size, |x, y| if photons[(x, y)] >= 80.0 { true_f[(x, y)] } else { 0.0 }), format!("{dir}/fraction-true.png"));
    println!("wrote the pictures to {dir}/");
}
