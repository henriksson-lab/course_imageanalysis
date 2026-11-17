// A fluorescence-like test image (the inverted micrograph: bright structures on a dark
// background), blurred, recorded as photon counts, and restored with your Richardson–Lucy.
// Results in out/richardson-lucy/.
//   cargo run -p richardson-lucy                      (your code)
//   cargo run -p richardson-lucy --features solution  (the model solution)

use imgcourse::*;
use richardson_lucy::*;

/// A Poisson-distributed photon count with the given mean (Knuth's method for small means,
/// the normal approximation for large ones).
fn poisson(rng: &mut Rng, mean: f32) -> f32 {
    if mean > 50.0 {
        return (mean + mean.sqrt() * rng.normal()).round().max(0.0);
    }
    let limit = (-mean).exp();
    let (mut k, mut p) = (0.0, rng.uniform());
    while p > limit {
        k += 1.0;
        p *= rng.uniform();
    }
    k
}

fn gaussian_psf(sigma: f32) -> Image<f32> {
    let r = (3.0 * sigma).ceil() as usize;
    let g = Image::from_fn(2 * r + 1, 2 * r + 1, |x, y| {
        let (dx, dy) = (x as f32 - r as f32, y as f32 - r as f32);
        (-(dx * dx + dy * dy) / (2.0 * sigma * sigma)).exp()
    });
    let total = g.sum();
    g.map(|v| v / total)
}

fn rmse(a: &Image<f32>, b: &Image<f32>) -> f32 {
    let s: f32 = a.pixels().iter().zip(b.pixels()).map(|(p, q)| (p - q).powi(2)).sum();
    (s / a.pixels().len() as f32).sqrt()
}

fn main() {
    let dir = "out/richardson-lucy";
    // The micrograph at half size (2 × 2 pixels averaged), inverted, the part with the nucleus.
    let cell = samples::cell();
    let half = Image::from_fn(128, 128, |x, y| {
        (cell[(2 * x, 2 * y)] + cell[(2 * x + 1, 2 * y)] + cell[(2 * x, 2 * y + 1)] + cell[(2 * x + 1, 2 * y + 1)]) / 4.0
    });
    let size = 72;
    let truth = half.crop(30, 16, size, size).map(|v| 1.0 - v);
    let psf = gaussian_psf(1.5);
    let mut rng = Rng::new(2);
    let photons = 100.0; // photons at brightness 1
    let expected = blur(&truth, &psf).map(|v| photons * v);
    let y = Image::from_fn(size, size, |x, yy| poisson(&mut rng, expected[(x, yy)]));
    println!("inverted micrograph, {size} × {size}, Gaussian PSF σ = 1.5 pixels, {photons} photons at brightness 1");
    println!("  data: {:.0} photons in all; RMSE of the data / {photons} against the truth: {:.4}", y.sum(), rmse(&y.map(|v| v / photons), &truth));
    save_gray(&truth, format!("{dir}/truth.png"));
    save_gray(&y.map(|v| v / photons), format!("{dir}/observed.png"));

    println!("     k     RMSE   deviance per pixel   total of the estimate   smallest pixel");
    let mut x = Image::new(size, size, y.mean());
    let mut done = 0;
    let mut best = (f32::INFINITY, 0);
    for k in [1, 2, 3, 5, 10, 20, 50, 100, 200] {
        for _ in done..k {
            x = rl_step(&x, &y, &psf, 0.0);
        }
        done = k;
        let error = rmse(&x.map(|v| v / photons), &truth);
        if error < best.0 {
            best = (error, k);
        }
        let deviance = poisson_deviance(&y, &blur(&x, &psf));
        println!("  {k:4}   {error:.4}   {deviance:18.3}   {:21.0}   {:14.3}", x.sum(), x.min());
        if [5, 20, 200].contains(&k) {
            save_gray(&x.map(|v| v / photons), format!("{dir}/rl-{k}.png"));
        }
    }
    println!("  smallest error at k = {} (of the values tried)", best.1);

    // The same with a background of 20 photons per pixel (a camera offset not removed),
    // restored without and with the background term.
    let y_bg = y.map(|v| v + 20.0);
    for (name, b) in [("ignored", 0.0), ("modelled", 20.0)] {
        let x = richardson_lucy(&y_bg, &psf, 20, b);
        let error = rmse(&x.map(|v| v / photons), &truth);
        println!("background 20 photons per pixel, {name:8}: RMSE after 20 steps {error:.4}, total {:.0}", x.sum());
        save_gray(&x.map(|v| v / photons), format!("{dir}/rl-20-background-{name}.png"));
    }
    println!("\nwrote the pictures to {dir}/");
}
