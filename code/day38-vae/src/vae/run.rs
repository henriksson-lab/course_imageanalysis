// The lecture's one-dimensional VAE: the loss of a few clouds, the noise of estimates from
// draws, and the best cloud found by gradient descent for several beta; a map of the loss
// with the descent paths in out/vae/.
//   cargo run -p day38 -- vae                      (your code)
//   cargo run -p day38 --features solution -- vae  (the model solution)

use imgcourse::*;
use crate::vae::*;

pub fn main() {
    let dir = "out/vae";
    let (x, w, sx) = (1.5, 1.0, 0.5);

    println!("x = {x}, w = {w}, sigma_x = {sx}, beta = 1");
    println!("{:<34} {:>8} {:>8} {:>8} {:>16}", "cloud", "R", "KL", "total", "R from 20 draws");
    let mut rng = Rng::new(4);
    let eps: Vec<f32> = (0..20).map(|_| rng.normal()).collect();
    for (name, mu, s) in [("prior", 0.0, 1.0), ("too confident", 1.5, 0.05), ("best", 1.2, 0.2f32.sqrt()), ("B6", 1.0, 0.5)] {
        let r = reconstruction_exact(x, w, sx, mu, s);
        let kl = kl_to_prior(&[mu], &[s]);
        let est = reconstruction_from_draws(x, w, sx, mu, s, &eps);
        println!("{:<34} {r:8.3} {kl:8.3} {:8.3} {est:16.3}", format!("{name} (mu {mu}, s {s:.3})"), r + kl);
    }

    // How noisy is an estimate from 20 draws? Five sets of draws at the prior.
    let estimates: Vec<String> = (0..5)
        .map(|seed| {
            let mut rng = Rng::new(100 + seed);
            let eps: Vec<f32> = (0..20).map(|_| rng.normal()).collect();
            format!("{:.3}", reconstruction_from_draws(x, w, sx, 0.0, 1.0, &eps))
        })
        .collect();
    println!("R at the prior from five sets of 20 draws: {} (exact 6.5)", estimates.join(", "));

    // The best cloud by gradient descent, against the formula of the lecture.
    println!("\n{:>6} {:>8} {:>8} {:>10} {:>10} {:>8} {:>8}", "beta", "mu", "s", "mu formula", "s formula", "R", "KL");
    let mut paths = Vec::new();
    for beta in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0] {
        let path = fit_cloud(x, w, sx, beta, 2000, 10, 0.05, 1);
        let tail = &path[1500..];
        let mu = tail.iter().map(|p| p.0).sum::<f32>() / tail.len() as f32;
        let s = tail.iter().map(|p| p.1).sum::<f32>() / tail.len() as f32;
        let mu_formula = w * x / (w * w + beta * sx * sx);
        let s_formula = (beta * sx * sx / (beta * sx * sx + w * w)).sqrt();
        let r = reconstruction_exact(x, w, sx, mu, s);
        let kl = kl_to_prior(&[mu], &[s]);
        println!("{beta:6} {mu:8.3} {s:8.3} {mu_formula:10.3} {s_formula:10.3} {r:8.3} {kl:8.3}");
        if beta == 1.0 || beta == 5.0 {
            paths.push(path);
        }
    }

    // A map of the total loss (beta = 1) over mu (across, −0.5 to 2) and s (up, 0 to 1.5):
    // dark is low. The descent paths for beta = 1 (red) and beta = 5 (blue) on top.
    let (width, height) = (400, 240);
    let to_mu = |px: usize| -0.5 + 2.5 * px as f32 / width as f32;
    let to_s = |py: usize| 1.5 * (height - py) as f32 / height as f32;
    let mut map = Image::from_fn(width, height, |px, py| {
        let (mu, s) = (to_mu(px), to_s(py).max(1e-3));
        let total = reconstruction_exact(x, w, sx, mu, s) + kl_to_prior(&[mu], &[s]);
        let v = (total.ln() - 1.705f32.ln()) / 2.0; // 0 at the best total, rising with the log
        let grey = v.clamp(0.0, 1.0);
        // contour bands every 0.25 in log
        let band = ((v * 8.0).fract() < 0.08) as u8 as f32 * 0.15;
        [grey - band, grey - band, grey - band].map(|c: f32| c.clamp(0.0, 1.0))
    });
    for (path, colour) in paths.iter().zip([[0.9, 0.15, 0.1], [0.2, 0.5, 1.0]]) {
        for &(mu, s) in path {
            let px = ((mu + 0.5) / 2.5 * width as f32) as isize;
            let py = (height as f32 - s / 1.5 * height as f32) as isize;
            if map.contains(px, py) {
                map[(px as usize, py as usize)] = colour;
            }
        }
    }
    save_rgb(&map, format!("{dir}/loss-map.png"));
    println!("\nwrote {dir}/loss-map.png (loss over mu across and s up; red: descent for beta = 1, blue: beta = 5)");
}
