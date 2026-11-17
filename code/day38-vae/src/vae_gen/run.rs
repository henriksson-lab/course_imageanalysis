// Generating, walking, denoising and anomaly detection with the spot decoder; pictures in
// out/vae-gen/ (rows of small images, enlarged 4 times).
//   cargo run -p day38 -- vae-gen                      (your code)
//   cargo run -p day38 --features solution -- vae-gen  (the model solution)

use imgcourse::*;
use crate::vae_gen::*;

const SIZE: usize = 24;

/// Rows of images side by side (grey gaps), every pixel enlarged 4 times.
fn sheet(rows: &[Vec<Image<f32>>]) -> Image<f32> {
    let (cols, gap, s) = (rows.iter().map(|r| r.len()).max().unwrap(), 2, 4);
    let (w, h) = (cols * (SIZE + gap) - gap, rows.len() * (SIZE + gap) - gap);
    let mut out = Image::new(w * s, h * s, 0.5f32);
    for (r, row) in rows.iter().enumerate() {
        for (c, img) in row.iter().enumerate() {
            for (x, y) in img.coords() {
                let (ox, oy) = ((c * (SIZE + gap) + x) * s, (r * (SIZE + gap) + y) * s);
                for d in 0..s * s {
                    out[(ox + d % s, oy + d / s)] = img[(x, y)].clamp(0.0, 1.0);
                }
            }
        }
    }
    out
}

/// A test image from a formula, with noise of sd 0.05 (the "camera").
fn noisy(f: impl Fn(f32, f32) -> f32, seed: u64) -> Image<f32> {
    let clean = Image::from_fn(SIZE, SIZE, |x, y| f(x as f32, y as f32));
    samples::add_noise(&clean, 0.05, seed)
}

pub fn main() {
    let dir = "out/vae-gen";
    let decode = spot_decoder(SIZE);
    let (lo, hi, n) = (-3.0, 3.0, 41); // the search grid, steps of 0.15

    // Generate: decode draws from the prior.
    let mut rng = Rng::new(3);
    let generated: Vec<Image<f32>> = (0..8).map(|_| decode([rng.normal(), rng.normal()])).collect();
    save_gray(&sheet(&[generated]), format!("{dir}/generated.png"));
    for r in [1.0, 2.0, 3.0] {
        println!("prior: fraction of draws within radius {r}: {:.3}", fraction_within(r, 100_000, 1));
    }

    // Walk in latent space against the pixel fade.
    let (za, zb) = ([-2.0, 1.0], [2.0, -1.0]);
    let walk = latent_walk(&decode, za, zb, 7);
    let fade = pixel_fade(&decode(za), &decode(zb), 7);
    println!("walk vs fade at the midpoint: brightest pixel {:.2} (walk) and {:.2} (fade)", walk[3].max(), fade[3].max());
    save_gray(&sheet(&[walk, fade]), format!("{dir}/walk-vs-fade.png"));

    // Anomaly detection: errors of 300 normal images (a spot anywhere the prior puts it,
    // plus noise), and a threshold at 99 %.
    let mut rng = Rng::new(11);
    let normal_errors: Vec<f32> = (0..300)
        .map(|k| {
            let z = [rng.normal().clamp(-2.9, 2.9), rng.normal().clamp(-2.9, 2.9)];
            let truth = decode(z);
            let img = samples::add_noise(&truth, 0.05, 1000 + k);
            encode_by_search(&img, &decode, lo, hi, n).1
        })
        .collect();
    let t95 = threshold(&normal_errors, 0.95);
    let t99 = threshold(&normal_errors, 0.99);
    let flagged = normal_errors.iter().filter(|&&e| e > t99).count();
    println!("normal images: errors {:.4} to {:.4}; 95 % below {t95:.4}, 99 % below {t99:.4}; {flagged} of 300 above", normal_errors.iter().cloned().fold(f32::MAX, f32::min), normal_errors.iter().cloned().fold(0.0, f32::max));

    let c = SIZE as f32 / 2.0;
    let spot = |x: f32, y: f32, cx: f32, cy: f32, sd: f32| (-((x - cx).powi(2) + (y - cy).powi(2)) / (2.0 * sd * sd)).exp();
    let tests: Vec<(&str, Image<f32>)> = vec![
        ("normal spot", noisy(|x, y| spot(x, y, c + 3.0, c - 2.0, 2.0), 1)),
        ("two spots", noisy(|x, y| spot(x, y, c - 5.0, c, 2.0) + spot(x, y, c + 5.0, c, 2.0), 2)),
        ("ring", noisy(|x, y| (-((((x - c).powi(2) + (y - c).powi(2)).sqrt() - 6.0).powi(2)) / 2.0).exp(), 3)),
        ("wider spot", noisy(|x, y| spot(x, y, c, c, 2.1), 4)),
        ("large spot", noisy(|x, y| spot(x, y, c, c, 3.0), 6)),
        ("faint spot", noisy(|x, y| 0.4 * spot(x, y, c, c, 2.0), 5)),
    ];
    let mut inputs = Vec::new();
    let mut rebuilt = Vec::new();
    for (name, img) in &tests {
        let (z, err) = encode_by_search(img, &decode, lo, hi, n);
        let verdict = if err > t99 { "UNUSUAL" } else { "normal" };
        println!("  {name:<12} code ({:5.2}, {:5.2})  error {err:.4}  {verdict}", z[0], z[1]);
        inputs.push(img.clone());
        rebuilt.push(decode(z));
    }
    save_gray(&sheet(&[inputs, rebuilt]), format!("{dir}/anomalies.png"));

    // Denoising: encode a very noisy spot, decode its code.
    let clean = decode([-1.0, 0.5]);
    let noisy_img = samples::add_noise(&clean, 0.3, 7);
    let (z, _) = encode_by_search(&noisy_img, &decode, lo, hi, n);
    let denoised = decode(z);
    println!(
        "denoising: noisy input error {:.4}, decoded error {:.4} (against the clean image); code found ({:.2}, {:.2}), true (-1, 0.5)",
        mean_squared_error(&noisy_img, &clean),
        mean_squared_error(&denoised, &clean),
        z[0],
        z[1]
    );
    save_gray(&sheet(&[vec![clean, noisy_img, denoised]]), format!("{dir}/denoise.png"));
    println!("wrote the results to {dir}/");
}
