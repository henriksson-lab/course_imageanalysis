// Runs your Perona–Malik diffusion on a noisy test card and on a few profiles;
// results in out/perona-malik/.
//   cargo run -p day10 -- perona-malik                      (your code)
//   cargo run -p day10 --features solution -- perona-malik  (the model solution)

use imgcourse::*;
use crate::perona_malik::*;

/// The conductivity map g2(|gradient|) (central differences), white where heat flows.
fn conductivity(img: &Image<f32>, k: f32) -> Image<f32> {
    img.map_xy(|x, y, _| {
        let (xi, yi, b) = (x as isize, y as isize, Border::Mirror);
        let gx = 0.5 * (img.get_border(xi + 1, yi, b) - img.get_border(xi - 1, yi, b));
        let gy = 0.5 * (img.get_border(xi, yi + 1, b) - img.get_border(xi, yi - 1, b));
        g2((gx * gx + gy * gy).sqrt(), k)
    })
}

/// A test card with known truth: a dark background (40), a bright square (200), discs of
/// radius 2 to 8, a 1-pixel line, a ramp and fine stripes (grey values / 255).
fn test_card(size: usize) -> Image<f32> {
    Image::from_fn(size, size, |x, y| {
        let (xf, yf) = (x as f32, y as f32);
        let mut v = 40.0;
        if (24..88).contains(&x) && (24..88).contains(&y) {
            v = 200.0;
        }
        for (i, r) in [2.0, 3.0, 5.0, 8.0].iter().enumerate() {
            let (cx, cy) = (130.0 + 30.0 * i as f32, 40.0);
            if (xf - cx).powi(2) + (yf - cy).powi(2) <= r * r {
                v = 160.0;
            }
        }
        if (110..200).contains(&y) && (110..230).contains(&x) && x % 4 < 2 {
            v = 120.0; // stripes of period 4
        }
        if (210..240).contains(&y) {
            v = if x < 40 { 40.0 } else if x < 120 { 40.0 + 2.0 * (x - 40) as f32 } else { 200.0 };
            if x == 180 {
                v = 40.0; // a 1-pixel line
            }
        }
        v / 255.0
    })
}

fn row_image(v: &[f32]) -> Image<f32> {
    Image::from_vec(v.len(), 1, v.to_vec())
}

pub fn main() {
    let dir = "out/perona-malik";
    let truth = test_card(256);
    save_gray(&truth, format!("{dir}/truth.png"));
    let noisy = samples::add_noise(&truth, 12.0 / 255.0, 1);
    save_gray(&noisy, format!("{dir}/noisy.png"));
    let k = 10.0 / 255.0;
    save_gray(&conductivity(&noisy, k), format!("{dir}/conductivity-raw.png"));
    save_gray(&conductivity(&gaussian_blur(&noisy, 1.0), k), format!("{dir}/conductivity-smoothed.png"));

    // Heat against Perona–Malik (g2, K = 10 grey levels, pre-smoothing 1), dt = 0.2.
    println!("test card, PSNR against time (noise 12 grey levels, noisy input {:.1} dB):", psnr(&noisy, &truth, 1.0));
    println!("     t   heat   Perona–Malik");
    let (mut heat, mut pm) = (noisy.clone(), noisy.clone());
    let (mut best, mut best_t) = (0.0, 0.0);
    for step in 1..=30 {
        heat = pm_step(&heat, 0.2, 1e9, g2, 0.0); // K huge: g = 1, the heat equation
        pm = pm_step(&pm, 0.2, k, g2, 1.0);
        let t = 0.2 * step as f32;
        let p = psnr(&pm, &truth, 1.0);
        if p > best {
            (best, best_t) = (p, t);
        }
        if [1, 2, 3, 5, 10, 20, 30].contains(&step) {
            println!("  {t:4.1}  {:5.1}  {p:5.1}", psnr(&heat, &truth, 1.0));
        }
        if step == 2 || step == 30 {
            save_gray(&heat, format!("{dir}/heat-t{t}.png"));
            save_gray(&pm, format!("{dir}/pm-t{t}.png"));
        }
    }
    println!("  best Perona–Malik: {best:.1} dB at t = {best_t:.1}");

    // Profiles, grey values 0-255 as in the lecture; dt = 0.25 (the 1D limit is 0.5).
    let show = |v: &Image<f32>| v.pixels().iter().map(|x| format!("{x:.0}")).collect::<Vec<_>>().join(" ");
    let steepest = |v: &Image<f32>| v.row(0).windows(2).map(|w| (w[1] - w[0]).abs()).fold(0.0, f32::max);

    // A soft edge from 60 to 180, steepest difference 20, g2 with K = 8.
    let soft = row_image(&(0..40).map(|i| 120.0 + 60.0 * ((i as f32 - 19.5) / 3.0).tanh()).collect::<Vec<_>>());
    let (mut a, mut h) = (soft.clone(), soft.clone());
    for _ in 0..150 {
        a = pm_step(&a, 0.25, 8.0, g2, 0.0);
        h = pm_step(&h, 0.25, 1e9, g2, 0.0);
    }
    println!("soft edge, steepest difference {:.1}; after 150 steps: Perona–Malik {:.1}, heat {:.1}", steepest(&soft), steepest(&a), steepest(&h));

    // A noisy ramp of slope 4, g2 with K = 3: a staircase.
    let mut rng = Rng::new(3);
    let ramp = row_image(&(0..40).map(|i| 40.0 + 4.0 * i as f32 + rng.normal()).collect::<Vec<_>>());
    let mut r = ramp.clone();
    for _ in 0..600 {
        r = pm_step(&r, 0.25, 3.0, g2, 0.0);
    }
    println!("noisy ramp, slope 4:\n  before:    {}\n  600 steps: {}", show(&ramp), show(&r));

    // A spike of 40 on a flat 60, g1 with K = 10, without and with pre-smoothing.
    let mut flat = vec![60.0; 21];
    flat[10] = 100.0;
    let spike = row_image(&flat);
    let (mut s0, mut s1) = (spike.clone(), spike.clone());
    for _ in 0..150 {
        s0 = pm_step(&s0, 0.25, 10.0, g1, 0.0);
        s1 = pm_step(&s1, 0.25, 10.0, g1, 1.5);
    }
    println!("spike after 150 steps, g1, K = 10: classical {:.1}, pre-smoothed (sigma 1.5) {:.1}", s0[(10, 0)], s1[(10, 0)]);
    println!("wrote the pictures to {dir}/");
}
