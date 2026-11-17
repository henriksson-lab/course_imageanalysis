// Co-occurrence features on test textures and on the course micrograph; results printed and
// pictures in out/texture-features/.
//   cargo run -p measurements -- texture-features                      (your code)
//   cargo run -p measurements --features solution -- texture-features  (the model solution)

use imgcourse::*;
use crate::texture_features::*;

/// Gaussian-like smoothing: `passes` binomial (1 2 1)/4 blurs along rows and columns, wrapping
/// around at the border. Each pass adds a variance of 1/2 pixel², so σ ≈ sqrt(passes / 2).
fn smooth(img: &Image<f32>, passes: usize) -> Image<f32> {
    let mut out = img.clone();
    for _ in 0..passes {
        let along_x = out.map_xy(|x, y, v| 0.25 * out.get_border(x as isize - 1, y as isize, Border::Wrap) + 0.5 * v + 0.25 * out.get_border(x as isize + 1, y as isize, Border::Wrap));
        out = along_x.map_xy(|x, y, v| 0.25 * along_x.get_border(x as isize, y as isize - 1, Border::Wrap) + 0.5 * v + 0.25 * along_x.get_border(x as isize, y as isize + 1, Border::Wrap));
    }
    out
}

/// Replace every value by its rank / n: the same picture with a flat histogram on [0, 1).
fn equalise(img: &Image<f32>) -> Image<f32> {
    let mut order: Vec<usize> = (0..img.pixels().len()).collect();
    order.sort_by(|&a, &b| img.pixels()[a].partial_cmp(&img.pixels()[b]).unwrap());
    let mut out = img.clone();
    let n = order.len() as f32;
    for (rank, &i) in order.iter().enumerate() {
        out.pixels_mut()[i] = rank as f32 / n;
    }
    out
}

/// The same pixel values in a random order (Fisher–Yates shuffle).
fn shuffle(img: &Image<f32>, seed: u64) -> Image<f32> {
    let mut rng = Rng::new(seed);
    let mut out = img.clone();
    let v = out.pixels_mut();
    for i in (1..v.len()).rev() {
        v.swap(i, rng.below(i + 1));
    }
    out
}

fn features(img: &Image<f32>, d: isize) -> Haralick {
    haralick(&glcm(&quantise(img, 8, 0.0, 1.0), 8, d, 0, None))
}

/// Enlarge `k` times, so that single pixels can be seen.
fn enlarge(img: &Image<f32>, k: usize) -> Image<f32> {
    Image::from_fn(img.width() * k, img.height() * k, |x, y| img[(x / k, y / k)])
}

pub fn main() {
    let dir = "out/texture-features";
    let white = Image::from_fn(64, 64, {
        let mut rng = Rng::new(5);
        move |_, _| rng.normal()
    });

    // 1. Same histogram, different texture.
    let smooth3 = equalise(&smooth(&white, 18)); // σ ≈ 3
    let shuffled = shuffle(&smooth3, 9);
    println!("same histogram, 8 levels, 0°, d = 1:");
    for (name, img) in [("smoothed noise", &smooth3), ("shuffled", &shuffled)] {
        let h = features(img, 1);
        println!(
            "  {name:<15} contrast {:>6.2}  homogeneity {:.3}  ASM {:.4}  entropy {:.2}  correlation {:>5.2}",
            h.contrast, h.homogeneity, h.asm, h.entropy, h.correlation
        );
        save_gray(&enlarge(img, 4), format!("{dir}/{}.png", name.replace(' ', "-")));
    }

    // 2. Correlation against the distance d: smoothed noise (σ ≈ 4) and white noise.
    let smooth4 = equalise(&smooth(&white, 32));
    let white_eq = equalise(&white);
    println!("\ncorrelation against d (0°):\n   d   smoothed  white");
    for d in [1, 2, 4, 6, 8, 12, 16] {
        println!("  {d:>2}   {:>7.2}  {:>5.2}", features(&smooth4, d).correlation, features(&white_eq, d).correlation);
    }

    // 3. Vertical stripes with period 8: contrast against d at 0° and 90°.
    let stripes = Image::from_fn(64, 64, |x, _| 0.5 + 0.49 * (2.0 * std::f32::consts::PI * (x as f32 + 0.5) / 8.0).cos());
    print!("\nstripes, period 8, contrast at 0° for d = 1..16:");
    for d in 1..=16 {
        print!(" {:.1}", features(&stripes, d).contrast);
    }
    let h90 = haralick(&glcm(&quantise(&stripes, 8, 0.0, 1.0), 8, 0, -4, None));
    println!("\n  at 90°, d = 4: contrast {:.1}", h90.contrast);
    save_gray(&enlarge(&stripes, 4), format!("{dir}/stripes.png"));

    // 4. Texture maps of the micrograph: contrast and entropy in a 17 × 17 window around
    //    every pixel (8 levels on 0..1, 0°, d = 1).
    let cell = samples::cell();
    let q = quantise(&cell, 8, 0.0, 1.0);
    let r = 8;
    let mut contrast = Image::new(cell.width(), cell.height(), 0.0);
    let mut entropy = contrast.clone();
    let mut window = Image::new(2 * r + 1, 2 * r + 1, 0u32);
    for (x, y) in cell.coords() {
        for (wx, wy) in window.coords() {
            let (sx, sy) = (x as isize + wx as isize - r as isize, y as isize + wy as isize - r as isize);
            window[(wx, wy)] = q[(Border::Mirror.index(sx, q.width()).unwrap(), Border::Mirror.index(sy, q.height()).unwrap())];
        }
        let h = haralick(&glcm(&window, 8, 1, 0, None));
        contrast[(x, y)] = h.contrast;
        entropy[(x, y)] = h.entropy;
    }
    let summary = |m: &Image<f32>, x0: usize, y0: usize, w: usize, h: usize| {
        let piece = m.crop(x0, y0, w, h);
        let mean = piece.mean();
        let sd = (piece.pixels().iter().map(|v| (v - mean).powi(2)).sum::<f32>() / piece.pixels().len() as f32).sqrt();
        format!("{mean:.2} ± {sd:.2}")
    };
    println!("\nmicrograph, 17 × 17 windows: contrast and entropy (mean ± sd over the pixels)");
    println!("  whole image:                  contrast {}, entropy {}", summary(&contrast, 0, 0, 256, 256), summary(&entropy, 0, 0, 256, 256));
    println!("  inside the nucleus (40 × 40): contrast {}, entropy {}", summary(&contrast, 125, 80, 40, 40), summary(&entropy, 125, 80, 40, 40));
    save_gray(&cell, format!("{dir}/cell.png"));
    save_gray_stretched(&contrast, format!("{dir}/cell-contrast.png"));
    save_gray_stretched(&entropy, format!("{dir}/cell-entropy.png"));

    // 5. Rotation-invariant LBP codes of the micrograph: the most frequent ones.
    let mut hist = [0usize; 256];
    for y in 1..cell.height() - 1 {
        for x in 1..cell.width() - 1 {
            hist[rotation_invariant(lbp(&cell, x, y)) as usize] += 1;
        }
    }
    let mut codes: Vec<usize> = (0..256).filter(|&c| hist[c] > 0).collect();
    codes.sort_by_key(|&c| std::cmp::Reverse(hist[c]));
    let total: usize = hist.iter().sum();
    print!("\nmicrograph: {} different rotation-invariant LBP codes; most frequent:", codes.len());
    for &c in &codes[..6] {
        print!("  {c:08b} {:.1}%", 100.0 * hist[c] as f32 / total as f32);
    }
    println!("\nwrote the pictures to {dir}/");
}
