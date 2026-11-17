// Masked image modelling in miniature: hide 6 × 6 patches of a test image and fill them by
// local interpolation, or with a linear model (PCA) of 18 × 18 windows learnt without labels
// from other images of the same kind. Synthetic nuclei (model from one image, test crop from
// another) and the micrograph (model from its upper half, test crop from its lower half).
// Pictures in out/pretext/.
//   cargo run -p selfsup -- pretext                      (your code)
//   cargo run -p selfsup --features solution -- pretext  (the model solution)

use crate::common::{enlarge, side_by_side};
use crate::pretext::*;
use imgcourse::{samples, save_gray, Image, Rng};

const P: usize = 6;
const LAMBDA: f64 = 100.0 / (255.0 * 255.0); // residual variance: 10 grey levels squared

/// Hide a fraction of the inner P × P patches (or of the inner pixels): 1 = hidden.
fn hide(size: usize, patches: bool, ratio: f32, seed: u64) -> Image<f32> {
    let mut rng = Rng::new(seed);
    let g = size / P;
    let mut mask = Image::new(size, size, 0.0);
    if patches {
        for gy in 1..g - 1 {
            for gx in 1..g - 1 {
                if rng.uniform() < ratio {
                    for y in 0..P {
                        for x in 0..P {
                            mask[(gx * P + x, gy * P + y)] = 1.0;
                        }
                    }
                }
            }
        }
    } else {
        for y in P..size - P {
            for x in P..size - P {
                if rng.uniform() < ratio {
                    mask[(x, y)] = 1.0;
                }
            }
        }
    }
    mask
}

/// Fill every inner patch that has hidden pixels from the completed 3P × 3P window around it.
fn fill_model(img: &Image<f32>, mask: &Image<f32>, m: &WindowModel) -> Image<f32> {
    let mut out = img.clone();
    let g = img.width() / P;
    for gy in 1..g - 1 {
        for gx in 1..g - 1 {
            let (x0, y0) = ((gx - 1) * P, (gy - 1) * P);
            let mut window = Vec::new();
            let mut visible = Vec::new();
            for y in 0..3 * P {
                for x in 0..3 * P {
                    window.push(img[(x0 + x, y0 + y)] as f64);
                    visible.push(mask[(x0 + x, y0 + y)] < 0.5);
                }
            }
            if (P..2 * P).all(|y| (P..2 * P).all(|x| visible[y * 3 * P + x])) {
                continue;
            }
            let full = complete_window(m, &window, &visible, LAMBDA);
            for y in P..2 * P {
                for x in P..2 * P {
                    if !visible[y * 3 * P + x] {
                        out[(x0 + x, y0 + y)] = full[y * 3 * P + x] as f32;
                    }
                }
            }
        }
    }
    out
}

pub fn main() {
    let dir = "out/pretext";
    let (train_nuclei, _) = samples::nuclei(256, 40, 1);
    let (test_nuclei, _) = samples::nuclei(256, 40, 2);
    let cell = samples::cell();
    let scenes = [
        ("nuclei", windows(&train_nuclei, 3 * P, 6), test_nuclei.crop(60, 60, 120, 120)),
        ("micrograph", windows(&cell.crop(0, 0, 256, 128), 3 * P, 6), cell.crop(68, 132, 120, 120)),
    ];
    for (name, wins, test) in &scenes {
        let model = learn_model(wins, 3 * P, 32);
        let total: f64 = model.variances.iter().sum();
        println!("{name}: {} training windows of 18 x 18; the first 5 variances {:?} (sum of 32: {total:.2})", wins.len(),
            model.variances[..5].iter().map(|v| (v * 1000.0).round() / 1000.0).collect::<Vec<_>>());
        println!("  {:<24} {:>14} {:>9} {:>9} {:>9}", "hidden", "interpolation", "k = 0", "k = 8", "k = 32");
        for (label, patches, ratio) in [("25 % of the patches", true, 0.25), ("75 % of the patches", true, 0.75), ("90 % of the patches", true, 0.9), ("75 % of the pixels", false, 0.75)] {
            let mask = hide(120, patches, ratio, 7);
            let interp = fill_interp(test, &mask, 3.0);
            let mut row = format!("  {label:<24} {:>11.2} dB", hidden_psnr(&interp, test, &mask));
            let mut fills = Vec::new();
            for k in [0, 8, 32] {
                let m = WindowModel { comps: model.comps[..k].to_vec(), variances: model.variances[..k].to_vec(), ..model.clone() };
                let f = fill_model(test, &mask, &m);
                row += &format!(" {:>6.2} dB", hidden_psnr(&f, test, &mask));
                fills.push(f);
            }
            println!("{row}");
            if patches && ratio == 0.75 {
                let seen = test.zip_map(&mask, |v, m| if m > 0.5 { 1.0 } else { v });
                save_gray(&enlarge(&side_by_side(&[test, &seen, &interp, &fills[2]]), 2), format!("{dir}/{name}-image-seen-interp-pca.png"));
            }
        }
        // the first 8 components as pictures (grey = 0)
        let comps: Vec<Image<f32>> = model.comps[..8]
            .iter()
            .map(|c| {
                let mx = c.iter().fold(0.0f64, |a, v| a.max(v.abs()));
                Image::from_fn(3 * P, 3 * P, |x, y| (0.5 + 0.5 * c[y * 3 * P + x] / mx) as f32)
            })
            .collect();
        let refs: Vec<&Image<f32>> = comps.iter().collect();
        save_gray(&enlarge(&side_by_side(&refs), 6), format!("{dir}/{name}-components.png"));
    }
    println!("(PSNR in dB over the hidden pixels only; k = number of principal components)");
    println!("wrote the pictures to {dir}/");
}
