// Applies gamma, log, equalisation and CLAHE to the course micrograph and to a synthetic
// 12-bit fluorescence image; results in out/contrast/.
//   cargo run -p contrast                      (your code)
//   cargo run -p contrast --features solution  (the model solution)

use contrast::*;
use imgcourse::*;

fn levels_used(img: &Image<u32>, levels: usize) -> usize {
    histogram(img, levels).iter().filter(|&&c| c > 0).count()
}

/// Save whole numbers 0 … levels − 1 as a grey picture.
fn save(img: &Image<u32>, levels: usize, path: String) {
    save_gray(&img.map(|v| v as f32 / (levels - 1) as f32), path);
}

/// The standard deviation of the pixels in a box.
fn sd_in(img: &Image<u32>, (x0, y0, w, h): (usize, usize, usize, usize)) -> f32 {
    let v: Vec<f32> = img.crop(x0, y0, w, h).pixels().iter().map(|&p| p as f32).collect();
    let m = v.iter().sum::<f32>() / v.len() as f32;
    (v.iter().map(|p| (p - m) * (p - m)).sum::<f32>() / v.len() as f32).sqrt()
}

fn main() {
    let dir = "out/contrast";

    // The micrograph, 8 bits.
    let cell = samples::cell().map(|v| (255.0 * v).round() as u32);
    let eq = apply_lut(&cell, &equalise_lut(&histogram(&cell, 256)));
    save(&eq, 256, format!("{dir}/cell-equalised.png"));
    save(&apply_lut(&cell, &gamma_lut(256, 256, 0.5)), 256, format!("{dir}/cell-gamma0.5.png"));
    println!("micrograph: {} levels used, {} after equalisation", levels_used(&cell, 256), levels_used(&eq, 256));

    // A synthetic 12-bit fluorescence image: nuclei on a background of about 160, with noise.
    let (f, labels) = samples::nuclei(256, 8, 4);
    let mut rng = Rng::new(9);
    let fluo = Image::from_fn(256, 256, |x, y| {
        let mean = 160.0 + 2500.0 * (f[(x, y)] - 0.2).max(0.0);
        (mean + mean.sqrt() * rng.normal()).round().clamp(0.0, 4095.0) as u32
    });
    // A 24 × 24 box of pure background, to measure the noise in.
    let boxed = (0..20)
        .flat_map(|j| (0..20).map(move |i| (10 * i, 10 * j)))
        .find(|&(x, y)| x + 24 <= 256 && y + 24 <= 256 && labels.crop(x, y, 24, 24).pixels().iter().all(|&l| l == 0))
        .map(|(x, y)| (x, y, 24, 24))
        .unwrap();
    // To 8 bits: the plain window 0 … 4095, the log curve, and global equalisation.
    let linear = apply_lut(&fluo, &(0..4096).map(|v| (v as f32 / 4095.0 * 255.0).round() as u32).collect::<Vec<_>>());
    let logged = apply_lut(&fluo, &log_lut(4096, 256));
    let eq12 = apply_lut(&fluo, &equalise_lut(&histogram(&fluo, 4096))).map(|v| (v as f32 / 4095.0 * 255.0).round() as u32);
    // CLAHE on the 8-bit linear picture, 8 × 8 tiles.
    let clahe3 = clahe(&linear, 256, 8, 3.0);
    let clahe_free = clahe(&linear, 256, 8, 1000.0);
    println!("fluorescence: background box {boxed:?}");
    for (name, img) in [("linear", &linear), ("log", &logged), ("equalised", &eq12), ("clahe-beta3", &clahe3), ("clahe-noclip", &clahe_free)] {
        save(img, 256, format!("{dir}/fluo-{name}.png"));
        println!("  {name:13}: background noise SD {:5.1} display levels, {:3} display levels used", sd_in(img, boxed), levels_used(img, 256));
    }

    // A curve cannot split levels: stretch ×4 on 8-bit data, and the same on the 12-bit data.
    let dim8 = fluo.map(|v| (v as f32 / 16.0).round().clamp(6.0, 70.0) as u32); // 8-bit, values 6 … 70
    let stretch8 = apply_lut(&dim8, &(0..256).map(|v| ((v as f32 - 6.0) * 4.0).clamp(0.0, 255.0) as u32).collect::<Vec<_>>());
    let stretch12 = apply_lut(&fluo, &(0..4096).map(|v| ((v as f32 - 96.0) / 4.0).clamp(0.0, 255.0).round() as u32).collect::<Vec<_>>());
    save(&stretch8, 256, format!("{dir}/stretch-from-8bit.png"));
    save(&stretch12, 256, format!("{dir}/stretch-from-12bit.png"));
    println!("stretch ×4: from 8 bits {} display levels used, from 12 bits {}", levels_used(&stretch8, 256), levels_used(&stretch12, 256));
    println!("wrote the results to {dir}/");
}
