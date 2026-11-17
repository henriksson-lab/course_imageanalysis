// Runs your threshold methods on a two-class test image with 10 % objects and on the
// micrograph; results in out/otsu/.
//   cargo run -p day12 -- otsu                      (your code)
//   cargo run -p day12 --features solution -- otsu  (the model solution)

use imgcourse::*;
use crate::otsu::*;

/// The histogram with 256 bins: an 8-bit grey level k (stored as k / 255) goes into bin k.
fn histogram256(img: &Image<f32>) -> Vec<usize> {
    let mut h = vec![0; 256];
    for &v in img.pixels() {
        h[(v * 255.0).round().clamp(0.0, 255.0) as usize] += 1;
    }
    h
}

/// Object (white) where the grey level is above `t`.
fn mask(img: &Image<f32>, t: usize) -> Image<f32> {
    img.map(|v| if (v * 255.0).round() > t as f32 { 1.0 } else { 0.0 })
}

pub fn main() {
    let dir = "out/otsu";

    // Two classes as in B4: background 50, objects 150, both with noise 20; small discs,
    // about 10 % of the pixels, are object.
    let truth = Image::from_fn(256, 256, |x, y| {
        let (dx, dy) = ((x % 32) as f32 - 15.5, (y % 32) as f32 - 15.5);
        dx * dx + dy * dy <= 32.0
    });
    let n = truth.pixels().len() as f32;
    let objects = truth.pixels().iter().filter(|&&t| t).count() as f32;
    let mut rng = Rng::new(3);
    let img = Image::from_fn(256, 256, |x, y| {
        let mean = if truth[(x, y)] { 150.0 } else { 50.0 };
        (mean + 20.0 * rng.normal()).round().clamp(0.0, 255.0) / 255.0
    });
    let h = histogram256(&img);
    println!("two classes, {:.1} % objects (the Bayes threshold of B4 is 108.8):", 100.0 * objects / n);
    let errors = |t: usize| {
        let wrong = img.zip_map(&truth, |v, tr| ((v * 255.0).round() > t as f32) != tr).pixels().iter().filter(|&&w| w).count();
        100.0 * wrong as f32 / n
    };
    let t_otsu = otsu(&h);
    let t_iso = isodata(&h, 128.0);
    let t_min = min_error(&h);
    println!("  Otsu        t = {t_otsu:3}, {:.2} % of the pixels wrong", errors(t_otsu));
    println!("  isodata     t = {t_iso:5.1}, {:.2} % wrong", errors(t_iso.floor() as usize));
    println!("  min. error  t = {t_min:3}, {:.2} % wrong", errors(t_min));
    println!("  Bayes       t = 108, {:.2} % wrong", errors(108));
    save_gray(&img, format!("{dir}/two-classes.png"));
    save_gray(&mask(&img, t_otsu), format!("{dir}/two-classes-otsu.png"));
    save_gray(&mask(&img, t_min), format!("{dir}/two-classes-min-error.png"));

    // The micrograph: two and three classes.
    let cell = samples::cell();
    let hc = histogram256(&cell);
    let t = otsu(&hc);
    let (t1, t2) = otsu3(&hc);
    println!("micrograph: Otsu t = {t}, three classes t1 = {t1}, t2 = {t2}, min. error t = {}", min_error(&hc));
    save_gray(&mask(&cell, t), format!("{dir}/cell-otsu.png"));
    // three classes as black, grey and white
    let classes = cell.map(|v| {
        let g = (v * 255.0).round() as usize;
        if g <= t1 {
            0.0
        } else if g <= t2 {
            0.5
        } else {
            1.0
        }
    });
    save_gray(&classes, format!("{dir}/cell-otsu3.png"));
    println!("wrote the results to {dir}/");
}
