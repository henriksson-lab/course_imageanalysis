// Rectangle features on the micrograph: response maps, normalising for lighting, a box
// feature stack, and the size of the feature pool. Pictures in out/rect-features/.
//   cargo run -p integral -- rect-features                      (your code)
//   cargo run -p integral --features solution -- rect-features  (the model solution)

use imgcourse::*;
use crate::rect_features::*;

/// The response of `feature` with rectangles of `w × h` and a template of `tw × th` pixels,
/// centred on every pixel where the template fits (0 elsewhere), divided by `area`.
fn response_map(img: &Image<f32>, tw: usize, th: usize, area: f64, feature: impl Fn(&Image<f64>, usize, usize) -> f64) -> Image<f32> {
    let s = integral(img);
    Image::from_fn(img.width(), img.height(), |x, y| {
        if x < tw / 2 || y < th / 2 || x + tw - tw / 2 > img.width() || y + th - th / 2 > img.height() {
            return 0.0;
        }
        (feature(&s, x - tw / 2, y - th / 2) / area) as f32
    })
}

/// Mean and standard deviation of the `w × h` window at `(x, y)` (two tables, previous unit).
fn window_sd(img: &Image<f32>, x: usize, y: usize, w: usize, h: usize) -> f64 {
    let (s1, s2) = (integral(img), integral(&img.map(|v| v * v)));
    let n = (w * h) as f64;
    let m = rect(&s1, x, y, w, h) / n;
    (rect(&s2, x, y, w, h) / n - m * m).max(0.0).sqrt()
}

pub fn main() {
    let dir = "out/rect-features";
    let img = samples::cell();
    save_gray(&img, format!("{dir}/input.png"));

    // Response maps, as differences of mean grey levels.
    let across = response_map(&img, 16, 16, 128.0, |s, x, y| edge_across(s, x, y, 8, 16));
    let down = response_map(&img, 16, 16, 128.0, |s, x, y| edge_down(s, x, y, 16, 8));
    let blob = response_map(&img, 24, 24, 64.0, |s, x, y| centre_surround(s, x, y, 8, 8));
    save_gray_signed(&across, format!("{dir}/edge-across.png"));
    save_gray_signed(&down, format!("{dir}/edge-down.png"));
    save_gray_signed(&blob, format!("{dir}/centre-surround.png"));

    // Lighting: the same edge feature (two 8 × 16 rectangles) on a bright and a dim copy,
    // where the bright copy has its strongest response.
    let (mut bx, mut by) = (0, 0);
    for (x, y) in across.coords() {
        if across[(x, y)].abs() > across[(bx, by)].abs() {
            (bx, by) = (x, y);
        }
    }
    let (x0, y0) = (bx - 8, by - 8);
    let dim = img.map(|v| 0.3 * v + 60.0 / 255.0);
    println!("edge feature (two 8 × 16 rectangles) with its top-left corner at ({x0}, {y0}):");
    for (name, im) in [("bright", &img), ("dim", &dim)] {
        let raw = edge_across(&integral(im), x0, y0, 8, 16) * 255.0; // in grey levels
        let sd = window_sd(im, x0, y0, 16, 16) * 255.0;
        println!("  {name:6}  raw {raw:9.1}   / area {:7.2}   / area / window SD {:6.3}", raw / 128.0, raw / 128.0 / sd);
    }

    // A box feature stack: box mean, box SD and centre − surround at scales 1, 4 and 16 px.
    // The image is padded with its mirror image, so that every template fits.
    let pad = 64;
    let padded = Image::from_fn(256 + 2 * pad, 256 + 2 * pad, |x, y| img.get_border(x as isize - pad as isize, y as isize - pad as isize, Border::Mirror));
    let (s1, s2) = (integral(&padded), integral(&padded.map(|v| v * v)));
    let mut sheet = Image::new(3 * 256 + 2 * 8, 3 * 256 + 2 * 8, 0.5f32);
    for (col, scale) in [1usize, 4, 16].into_iter().enumerate() {
        let k = 2 * scale + 1; // window k × k around the pixel
        let n = (k * k) as f64;
        let mean = Image::from_fn(256, 256, |x, y| (rect(&s1, x + pad - scale, y + pad - scale, k, k) / n) as f32);
        let sd = Image::from_fn(256, 256, |x, y| {
            let (m, q) = (rect(&s1, x + pad - scale, y + pad - scale, k, k) / n, rect(&s2, x + pad - scale, y + pad - scale, k, k) / n);
            (q - m * m).max(0.0).sqrt() as f32
        });
        let cs = Image::from_fn(256, 256, |x, y| (centre_surround(&s1, x + pad - 3 * scale - 1, y + pad - 3 * scale - 1, 2 * scale + 1, 2 * scale + 1) / n) as f32);
        let (sd_max, cs_max) = (sd.max(), cs.pixels().iter().fold(0.0f32, |m, v| m.max(v.abs())));
        for (x, y) in mean.coords() {
            let ox = col * 264;
            sheet[(ox + x, y)] = mean[(x, y)];
            sheet[(ox + x, 264 + y)] = sd[(x, y)] / sd_max;
            sheet[(ox + x, 528 + y)] = 0.5 + 0.5 * cs[(x, y)] / cs_max;
        }
    }
    save_gray(&sheet, format!("{dir}/stack.png"));

    // The pool of features in a 24 × 24 window.
    println!("\nfeatures in a 24 × 24 window (576 pixels):");
    let mut total = 0;
    for (name, a, b) in [("edge across", 2, 1), ("edge down", 1, 2), ("line across", 3, 1), ("line down", 1, 3), ("diagonal", 2, 2), ("centre − surround", 3, 3)] {
        let n = count_placements(a, b, 24, 24);
        total += n;
        println!("  {name:18} {n:7}");
    }
    println!("  {:18} {total:7}", "all six");
    println!("wrote the results to {dir}/");
}
