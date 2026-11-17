// Box means, repeated boxes, local statistics, contrast normalisation and the precision of
// the tables, on the micrograph and synthetic images. Pictures in out/box-filters/.
//   cargo run -p integral -- box-filters                      (your code)
//   cargo run -p integral --features solution -- box-filters  (the model solution)

use crate::box_filters::*;
use imgcourse::*;

pub fn main() {
    let dir = "out/box-filters";
    let img = samples::cell();

    // A 51 × 51 box mean: clipped at the border, and with zeros outside (always divide by 51²).
    save_gray(&box_mean(&img, 25), format!("{dir}/box51-clipped.png"));
    let zeros = box_mean(&img, 25).map_xy(|x, y, m| {
        let (x0, x1) = clip(x, 25, 256);
        let (y0, y1) = clip(y, 25, 256);
        m * ((x1 - x0 + 1) * (y1 - y0 + 1)) as f32 / (51.0 * 51.0)
    });
    save_gray(&zeros, format!("{dir}/box51-zeros.png"));

    // Repeated boxes: three bright points, after 1, 2 and 3 passes of a 15 × 15 box.
    let mut points = Image::new(160, 60, 0.0f32);
    for x in [30, 80, 130] {
        points[(x, 30)] = 1.0;
    }
    for n in 1..=3 {
        save_gray_stretched(&repeated_box(&points, 7, n), format!("{dir}/points-{n}box.png"));
    }

    // Local mean and standard deviation, 9 × 9 window.
    let (m, s) = local_mean_sd(&img, 4);
    save_gray(&m, format!("{dir}/local-mean.png"));
    save_gray_stretched(&s, format!("{dir}/local-sd.png"));

    // Contrast normalisation: synthetic nuclei whose contrast fades from right to left.
    let (nuc, _) = samples::nuclei(256, 14, 3);
    let faded = nuc.map_xy(|x, _, v| v * (0.1 + 0.9 * x as f32 / 255.0));
    save_gray(&faded, format!("{dir}/faded.png"));
    for (name, r, eps) in [("17-eps0.02", 8, 0.02), ("17-eps0", 8, 0.0), ("3-eps0.02", 1, 0.02)] {
        let out = normalise(&faded, r, eps);
        save_gray(&out.map(|v| 0.5 + v / 6.0), format!("{dir}/normalised-{name}.png"));
    }

    // Precision: the local SD (7 × 7) from tables in float32 against the exact one, for the
    // micrograph in grey levels plus 200 (bright), without and with an offset subtracted.
    let bright = img.map(|v| 255.0 * v + 200.0);
    let (_, exact) = local_mean_sd(&bright, 3);
    let mean = bright.mean();
    for (name, offset) in [("no offset", 0.0), ("mean subtracted", mean)] {
        let (sd, negative) = local_sd_float32(&bright, 3, offset);
        let err = sd.zip_map(&exact, |a, b| (a - b).abs());
        println!("float32 tables, {name}: largest SD error {:.2} grey levels, {negative} negative variances", err.max());
        save_gray_stretched(&err, format!("{dir}/sd-error-{}.png", if offset == 0.0 { "raw" } else { "offset" }));
    }
    println!("wrote the results to {dir}/");
}

/// The local SD as in `local_mean_sd`, but with the two tables kept in float32 (with about 7
/// significant digits), after subtracting `offset` from every pixel. Also counts the windows
/// whose variance came out negative.
fn local_sd_float32(img: &Image<f32>, r: usize, offset: f32) -> (Image<f32>, usize) {
    let (w, h) = (img.width(), img.height());
    let mut s1 = Image::new(w + 1, h + 1, 0.0f32);
    let mut s2 = Image::new(w + 1, h + 1, 0.0f32);
    for (x, y) in img.coords() {
        let v = img[(x, y)] - offset;
        s1[(x + 1, y + 1)] = v + s1[(x, y + 1)] + s1[(x + 1, y)] - s1[(x, y)];
        s2[(x + 1, y + 1)] = v * v + s2[(x, y + 1)] + s2[(x + 1, y)] - s2[(x, y)];
    }
    let mut negative = 0;
    let mut sd = Image::new(w, h, 0.0f32);
    for (x, y) in img.coords() {
        let (x0, x1) = clip(x, r, w);
        let (y0, y1) = clip(y, r, h);
        let n = ((x1 - x0 + 1) * (y1 - y0 + 1)) as f32;
        let sum = |s: &Image<f32>| s[(x1 + 1, y1 + 1)] - s[(x0, y1 + 1)] - s[(x1 + 1, y0)] + s[(x0, y0)];
        let m = sum(&s1) / n;
        let var = sum(&s2) / n - m * m;
        if var < 0.0 {
            negative += 1;
        }
        sd[(x, y)] = var.max(0.0).sqrt();
    }
    (sd, negative)
}
