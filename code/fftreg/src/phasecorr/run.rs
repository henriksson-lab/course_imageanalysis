// Phase correlation against cross-correlation on two crops of the micrograph: a brightness
// ramp, noise and a low-pass weight, uneven illumination and the taper, and a periodic
// pattern. Surfaces in out/phasecorr/ (shift 0 in the middle).
//   cargo run -p fftreg -- phasecorr                      (your code)
//   cargo run -p fftreg --features solution -- phasecorr  (the model solution)

use imgcourse::*;
use crate::common::centred;
use crate::phasecorr::*;

/// Registers the two crops of `field` (A at (40, 50), B 21 px right and 13 px down) with the
/// settings, prints the result and saves the surface.
fn run(name: &str, field: &Image<f32>, change_b: impl Fn(Image<f32>) -> Image<f32>, s: Settings, noise: f32) {
    let mut a = field.crop(40, 50, 128, 128);
    let mut b = change_b(field.crop(61, 63, 128, 128));
    if noise > 0.0 {
        a = samples::add_noise(&a, noise, 1);
        b = samples::add_noise(&b, noise, 2);
    }
    let r = phase_correlation(&a, &b, s);
    let (tx, ty, h, ratio) = peak_and_ratio(&r);
    let at_truth = r[(21, 13)];
    let extra = format!("{}{}", if s.taper { " taper" } else { "" }, if s.lowpass_sd > 0.0 { " low-pass" } else { "" });
    println!("{name:20} p = {:3.1}{extra:15} peak ({tx:3}, {ty:3})  ratio {ratio:5.2}  height {h:.3}, at (21, 13) {at_truth:.3}", s.p);
    save_gray_signed(&centred(&r), format!("out/phasecorr/{name}.png"));
}

pub fn main() {
    let cell = samples::cell();
    let same = |b: Image<f32>| b;
    let cc = Settings { p: 0.0, taper: true, lowpass_sd: 0.0 };
    let phase = Settings { p: 1.0, ..cc };
    println!("B is 21 px right and 13 px down of A (128 × 128 crops):\n");
    run("cc", &cell, same, cc, 0.0);
    run("phase", &cell, same, phase, 0.0);
    run("phase-no-taper", &cell, same, Settings { taper: false, ..phase }, 0.0);

    // A brightness ramp of 500 grey levels across B.
    let ramp = |b: Image<f32>| b.map_xy(|x, _, v| v + 500.0 / 255.0 * x as f32 / 127.0);
    run("cc-ramp", &cell, ramp, cc, 0.0);
    run("phase-ramp", &cell, ramp, phase, 0.0);

    // Independent noise, SD 120 grey levels, on both crops.
    let sigma = 120.0 / 255.0;
    run("cc-noise", &cell, same, cc, sigma);
    run("phase-noise", &cell, same, phase, sigma);
    run("phase-noise-p0.5", &cell, same, Settings { p: 0.5, ..phase }, sigma);
    run("phase-noise-lowpass", &cell, same, Settings { lowpass_sd: 0.08, ..phase }, sigma);

    // The whole field unevenly lit: 600 grey levels more in the bottom-right corner.
    let uneven = cell.map_xy(|x, y, v| v + 600.0 / 255.0 * (x + y) as f32 / 510.0);
    run("cc-uneven-no-taper", &uneven, same, Settings { taper: false, ..cc }, 0.0);
    run("cc-uneven-taper", &uneven, same, cc, 0.0);

    // A lattice of blobs 18 px apart.
    let lattice = Image::from_fn(256, 256, |x, y| {
        let (dx, dy) = ((x % 18) as f32 - 9.0, (y % 18) as f32 - 9.0);
        (-(dx * dx + dy * dy) / 18.0).exp()
    });
    run("phase-lattice", &lattice, same, Settings { lowpass_sd: 0.1, ..phase }, 0.02);
    println!("wrote the results to out/phasecorr/");
}
