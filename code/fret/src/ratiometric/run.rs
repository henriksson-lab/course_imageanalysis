// Ratio imaging of a simulated biosensor: background, masks, the display, calibration and the
// noise of a ratio; pictures in out/ratiometric/.
//   cargo run -p fret -- ratiometric                      (your code)
//   cargo run -p fret --features solution -- ratiometric  (the model solution)

use imgcourse::*;
use crate::ratiometric::sim;
use crate::ratiometric::*;

pub fn main() {
    let dir = "out/ratiometric";
    let bg = 40.0;
    let f = sim::field(150.0, bg, 0.0, 4);
    let mask_of = |l: u32| f.labels.map(|v| v == l);

    // 1. Ratio images: background subtracted with a mask, and the mistakes
    let ratio = ratio_image(&f.acceptor, &f.donor, bg, bg, 30.0);
    let no_mask = ratio_image(&f.acceptor, &f.donor, bg, bg, f32::NEG_INFINITY);
    let span = |r: &Image<f32>| r.map(|v| if v.is_nan() { 0.0 } else { ((v - sim::R_MIN) / (sim::R_MAX - sim::R_MIN)).clamp(0.0, 1.0) });
    save_gray_stretched(&f.donor, format!("{dir}/donor.png"));
    save_gray(&span(&ratio), format!("{dir}/ratio.png"));
    save_gray(&span(&no_mask), format!("{dir}/ratio-no-mask.png"));
    let intensity = f.donor.zip_map(&f.acceptor, |d, a| d + a - 2.0 * bg);
    save_rgb(&intensity_modulated(&no_mask, &intensity, sim::R_MIN, sim::R_MAX, 400.0), format!("{dir}/ratio-imd.png"));
    let outside = f.labels.map(|l| l == 0);
    let bg_ratios: Vec<f32> = no_mask.coords().filter(|&p| outside[p]).map(|p| no_mask[p]).collect();
    let negative = bg_ratios.iter().filter(|&&r| r < 0.0).count();
    let extreme = bg_ratios.iter().filter(|&&r| r.abs() > 5.0 || r.is_nan()).count();
    println!("background pixels without a mask: {} of {} negative, {} beyond ±5 or undefined", negative, bg_ratios.len(), extreme);

    println!("\ncell  true c (nM)  true R   R (subtracted)  c (nM)    R (not subtracted)  c (nM)");
    let sub_a = f.acceptor.map(|v| v - bg);
    let sub_d = f.donor.map(|v| v - bg);
    for l in 1..=8 {
        let m = mask_of(l);
        let area = m.pixels().iter().filter(|&&b| b).count();
        if area < 30 {
            continue;
        }
        let c_true = f.concentration.coords().filter(|&p| m[p]).map(|p| f.concentration[p]).sum::<f32>() / area as f32;
        let r_true = f.concentration.coords().filter(|&p| m[p]).map(|p| sim::true_ratio(f.concentration[p])).sum::<f32>() / area as f32;
        let r = ratio_of_sums(&sub_a, &sub_d, &m);
        let r_raw = ratio_of_sums(&f.acceptor, &f.donor, &m);
        let c = |r: f32| calibrate(r, sim::KD, sim::R_MIN, sim::R_MAX);
        println!("{l:>4} {c_true:>10.0} {r_true:>9.3} {r:>12.3} {:>10.0} {r_raw:>16.3} {:>10.0}", c(r), c(r_raw));
    }

    // 2. Few photons: ratio of sums against mean of ratios, before and after binning
    let dim = sim::field(6.0, 0.0, 0.0, 4);
    println!("\n6 photons per pixel (no background), cells 3 to 8 at 150 nM (true R = {:.3}):", sim::true_ratio(150.0));
    let control = dim.labels.map(|l| l >= 3);
    println!("  ratio of sums {:.3}, mean of pixel ratios {:.3}", ratio_of_sums(&dim.acceptor, &dim.donor, &control), mean_of_ratios(&dim.acceptor, &dim.donor, &control));
    for k in [2, 4] {
        // a block counts if most of it is in the control cells
        let bm = bin(&control.map(|b| if b { 1.0 } else { 0.0 }), k).map(|v| v > 0.5 * (k * k) as f32);
        let (ba, bd) = (bin(&dim.acceptor, k), bin(&dim.donor, k));
        println!("  binned {k} × {k}: ratio of sums {:.3}, mean of pixel ratios {:.3}", ratio_of_sums(&ba, &bd, &bm), mean_of_ratios(&ba, &bd, &bm));
    }

    // 3. The noise of a ratio: 20 000 Poisson pairs
    let mut rng = Rng::new(1);
    for (na, nd) in [(400.0, 1000.0), (100.0, 400.0), (20.0, 5.0), (5.0, 20.0)] {
        let mut ratios = Vec::new();
        let mut zero = 0;
        for _ in 0..20000 {
            let (a, d) = (rng.poisson(na), rng.poisson(nd));
            if d > 0.0 {
                ratios.push(a / d);
            } else {
                zero += 1;
            }
        }
        let m = ratios.iter().sum::<f32>() / ratios.len() as f32;
        let sd = (ratios.iter().map(|r| (r - m).powi(2)).sum::<f32>() / ratios.len() as f32).sqrt();
        let truth = na / nd;
        println!("N_A = {na:>4}, N_D = {nd:>4}: mean ratio {m:.3} (true {truth:.3}, {:+.1} %), CV {:.1} % (predicted {:.1} %), D = 0 in {zero} draws",
            100.0 * (m / truth - 1.0), 100.0 * sd / m, 100.0 * (1.0 / na + 1.0 / nd).sqrt());
    }
    println!("wrote the pictures to {dir}/");
}
