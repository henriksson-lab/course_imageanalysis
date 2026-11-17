// Phasors of a simulated FLIM image of two species, with your code: the phasor plot before and
// after calibration, the line through the cloud, and a fraction map; pictures in out/phasor/.
//   cargo run -p flim -- phasor                      (your code)
//   cargo run -p flim --features solution -- phasor  (the model solution)

use imgcourse::*;
use crate::phasor::*;
use std::f32::consts::PI;

const PERIOD: f32 = 12.5;
const BINS: usize = 64;

/// A TCSPC histogram, photon by photon: each photon comes from species k with probability
/// `species[k].1` and waits an exponential time `species[k].0`, plus a Gaussian IRF.
fn simulate(species: &[(f32, f32)], photons: usize, irf: (f32, f32), rng: &mut Rng) -> Vec<f32> {
    let mut counts = vec![0.0; BINS];
    for _ in 0..photons {
        let (mut u, mut k) = (rng.uniform(), 0);
        while k + 1 < species.len() && u >= species[k].1 {
            u -= species[k].1;
            k += 1;
        }
        let t = irf.0 + irf.1 * rng.normal() - species[k].0 * (1.0 - rng.uniform()).ln();
        let j = ((t.rem_euclid(PERIOD) / PERIOD * BINS as f32) as usize).min(BINS - 1);
        counts[j] += 1.0;
    }
    counts
}

/// The phasor plot: a 2D histogram of the points (brighter = more pixels), the universal
/// semicircle in grey and `marks` in red. g from 0 to 1 across, s from 0 to 0.7 upwards.
fn phasor_plot(points: &[(f32, f32)], marks: &[(f32, f32)]) -> Image<[f32; 3]> {
    let (w, h, scale) = (420, 300, 400.0);
    let to_px = |g: f32, s: f32| ((10.0 + g * scale) as isize, (h as f32 - 10.0 - s * scale) as isize);
    let mut hist = Image::new(w, h, 0.0f32);
    for &(g, s) in points {
        let (x, y) = to_px(g, s);
        if hist.contains(x, y) {
            hist[(x as usize, y as usize)] += 1.0;
        }
    }
    let top = hist.max().max(1.0);
    let mut img = hist.map(|c| if c > 0.0 { let v = 0.25 + 0.75 * (c.ln_1p() / top.ln_1p()); [v, v, v] } else { [0.0; 3] });
    for k in 0..=1000 {
        let a = PI * k as f32 / 1000.0;
        let (x, y) = to_px(0.5 + 0.5 * a.cos(), 0.5 * a.sin());
        if img.contains(x, y) && img[(x as usize, y as usize)] == [0.0; 3] {
            img[(x as usize, y as usize)] = [0.45, 0.45, 0.45];
        }
    }
    for &(g, s) in marks {
        let (cx, cy) = to_px(g, s);
        for dy in -3..=3 {
            for dx in -3..=3 {
                if dx * dx + dy * dy <= 9 && img.contains(cx + dx, cy + dy) {
                    img[((cx + dx) as usize, (cy + dy) as usize)] = [1.0, 0.2, 0.2];
                }
            }
        }
    }
    img
}

/// The straight line through a cloud of points (its centre and main direction), and where it
/// meets the semicircle: the two lifetimes, without any fit (None if it misses the semicircle).
fn line_lifetimes(points: &[(f32, f32)], omega: f32) -> Option<(f32, f32)> {
    let n = points.len() as f32;
    let (mg, ms) = (points.iter().map(|p| p.0).sum::<f32>() / n, points.iter().map(|p| p.1).sum::<f32>() / n);
    let (mut cgg, mut css, mut cgs) = (0.0, 0.0, 0.0);
    for &(g, s) in points {
        cgg += (g - mg) * (g - mg);
        css += (s - ms) * (s - ms);
        cgs += (g - mg) * (s - ms);
    }
    let angle = 0.5 * (2.0 * cgs).atan2(cgg - css);
    let (dg, ds) = (angle.cos(), angle.sin());
    // (mg + t dg − ½)² + (ms + t ds)² = ¼, a quadratic in t
    let (b, c) = (2.0 * ((mg - 0.5) * dg + ms * ds), (mg - 0.5).powi(2) + ms * ms - 0.25);
    if b * b - 4.0 * c < 0.0 {
        return None;
    }
    let root = (b * b - 4.0 * c).sqrt();
    let tau_at = |t: f32| (ms + t * ds) / ((mg + t * dg) * omega);
    let (t1, t2) = (tau_at((-b - root) / 2.0), tau_at((-b + root) / 2.0));
    Some((t1.min(t2), t1.max(t2)))
}

pub fn main() {
    let dir = "out/phasor";
    std::fs::create_dir_all(dir).unwrap();
    let omega = 2.0 * PI / PERIOD;
    let irf = (0.6, 0.3); // delay and width of the IRF (ns)
    let mut rng = Rng::new(2);
    let (short, long) = (0.6, 3.5);
    let (p_short, p_long) = (single_phasor(short, omega), single_phasor(long, omega));

    // The reference: a 4 ns dye measured through the same IRF.
    let reference = phasor(&simulate(&[(4.0, 1.0)], 1_000_000, irf, &mut rng), PERIOD);
    let truth = single_phasor(4.0, omega);
    let polar = |p: (f32, f32)| (p.1.atan2(p.0).to_degrees(), (p.0 * p.0 + p.1 * p.1).sqrt());
    println!("4 ns reference: measured φ = {:.1}°, m = {:.3}; true φ = {:.1}°, m = {:.3}", polar(reference).0, polar(reference).1, polar(truth).0, polar(truth).1);

    // A FLIM image of nuclei: two species, the photon fraction of the long one rising from 0.15
    // on the left to 0.85 on the right. About 300 photons in a bright pixel.
    let size = 128;
    let (img, labels) = samples::nuclei(size, 14, 5);
    let mut raw = vec![];
    let mut calibrated = vec![];
    let mut fraction = Image::new(size, size, 0.0f32);
    let mut errors = vec![];
    for (x, y) in img.coords() {
        if labels[(x, y)] == 0 {
            continue;
        }
        let f = 0.15 + 0.7 * x as f32 / size as f32;
        let n = (500.0 * (img[(x, y)] - 0.1)).max(1.0) as usize;
        let counts = simulate(&[(short, 1.0 - f), (long, f)], n, irf, &mut rng);
        let p = phasor(&counts, PERIOD);
        let c = calibrate(p, reference, 4.0, omega);
        raw.push(p);
        calibrated.push(c);
        let est = fraction_along(c, p_short, p_long);
        fraction[(x, y)] = est.clamp(0.0, 1.0);
        errors.push(est - f);
    }
    save_rgb(&phasor_plot(&raw, &[p_short, p_long]), format!("{dir}/phasor-uncalibrated.png"));
    save_rgb(&phasor_plot(&calibrated, &[p_short, p_long]), format!("{dir}/phasor-calibrated.png"));
    save_gray(&fraction, format!("{dir}/fraction-long.png"));
    let m = errors.iter().sum::<f32>() / errors.len() as f32;
    let sd = (errors.iter().map(|e| (e - m) * (e - m)).sum::<f32>() / errors.len() as f32).sqrt();
    println!("{} nucleus pixels; photon fraction of the 3.5 ns species by the lever rule: error {m:+.3} ± {sd:.3}", errors.len());
    for (name, cloud) in [("uncalibrated", &raw), ("calibrated", &calibrated)] {
        match line_lifetimes(cloud, omega) {
            Some((a, b)) => println!("line through the {name} cloud meets the semicircle at {a:.2} and {b:.2} ns (true {short} and {long})"),
            None => println!("line through the {name} cloud misses the semicircle"),
        }
    }

    // One mixture pixel with many photons: half the photons from each species.
    let p = calibrate(phasor(&simulate(&[(short, 0.5), (long, 0.5)], 1_000_000, irf, &mut rng), PERIOD), reference, 4.0, omega);
    let (tp, tm) = phase_and_modulation_lifetimes(p.0, p.1, omega);
    println!("half and half, 10⁶ photons: phasor ({:.3}, {:.3}), τφ = {tp:.2} ns, τm = {tm:.2} ns", p.0, p.1);
    println!("wrote the pictures to {dir}/");
}
