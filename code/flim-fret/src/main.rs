// FLIM-FRET and NADH on simulated cells, with your code: E from the phasor line, maps of the
// interacting fraction, and three species in a triangle; pictures in out/flim-fret/.
//   cargo run -p flim-fret                      (your code)
//   cargo run -p flim-fret --features solution  (the model solution)

use flim_fret::*;
use imgcourse::*;
use std::f32::consts::PI;

const OMEGA: f32 = 2.0 * PI * 0.08; // 80 MHz

fn single(tau: f32) -> (f32, f32) {
    let wt = OMEGA * tau;
    (1.0 / (1.0 + wt * wt), wt / (1.0 + wt * wt))
}

/// The (calibrated) phasor of `n` photons: each comes from species k with probability
/// `species[k].1` and arrives after an exponential time `species[k].0`.
fn pixel_phasor(species: &[(f32, f32)], n: usize, rng: &mut Rng) -> (f32, f32) {
    let (mut g, mut s) = (0.0, 0.0);
    for _ in 0..n {
        let (mut u, mut k) = (rng.uniform(), 0);
        while k + 1 < species.len() && u >= species[k].1 {
            u -= species[k].1;
            k += 1;
        }
        let t = -species[k].0 * (1.0 - rng.uniform()).ln();
        g += (OMEGA * t).cos();
        s += (OMEGA * t).sin();
    }
    (g / n as f32, s / n as f32)
}

/// How far along the line from `a` to `b` the point `p` lies (projection).
fn fraction_along(p: (f32, f32), a: (f32, f32), b: (f32, f32)) -> f32 {
    let (dx, dy) = (b.0 - a.0, b.1 - a.1);
    ((p.0 - a.0) * dx + (p.1 - a.1) * dy) / (dx * dx + dy * dy)
}

fn main() {
    let dir = "out/flim-fret";
    std::fs::create_dir_all(dir).unwrap();
    let mut rng = Rng::new(3);
    let size = 128;
    let (img, labels) = samples::nuclei(size, 14, 9);
    let cells: Vec<(usize, usize)> = img.coords().filter(|&(x, y)| labels[(x, y)] > 0).collect();

    // 1. FRET: donor 2.6 ns; in the FRET state 1.3 ns (E = 0.5). Each nucleus has its own
    //    fraction α of interacting donors, between 0 and 0.6.
    let (tau_d, tau_fret) = (2.6, 1.3);
    let alpha_of = |l: u32| 0.6 * ((l * 37) % 10) as f32 / 9.0;
    let true_alpha = labels.map(|l| if l > 0 { alpha_of(l) } else { 0.0 });
    save_gray(&true_alpha, format!("{dir}/alpha-true.png"));
    let donor = single(tau_d);
    println!("FRET: τD = {tau_d} ns, τDA = {tau_fret} ns (E = 0.5); α per nucleus from 0 to 0.6, mean over pixels {:.3}",
        cells.iter().map(|&(x, y)| true_alpha[(x, y)]).sum::<f32>() / cells.len() as f32);
    for photons in [300.0, 40.0] {
        let mut phasors = Image::new(size, size, (0.0f32, 0.0f32));
        for &(x, y) in &cells {
            let f = photon_fraction(true_alpha[(x, y)], tau_fret, tau_d);
            let n = ((photons / 0.8) * (img[(x, y)] - 0.1)).max(5.0) as usize;
            phasors[(x, y)] = pixel_phasor(&[(tau_d, 1.0 - f), (tau_fret, f)], n, &mut rng);
        }
        // E from the line from the donor point through the mean of the cloud.
        let n = cells.len() as f32;
        let mean = (cells.iter().map(|&(x, y)| phasors[(x, y)].0).sum::<f32>() / n, cells.iter().map(|&(x, y)| phasors[(x, y)].1).sum::<f32>() / n);
        let (tau_da, _) = fret_from_line(donor, mean, OMEGA);
        let fret_point = single(tau_da);
        println!("\n{photons} photons in a bright pixel: line gives τDA = {tau_da:.2} ns, E = {:.2}", fret_efficiency(tau_da, tau_d));
        // α per pixel: the position along the line, as a photon fraction, then as a fraction of donors.
        for radius in [0isize, 1] {
            let alpha = Image::from_fn(size, size, |x, y| {
                if labels[(x, y)] == 0 {
                    return 0.0;
                }
                let (mut g, mut s, mut k) = (0.0, 0.0, 0.0);
                for dy in -radius..=radius {
                    for dx in -radius..=radius {
                        let (xx, yy) = (x as isize + dx, y as isize + dy);
                        if labels.contains(xx, yy) && labels[(xx as usize, yy as usize)] > 0 {
                            g += phasors[(xx as usize, yy as usize)].0;
                            s += phasors[(xx as usize, yy as usize)].1;
                            k += 1.0;
                        }
                    }
                }
                let f = fraction_along((g / k, s / k), donor, fret_point).clamp(0.0, 1.0);
                interacting_fraction(f, tau_da, tau_d)
            });
            let mean_alpha = cells.iter().map(|&(x, y)| alpha[(x, y)]).sum::<f32>() / n;
            println!("  {}: mean α over the nucleus pixels {mean_alpha:.3}", if radius == 0 { "per pixel " } else { "3 × 3 mean" });
            save_gray(&alpha, format!("{dir}/alpha-{}photons-{}.png", photons as u32, if radius == 0 { "1x1" } else { "3x3" }));
        }
    }

    // 2. NADH: free (0.4 ns) and bound (2.5 ns), equally bright per unit lifetime; the molecule
    //    fraction bound differs per nucleus (0.1 to 0.5). Nuclei with an odd label also hold a
    //    label dye of 4.5 ns, giving 20 % of their photons.
    let (free, bound, label) = (single(0.4), single(2.5), single(4.5));
    let bound_of = |l: u32| 0.1 + 0.4 * ((l * 53) % 7) as f32 / 6.0;
    let (mut truth, mut two, mut three) = (0.0, 0.0, 0.0);
    let mut composite = Image::new(size, size, [0.0f32; 3]);
    let molecules = |f_free: f32, f_bound: f32| (f_bound / 2.5) / (f_bound / 2.5 + f_free / 0.4);
    for &(x, y) in &cells {
        let l = labels[(x, y)];
        let a = bound_of(l);
        let f_bound = a * 2.5 / (a * 2.5 + (1.0 - a) * 0.4); // photon fraction of bound NADH
        let f_label = if l % 2 == 1 { 0.2 } else { 0.0 };
        let species = [(0.4, (1.0 - f_label) * (1.0 - f_bound)), (2.5, (1.0 - f_label) * f_bound), (4.5, f_label)];
        let n = (375.0 * (img[(x, y)] - 0.1)).max(5.0) as usize;
        let p = pixel_phasor(&species, n, &mut rng);
        let fb = fraction_along(p, free, bound).clamp(0.0, 1.0);
        let tri = barycentric(p, free, bound, label).map(|v| v.max(0.0));
        truth += a;
        two += molecules(1.0 - fb, fb);
        three += molecules(tri[0], tri[1]);
        composite[(x, y)] = [tri[0], tri[1], tri[2]];
    }
    let n = cells.len() as f32;
    println!("\nNADH: mean molecule fraction bound: true {:.3}, free–bound line {:.3}, triangle {:.3}", truth / n, two / n, three / n);
    save_rgb(&composite, format!("{dir}/nadh-triangle.png"));
    println!("wrote the pictures to {dir}/");
}
