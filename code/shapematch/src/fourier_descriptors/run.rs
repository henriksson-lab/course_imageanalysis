// Fourier descriptors of traced outlines: the by-hand examples, the letter F matched to a
// turned, scaled copy (with and without arc-length resampling, and its mirror image),
// smoothing by truncation, and "which two blobs are the same shape?". Pictures in
// out/fourier-descriptors/.
//   cargo run -p shapematch -- fourier-descriptors                      (your code)
//   cargo run -p shapematch --features solution -- fourier-descriptors  (the model solution)

use crate::fourier_descriptors::*;
use imgcourse::{save_rgb, Complex, Image};

const N: usize = 64;
const DIR: &str = "out/fourier-descriptors";

fn c(z: Complex) -> String {
    let tidy = |v: f32| if v.abs() < 5e-4 { 0.0 } else { v }; // no "−0.000"
    let (re, im) = (tidy(z.re), tidy(z.im));
    let s = if im < 0.0 { '−' } else { '+' };
    format!("{re:6.3} {s} {:.3}i", im.abs())
}

/// The mask in light grey, then the outlines in their colours.
fn picture(mask: &Image<bool>, outlines: &[(&[Complex], [f32; 3])]) -> Image<[f32; 3]> {
    let mut img = mask.map(|v| if v { [0.82, 0.82, 0.82] } else { [1.0, 1.0, 1.0] });
    for (z, col) in outlines {
        draw_outline(&mut img, z, *col);
    }
    img
}

const RED: [f32; 3] = [0.84, 0.15, 0.16];
const BLUE: [f32; 3] = [0.12, 0.47, 0.71];

fn report(what: &str, fit: &Match) {
    println!(
        "  {what:<28} start m = {:2}, angle {:7.2}°, scale {:.3}, score {:.4}, misfit {:4.1} %{}",
        fit.m,
        fit.angle_deg(),
        fit.scale(),
        fit.score,
        100.0 * fit.misfit(),
        if fit.mirrored { ", MIRRORED" } else { "" }
    );
}

fn distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(p, q)| (p - q) * (p - q)).sum::<f32>().sqrt()
}

pub fn main() {
    // ---- by hand ----
    println!("By hand (the c_k with k = −1, 0, 1, 2):");
    let i = Complex::new(0.0, 1.0);
    let one = Complex::from(1.0);
    let square = [one, i, -one, -i];
    let ellipse = [one * 2.0, i, one * -2.0, -i];
    let rotated: Vec<Complex> = ellipse.iter().map(|&z| i * z).collect();
    let later = restart(&ellipse, 1);
    for (name, z) in [("(1, i, −1, −i)", &square[..]), ("(2, i, −2, −i)", &ellipse[..]), ("  × i (turned 90°)", &rotated[..]), ("  started one later", &later[..])] {
        let d = descriptors(z);
        let p = coef(&d, 1) * coef(&d, -1);
        println!("  {name:<20} c_−1 = {}, c_0 = {}, c_1 = {}, c_2 = {};  c_1·c_−1 = {}", c(coef(&d, -1)), c(coef(&d, 0)), c(coef(&d, 1)), c(coef(&d, 2)), c(p));
    }

    // ---- the letter F ----
    let f = letter_f();
    let (mask_a, trace_a) = raster_outline(&f, 40.0, 0.0);
    let (mask_b, trace_b) = raster_outline(&f, 60.0, 30.0);
    println!("\nThe letter F: A upright (40 px per unit), B turned 30° and 1.5 times as large.");
    println!("  traced: {} boundary pixels in A, {} in B; both resampled to N = {N}", trace_a.len(), trace_b.len());
    let a = resample_arc_length(&trace_a, N);
    let b = resample_arc_length(&trace_b, N);
    let fit = match_outlines(&a, &b, false);
    report("arc length:", &fit);
    save_rgb(&picture(&mask_b, &[(&b, RED), (&aligned(&a, &fit), BLUE)]), format!("{DIR}/f-match.png"));
    let ai = resample_by_index(&trace_a, N);
    let bi = resample_by_index(&trace_b, N);
    let fit_i = match_outlines(&ai, &bi, false);
    report("equal steps in index:", &fit_i);
    save_rgb(&picture(&mask_b, &[(&bi, RED), (&aligned(&ai, &fit_i), BLUE)]), format!("{DIR}/f-match-index.png"));

    // the mirror image of F (y → −y), turned 30°
    let fm: Vec<Complex> = f.iter().map(|z| z.conj()).collect();
    let (mask_m, trace_m) = raster_outline(&fm, 60.0, 30.0);
    let bm = resample_arc_length(&trace_m, N);
    let no = match_outlines(&a, &bm, false);
    let yes = match_outlines(&a, &bm, true);
    println!("  B = the mirror image of F, turned 30°:");
    report("  mirror not allowed:", &no);
    report("  mirror allowed:", &yes);
    save_rgb(&picture(&mask_m, &[(&bm, RED), (&aligned(&a, &no), BLUE)]), format!("{DIR}/f-mirror.png"));
    save_rgb(&picture(&mask_m, &[(&bm, RED), (&aligned(&a, &yes), BLUE)]), format!("{DIR}/f-mirror-allowed.png"));
    let (sa, sb, sm) = (magnitude_signature(&descriptors(&a), 8), magnitude_signature(&descriptors(&b), 8), magnitude_signature(&descriptors(&bm), 8));
    println!("  magnitude signatures (|k| ≤ 8): distance A–B {:.3}, A–mirror image {:.3}", distance(&sa, &sb), distance(&sa, &sm));

    // smoothing by truncation
    let ca = descriptors(&a);
    print!("  smoothing A: rms distance to the outline with |k| ≤ K:");
    for k in [1, 2, 4, 8, 16, 32] {
        let r = reconstruct(&ca, k);
        let rms = (a.iter().zip(&r).map(|(p, q)| (*p - *q).norm_sqr()).sum::<f32>() / N as f32).sqrt();
        print!("  K={k}: {rms:.2} px");
        save_rgb(&picture(&mask_a, &[(&r, BLUE)]), format!("{DIR}/smooth-{k:02}.png"));
    }
    println!();

    // ---- which two blobs are the same shape? ----
    let shapes: [&[(f32, f32, f32)]; 5] = [
        &[(2.0, 0.25, 0.0), (3.0, 0.12, 1.0)],
        &[(2.0, 0.2, 0.5), (3.0, 0.15, 0.3), (5.0, 0.05, 0.0)],
        &[(3.0, 0.2, 0.0), (2.0, 0.1, 1.2)],
        &[(2.0, 0.2, 0.5), (3.0, 0.15, 0.3), (5.0, 0.05, 0.0)],
        &[(2.0, 0.3, 2.0), (4.0, 0.08, 0.4)],
    ];
    let poses = [(16.0, 10.0), (14.0, 75.0), (18.0, -30.0), (20.0, 200.0), (15.0, 120.0)];
    let outlines: Vec<Vec<Complex>> = shapes
        .iter()
        .zip(poses)
        .map(|(t, (s, ang))| resample_arc_length(&raster_outline(&blob(t), s, ang).1, N))
        .collect();
    println!("\nFive blobs, rasterised at different sizes and angles. Misfit of every pair (%), and the");
    println!("distance of the magnitude signatures (|k| ≤ 8):");
    println!("        {}", (1..=5).map(|j| format!("{j:>6}")).collect::<String>());
    for (i, oi) in outlines.iter().enumerate() {
        let mut row = format!("  {:>2}    ", i + 1);
        for oj in &outlines {
            row += &format!("{:6.1}", 100.0 * match_outlines(oi, oj, false).misfit());
        }
        row += "   |";
        let si = magnitude_signature(&descriptors(oi), 8);
        for oj in &outlines {
            row += &format!("{:6.3}", distance(&si, &magnitude_signature(&descriptors(oj), 8)));
        }
        println!("{row}");
    }
    println!("\nwrote the pictures to {DIR}/ (grey: the mask of B, red: B, blue: A moved onto B)");
}
