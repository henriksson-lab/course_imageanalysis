// Smoothing noisy samples with λ chosen by leave-one-out, and the perimeter of digital discs
// from a closed smoothing spline; pictures in out/spline-fit/.
//   cargo run -p splines -- spline-fit                      (your code)
//   cargo run -p splines --features solution -- spline-fit  (the model solution)
use imgcourse::{Image, Rng};
use crate::common::*;
use crate::spline_fit::*;

/// The true curve: a sine with a kink at x = 6.
fn truth(x: f32) -> f32 {
    x.sin() + 0.5 * (x - 6.0).max(0.0)
}

/// RMS distance between the natural spline through `values` (at the sample positions) and the
/// true curve, on a fine grid over [0, 10].
fn rms_to_truth(values: &[f32], xs: &[f32]) -> f32 {
    let m = natural_spline(values);
    let h = xs[1] - xs[0];
    let mut sum = 0.0;
    for k in 0..=1000 {
        let x = 10.0 * k as f32 / 1000.0;
        let d = natural_eval(values, &m, x / h) - truth(x);
        sum += d * d;
    }
    (sum / 1001.0).sqrt()
}

/// The curve through `values` as plot points (the natural spline, x in [0, 10]).
fn curve_points(values: &[f32], xs: &[f32]) -> Vec<Point> {
    let m = natural_spline(values);
    let h = xs[1] - xs[0];
    (0..=500).map(|k| 10.0 * k as f32 / 500.0).map(|x| [x, natural_eval(values, &m, x / h)]).collect()
}

/// A digital disc of radius r on a 64 × 64 grid, centred on the pixel corner (32, 32).
fn disc(r: f32) -> Image<bool> {
    Image::from_fn(64, 64, |x, y| {
        let (dx, dy) = (x as f32 + 0.5 - 32.0, y as f32 + 0.5 - 32.0);
        dx * dx + dy * dy <= r * r
    })
}

/// The midpoints of the boundary edges, in order of their angle around (32, 32).
fn edge_midpoints(mask: &Image<bool>) -> Vec<Point> {
    let mut pts = Vec::new();
    for (x, y) in mask.coords() {
        if !mask[(x, y)] {
            continue;
        }
        let (fx, fy) = (x as f32, y as f32);
        for (dx, dy, mid) in [(1, 0, [fx + 1.0, fy + 0.5]), (-1, 0, [fx, fy + 0.5]), (0, 1, [fx + 0.5, fy + 1.0]), (0, -1, [fx + 0.5, fy])] {
            if !mask.contains(x as isize + dx, y as isize + dy) || !mask[((x as isize + dx) as usize, (y as isize + dy) as usize)] {
                pts.push(mid);
            }
        }
    }
    pts.sort_by(|a, b| (a[1] - 32.0).atan2(a[0] - 32.0).total_cmp(&(b[1] - 32.0).atan2(b[0] - 32.0)));
    pts
}

pub fn main() {
    let dir = "out/spline-fit";

    // B2: the natural spline through (0, 0), (1, 1), (2, 0).
    let m = natural_spline(&[0.0, 1.0, 0.0]);
    println!("natural spline through (0,0), (1,1), (2,0): M = {m:?}, s(0.5) = {}", natural_eval(&[0.0, 1.0, 0.0], &m, 0.5));

    // 30 noisy samples on [0, 10].
    let n = 30;
    let xs: Vec<f32> = (0..n).map(|i| 10.0 * i as f32 / (n - 1) as f32).collect();
    let mut rng = Rng::new(7);
    let ys: Vec<f32> = xs.iter().map(|&x| truth(x) + 0.25 * rng.normal()).collect();

    println!("\nsmoothing 30 noisy samples (noise 0.25):");
    println!("      λ       LOO    tr H   RMS to truth");
    let mut best = (f32::INFINITY, 0.0);
    for k in -16..=24 {
        let lambda = 10f32.powf(k as f32 / 4.0);
        let (loo, df) = loo_cv(&ys, lambda);
        if loo < best.0 {
            best = (loo, lambda);
        }
        if k % 4 == 0 {
            println!("  {lambda:7.0e}   {loo:7.4}  {df:6.2}   {:.3}", rms_to_truth(&smooth(&ys, lambda, false), &xs));
        }
    }
    let lambda_best = best.1;
    let f_best = smooth(&ys, lambda_best, false);
    println!("LOO minimum at λ = {lambda_best:.3}: LOO {:.4}, tr H {:.2}, RMS to truth {:.3}", best.0, loo_cv(&ys, lambda_best).1, rms_to_truth(&f_best, &xs));
    println!("interpolating spline (λ = 0): RMS to truth {:.3}", rms_to_truth(&ys, &xs));

    let mut p = Plot::new(700, 360, (-0.3, 10.3), (-1.6, 2.8));
    p.dashed(&(0..=500).map(|k| 10.0 * k as f32 / 500.0).map(|x| [x, truth(x)]).collect::<Vec<_>>(), GREY);
    p.curve(&curve_points(&ys, &xs), ORANGE);
    p.curve(&curve_points(&smooth(&ys, 1e6, false), &xs), PURPLE);
    p.curve(&curve_points(&f_best, &xs), BLUE);
    for i in 0..n {
        p.dot([xs[i], ys[i]], 3.5, BLACK);
    }
    p.save(format!("{dir}/smoothing.png"));

    // Perimeters of digital discs.
    println!("\nperimeter of a digital disc: edge count, closed smoothing spline for several λ, truth");
    for r in [10.0, 3.0] {
        let mask = disc(r);
        let pts = edge_midpoints(&mask);
        let mut line = format!("  r = {r:2}: edges {:3}", edge_count(&mask));
        for lambda in [0.0, 0.1, 1.0, 10.0, 100.0] {
            line += &format!(" | λ {lambda}: {:.1}", closed_length(&closed_fit(&pts, lambda), 10));
        }
        println!("{line} | true {:.1}", std::f32::consts::TAU * r);
    }
    // Picture of the radius-10 disc: edge midpoints, the true circle, fits with λ = 0 and 1.
    let pts = edge_midpoints(&disc(10.0));
    let mut p = Plot::new(500, 500, (20.5, 43.5), (20.5, 43.5));
    let circle: Vec<Point> = (0..=300).map(|k| (std::f32::consts::TAU * k as f32 / 300.0).sin_cos()).map(|(s, c)| [32.0 + 10.0 * c, 32.0 + 10.0 * s]).collect();
    p.dashed(&circle, GREEN);
    for (lambda, colour) in [(0.0, ORANGE), (1.0, BLUE)] {
        let ctrl = closed_fit(&pts, lambda);
        let curve: Vec<Point> = (0..=pts.len() * 10).map(|k| closed_cubic(&ctrl, k as f32 / 10.0)).collect();
        p.curve(&curve, colour);
    }
    for q in &pts {
        p.dot(*q, 2.5, BLACK);
    }
    p.save(format!("{dir}/outline.png"));
    println!("wrote smoothing.png and outline.png to {dir}/");
}
