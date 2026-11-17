// Basis functions, corner cutting, local control and exact circles; pictures in out/bsplines/.
//   cargo run -p bsplines                      (your code)
//   cargo run -p bsplines --features solution  (the model solution)

mod plot;

use bsplines::*;
use plot::*;

const COLOURS: [Colour; 7] = [RED, ORANGE, GREEN, BLUE, PURPLE, [0.0, 0.6, 0.6], [0.6, 0.4, 0.2]];

/// Plot all basis functions of degree 3 on `knots` and their sum (dashed).
fn plot_basis(knots: &[f32], path: String) {
    let n = knots.len() - 4;
    let (lo, hi) = (knots[0], knots[knots.len() - 1]);
    let mut p = Plot::new(700, 260, (lo - 0.3, hi + 0.3), (-0.1, 1.15));
    let ts: Vec<f32> = (0..1000).map(|k| lo + (hi - lo) * k as f32 / 1000.0).collect();
    for i in 0..n {
        let pts: Vec<Point> = ts.iter().map(|&t| [t, basis(i, 3, knots, t)]).collect();
        p.curve(&pts, COLOURS[i % COLOURS.len()]);
    }
    let sum: Vec<Point> = ts.iter().map(|&t| [t, (0..n).map(|i| basis(i, 3, knots, t)).sum()]).collect();
    p.dashed(&sum, BLACK);
    for &k in knots {
        p.dot([k, 0.0], 3.0, BLACK);
    }
    p.save(path);
}

/// The largest distance from the origin minus `r`, over the curve points.
fn radius_error(points: &[Point], r: f32) -> f32 {
    points.iter().map(|c| ((c[0] * c[0] + c[1] * c[1]).sqrt() - r).abs()).fold(0.0, f32::max)
}

fn main() {
    let dir = "out/bsplines";

    // The basis: uniform knots, and clamped knots (end knots repeated 4 times).
    let uniform: Vec<f32> = (0..=10).map(|k| k as f32).collect();
    plot_basis(&uniform, format!("{dir}/basis-uniform.png"));
    let clamped = [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0, 4.0];
    plot_basis(&clamped, format!("{dir}/basis-clamped.png"));

    // Corner cutting: the closed uniform cubic on a square.
    let square = [[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]];
    let curve: Vec<Point> = (0..=400).map(|k| closed_cubic(&square, 4.0 * k as f32 / 400.0)).collect();
    let mut p = Plot::new(400, 400, (-0.5, 4.5), (-0.5, 4.5));
    p.curve(&[square[0], square[1], square[2], square[3], square[0]], GREY);
    p.curve(&curve, BLUE);
    for k in 0..4 {
        p.dot(square[k], 5.0, BLACK);
        p.dot(closed_cubic(&square, k as f32), 4.0, RED);
    }
    p.save(format!("{dir}/corner-cutting.png"));
    let dist: Vec<f32> = curve.iter().map(|c| ((c[0] - 2.0).powi(2) + (c[1] - 2.0).powi(2)).sqrt()).collect();
    println!(
        "closed cubic on the square: distance from the centre between {:.3} and {:.3}",
        dist.iter().cloned().fold(f32::INFINITY, f32::min),
        dist.iter().cloned().fold(0.0, f32::max)
    );

    // Local control: a clamped cubic with 8 control points; move P4 up by 2.
    let mut ctrl = [[0.0, 0.0], [1.0, 2.0], [2.5, 2.5], [4.0, 1.0], [5.0, 0.5], [6.5, 2.0], [8.0, 2.5], [9.0, 0.5]];
    let knots = [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 5.0, 5.0];
    let ts: Vec<f32> = (0..=500).map(|k| (5.0 * k as f32 / 500.0).min(4.9999)).collect();
    let before: Vec<Point> = ts.iter().map(|&t| bspline_point(&ctrl, 3, &knots, t)).collect();
    let old_ctrl = ctrl;
    ctrl[4][1] += 2.0;
    let after: Vec<Point> = ts.iter().map(|&t| bspline_point(&ctrl, 3, &knots, t)).collect();
    let mut p = Plot::new(700, 300, (-0.5, 9.5), (-0.5, 3.5));
    p.curve(&old_ctrl, [0.85; 3]);
    p.curve(&ctrl, GREY);
    p.curve(&before, BLACK);
    let mut changed = (f32::INFINITY, f32::NEG_INFINITY);
    for k in 0..ts.len() - 1 {
        let moved = (after[k][0] - before[k][0]).hypot(after[k][1] - before[k][1]) > 1e-4;
        if moved {
            changed = (changed.0.min(ts[k]), changed.1.max(ts[k]));
        }
        p.line(after[k], after[k + 1], if moved { ORANGE } else { BLACK });
    }
    for q in &ctrl {
        p.dot(*q, 4.0, BLACK);
    }
    p.save(format!("{dir}/local-control.png"));
    println!("moving P4 changes the curve for t from {:.2} to {:.2} (the curve runs from 0 to 5)", changed.0, changed.1);

    // NURBS: a quarter circle with weights 1, w, 1, and a full circle from 9 control points.
    let quarter = [[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
    let qknots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    let mut p = Plot::new(500, 500, (-1.2, 1.2), (-1.2, 1.2));
    let circle: Vec<Point> = (0..=400).map(|k| (k as f32 * std::f32::consts::TAU / 400.0).sin_cos()).map(|(s, c)| [c, s]).collect();
    p.dashed(&circle, GREY);
    for (w, colour) in [(1.0, ORANGE), (std::f32::consts::FRAC_1_SQRT_2, BLUE)] {
        let pts: Vec<Point> = (0..=200).map(|k| nurbs_point(&quarter, &[1.0, w, 1.0], 2, &qknots, (k as f32 / 200.0).min(0.9999))).collect();
        p.curve(&pts, colour);
        println!("quarter circle, middle weight {w:.4}: largest radius error {:.5}", radius_error(&pts, 1.0));
    }
    let r = std::f32::consts::FRAC_1_SQRT_2;
    let nine = [[1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [-1.0, 1.0], [-1.0, 0.0], [-1.0, -1.0], [0.0, -1.0], [1.0, -1.0], [1.0, 0.0]];
    let weights = [1.0, r, 1.0, r, 1.0, r, 1.0, r, 1.0];
    let cknots = [0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 4.0];
    let full: Vec<Point> = (0..=800).map(|k| nurbs_point(&nine, &weights, 2, &cknots, (4.0 * k as f32 / 800.0).min(3.9999))).collect();
    let shifted: Vec<Point> = full.iter().map(|c| [0.6 * c[0], 0.6 * c[1]]).collect();
    p.curve(&shifted, GREEN);
    for q in &quarter {
        p.dot(*q, 5.0, BLACK);
    }
    p.save(format!("{dir}/circles.png"));
    println!("full circle from 9 control points: largest radius error {:.5}", radius_error(&full, 1.0));
    println!("wrote basis-uniform.png, basis-clamped.png, corner-cutting.png, local-control.png and circles.png to {dir}/");
}
