// Runge's phenomenon, the worked Bézier curve with de Casteljau and its curvature comb, and
// three control polygons (B4); pictures in out/bezier/.
//   cargo run -p day06 -- bezier                      (your code)
//   cargo run -p day06 --features solution -- bezier  (the model solution)
use super::plot;
use crate::bezier::*;
use plot::*;

/// The bump of the lecture, 1 / (1 + 25 x²).
fn bump(x: f32) -> f32 {
    1.0 / (1.0 + 25.0 * x * x)
}

/// Piecewise linear interpolation through equispaced samples on [-1, 1].
fn piecewise_linear(ys: &[f32], x: f32) -> f32 {
    let n = ys.len();
    let pos = (x + 1.0) / 2.0 * (n - 1) as f32;
    let i = (pos.floor() as usize).min(n - 2);
    let f = pos - i as f32;
    (1.0 - f) * ys[i] + f * ys[i + 1]
}

fn samples(n: usize) -> (Vec<f32>, Vec<f32>) {
    let xs: Vec<f32> = (0..n).map(|i| -1.0 + 2.0 * i as f32 / (n - 1) as f32).collect();
    let ys = xs.iter().map(|&x| bump(x)).collect();
    (xs, ys)
}

pub fn main() {
    let dir = "out/bezier";

    // Runge: the largest error of the one polynomial and of piecewise linear interpolation.
    println!("bump 1/(1 + 25x²), n equispaced samples on [-1, 1]: largest error");
    println!("   n   polynomial   piecewise linear");
    for n in [5, 7, 9, 11, 15, 21] {
        let (xs, ys) = samples(n);
        let (mut e_poly, mut e_lin) = (0.0f32, 0.0f32);
        for k in 0..=2000 {
            let x = -1.0 + k as f32 / 1000.0;
            e_poly = e_poly.max((lagrange(&xs, &ys, x) - bump(x)).abs());
            e_lin = e_lin.max((piecewise_linear(&ys, x) - bump(x)).abs());
        }
        println!("  {n:2}   {e_poly:10.3}   {e_lin:10.3}");
    }
    let (xs, ys) = samples(11);
    let mut p = Plot::new(600, 400, (-1.05, 1.05), (-0.4, 2.0));
    let grid: Vec<f32> = (0..=1000).map(|k| -1.0 + k as f32 / 500.0).collect();
    p.dashed(&grid.iter().map(|&x| [x, bump(x)]).collect::<Vec<_>>(), GREY);
    p.curve(&grid.iter().map(|&x| [x, piecewise_linear(&ys, x)]).collect::<Vec<_>>(), GREEN);
    p.curve(&grid.iter().map(|&x| [x, lagrange(&xs, &ys, x)]).collect::<Vec<_>>(), RED);
    for i in 0..xs.len() {
        p.dot([xs[i], ys[i]], 4.0, BLACK);
    }
    p.save(format!("{dir}/runge.png"));

    // The worked curve: control polygon, de Casteljau at t = 1/2, the two halves, the comb.
    let ctrl = [[0.0, 0.0], [1.0, 2.0], [3.0, 2.0], [4.0, 0.0]];
    let mut p = Plot::new(640, 400, (-0.5, 4.5), (-0.6, 2.5));
    p.dashed(&ctrl, GREY);
    let curve: Vec<Point> = (0..=200).map(|k| de_casteljau(&ctrl, k as f32 / 200.0)).collect();
    // Curvature comb: a bristle along the normal, length proportional to the curvature,
    // on the side away from the centre of the touching circle.
    let d = derivative(&ctrl);
    for k in 0..=40 {
        let t = k as f32 / 40.0;
        let c = de_casteljau(&ctrl, t);
        let v = de_casteljau(&d, t);
        let len = (v[0] * v[0] + v[1] * v[1]).sqrt();
        let normal = [-v[1] / len, v[0] / len];
        let kappa = curvature(&ctrl, t);
        p.line(c, [c[0] - 1.0 * kappa * normal[0], c[1] - 1.0 * kappa * normal[1]], [0.75, 0.75, 1.0]);
    }
    p.curve(&curve, BLACK);
    let mut level = ctrl.to_vec();
    for colour in [GREEN, ORANGE] {
        level = (0..level.len() - 1).map(|i| lerp(level[i], level[i + 1], 0.5)).collect();
        p.curve(&level, colour);
        for &q in &level {
            p.dot(q, 3.0, colour);
        }
    }
    let (left, right) = split(&ctrl, 0.5);
    for q in &ctrl {
        p.dot(*q, 5.0, BLACK);
    }
    p.dot(de_casteljau(&ctrl, 0.5), 5.0, RED);
    p.save(format!("{dir}/bezier.png"));
    println!("worked curve: B(1/2) = {:?}, weights at 1/2 = {:?}", de_casteljau(&ctrl, 0.5), bernstein(3, 0.5));
    println!("  halves: {left:?} and {right:?}");
    println!("  curvature at t = 0, 1/4, 1/2: {:.3}, {:.3}, {:.3}", curvature(&ctrl, 0.0), curvature(&ctrl, 0.25), curvature(&ctrl, 0.5));

    // B4: three control polygons side by side (shifted by 6 and 12 along x).
    let shapes: [(&str, [Point; 4]); 3] = [
        ("(a)", [[0.0, 0.0], [5.0, 3.0], [-1.0, 3.0], [4.0, 0.0]]),
        ("(b)", [[0.0, 0.0], [1.0, 2.0], [3.0, 2.0], [4.0, 0.0]]),
        ("(c)", [[0.0, 0.0], [4.0, 2.5], [0.0, 2.5], [4.0, 0.0]]),
    ];
    let mut p = Plot::new(900, 300, (-1.5, 17.5), (-0.5, 3.5));
    for (k, (name, poly)) in shapes.iter().enumerate() {
        let shifted: Vec<Point> = poly.iter().map(|q| [q[0] + 6.5 * k as f32, q[1]]).collect();
        p.dashed(&shifted, GREY);
        let curve: Vec<Point> = (0..=300).map(|i| de_casteljau(&shifted, i as f32 / 300.0)).collect();
        p.curve(&curve, BLUE);
        for q in &shifted {
            p.dot(*q, 4.0, BLACK);
        }
        let v = de_casteljau(&derivative(poly), 0.5);
        println!("B4 {name}: velocity at t = 1/2 is ({:.2}, {:.2})", v[0], v[1]);
    }
    p.save(format!("{dir}/shapes.png"));
    println!("wrote runge.png, bezier.png and shapes.png to {dir}/");
}
