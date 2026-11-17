// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p day21 ght::

use day21::ght::sim::EdgePoint;
use day21::ght::*;
use std::f32::consts::{FRAC_PI_2, PI};

fn ep(x: f32, y: f32, theta: f32) -> EdgePoint {
    EdgePoint { x, y, theta, strength: 1.0 }
}

/// The edge points of the bright square of B1 with top-left corner (x0, y0): three per side,
/// gradient pointing into the square.
fn square(x0: f32, y0: f32) -> Vec<EdgePoint> {
    let mut v = Vec::new();
    for i in 1..=3 {
        let i = i as f32;
        v.push(ep(x0 + i, y0, FRAC_PI_2)); // top side: gradient down
        v.push(ep(x0 + i, y0 + 4.0, -FRAC_PI_2)); // bottom: up
        v.push(ep(x0, y0 + i, 0.0)); // left: right
        v.push(ep(x0 + 4.0, y0 + i, PI)); // right: left
    }
    v
}

fn sorted(mut v: Vec<[f32; 2]>) -> Vec<[f32; 2]> {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v
}

#[test]
fn c1_bins() {
    assert_eq!(bin_of(0.0, 4), 0, "0° is bin 0");
    assert_eq!(bin_of(FRAC_PI_2, 4), 1, "90° (down) is bin 1");
    assert_eq!(bin_of(-FRAC_PI_2, 4), 3, "−90° (up) is bin 3");
    assert_eq!(bin_of(PI, 4), 2, "180° is bin 2");
    assert_eq!(bin_of(-0.05, 36), 0, "−0.05 rad (−2.9°) is closest to 0°");
    assert_eq!(bin_of(-0.1, 36), 35, "−0.1 rad (−5.7°) is closest to −10° = 350°");
    assert_eq!(bin_of(356f32.to_radians(), 36), 0, "356° rounds to 360° = bin 0");
}

#[test]
fn c1_r_table_b1a() {
    let t = build_r_table(&square(0.0, 0.0), [2.0, 2.0], 4);
    assert_eq!(t.len(), 4);
    assert_eq!(sorted(t[1].clone()), vec![[-1.0, 2.0], [0.0, 2.0], [1.0, 2.0]], "B1(a): bin 'down' (the top side)");
    assert_eq!(sorted(t[3].clone()), vec![[-1.0, -2.0], [0.0, -2.0], [1.0, -2.0]], "B1(a): bin 'up' (the bottom side)");
    assert_eq!(sorted(t[0].clone()), vec![[2.0, -1.0], [2.0, 0.0], [2.0, 1.0]], "B1(a): bin 'right' (the left side)");
    assert_eq!(sorted(t[2].clone()), vec![[-2.0, -1.0], [-2.0, 0.0], [-2.0, 1.0]], "B1(a): bin 'left' (the right side)");
}

#[test]
fn c2_votes_b1bc() {
    let t = build_r_table(&square(0.0, 0.0), [2.0, 2.0], 4);
    let copy = square(10.0, 5.0);
    // B1(b): the top side alone
    let top: Vec<EdgePoint> = copy.iter().copied().filter(|e| e.y == 5.0 && e.theta == FRAC_PI_2).collect();
    let acc = vote(&t, &top, 20, 15);
    let row: Vec<f32> = (10..=14).map(|x| acc[(x, 7)]).collect();
    assert_eq!(row, vec![1.0, 2.0, 3.0, 2.0, 1.0], "B1(b): votes of the top side at (10..14, 7)");
    // B1(c): all 12 points
    let acc = vote(&t, &copy, 20, 15);
    assert_eq!(acc.sum(), 36.0, "B1(c): 12 points × 3 vectors = 36 votes");
    assert_eq!(acc[(12, 7)], 12.0, "B1(c): the centre (12, 7) gets 12 votes");
    let mut others: Vec<f32> = acc.coords().filter(|&p| p != (12, 7)).map(|p| acc[p]).collect();
    others.sort_by(|a, b| b.total_cmp(a));
    assert_eq!(others[0], 4.0, "B1(c): the best competitor gets 4 votes");
}

/// An L-shaped set of edge points (not symmetric), with gradient directions.
fn ell(cx: f32, cy: f32, phi: f32, scale: f32) -> Vec<EdgePoint> {
    let mut base = Vec::new();
    for i in 0..12 {
        base.push(([i as f32 - 6.0, -6.0], FRAC_PI_2)); // a long side, gradient down
    }
    for i in 0..6 {
        base.push(([-6.0, i as f32 - 6.0], 0.0)); // a short side, gradient right
    }
    let (c, s) = (phi.cos(), phi.sin());
    base.iter()
        .map(|&([x, y], th)| ep(cx + scale * (c * x - s * y), cy + scale * (s * x + c * y), th + phi))
        .collect()
}

#[test]
fn c3_rotation_and_scale_b5() {
    let table = build_r_table(&ell(0.0, 0.0, 0.0, 1.0), [0.0, 0.0], 36);
    let n = 18.0;
    // rotated by 90°: missed without a rotation search, found with the right pose
    let rotated = ell(30.0, 30.0, FRAC_PI_2, 1.0);
    let plain = vote(&table, &rotated, 60, 60);
    let posed = vote_pose(&table, &rotated, 60, 60, FRAC_PI_2, 1.0);
    assert_eq!(posed[(30, 30)], n, "pose φ = 90°: all 18 points vote for the reference point (30, 30)");
    assert!(plain.max() <= 12.0, "without the rotation the votes do not meet: largest bin {}", plain.max());
    // scaled by 1.5 and rotated by 30°
    let a = 30f32.to_radians();
    let copy = ell(30.0, 30.0, a, 1.5);
    let posed = vote_pose(&table, &copy, 60, 60, a, 1.5);
    assert_eq!(posed[(30, 30)], n, "pose φ = 30°, s = 1.5: all 18 points at (30, 30), got {}", posed[(30, 30)]);
}
