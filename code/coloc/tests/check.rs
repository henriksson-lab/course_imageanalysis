// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p coloc

use coloc::sim::{self, Scene};
use coloc::*;
use imgcourse::Image;

/// The two 3 × 3 channels of B1 (the region is all 9 pixels).
fn b1() -> (Image<f32>, Image<f32>, Image<bool>) {
    let a = Image::from_rows(&[[0., 2., 8.], [1., 6., 9.], [0., 1., 3.]]);
    let b = Image::from_rows(&[[1., 1., 6.], [0., 5., 8.], [1., 2., 0.]]);
    (a, b, sim::everywhere(3, 3))
}

fn close(got: f32, expected: f32, tol: f32) -> bool {
    (got - expected).abs() < tol
}

#[test]
fn c1_pearson_b1() {
    let (a, b, all) = b1();
    let r = pearson(&a, &b, &all);
    assert!(close(r, 0.916, 1e-3), "B1(b): got r = {r}, expected 0.916");
    // B7: a gain and an offset of A change nothing; a negative gain flips the sign
    let r2 = pearson(&a.map(|v| 3.0 * v + 7.0), &b, &all);
    assert!(close(r2, r, 1e-4), "r of 3A + 7: got {r2}, expected {r}");
    let r3 = pearson(&a.map(|v| -v), &b, &all);
    assert!(close(r3, -r, 1e-4), "r of −A: got {r3}, expected {}", -r);
}

#[test]
fn c1_pearson_b4_and_outlier_b5() {
    let a = Image::from_rows(&[[0., 0., 0.], [0., 9., 0.], [0., 6., 0.]]);
    let b = Image::from_rows(&[[0., 5., 5.], [0., 5., 5.], [0., 5., 5.]]);
    let r = pearson(&a, &b, &sim::everywhere(3, 3));
    assert!(close(r, 0.37, 0.01), "B4: got r = {r}, expected 0.37");
    let (mut a, b, all) = b1();
    a[(2, 1)] = 90.0; // the 9 in A becomes 90
    let r = pearson(&a, &b, &all);
    assert!(close(r, 0.742, 1e-3), "B5(c): got r = {r}, expected 0.742");
}

#[test]
fn c1_only_the_mask_counts() {
    let (a, b, _) = b1();
    // only the top row: A = 0, 2, 8 and B = 1, 1, 6
    let top = Image::from_fn(3, 3, |_, y| y == 0);
    let r = pearson(&a, &b, &top);
    assert!(close(r, 0.9707, 1e-3), "r over the top row only: got {r}, expected 0.971");
}

#[test]
fn c2_overlap_b1_b3_b6() {
    let (a, b, all) = b1();
    let (ov, k1, k2) = overlap(&a, &b, &all);
    assert!(close(ov, 0.957, 1e-3), "B1(c): got overlap {ov}, expected 0.957");
    assert!(close(k1, 0.786, 1e-3) && close(k2, 1.167, 1e-3), "B6: got k1 = {k1}, k2 = {k2}, expected 0.786 and 1.167");
    let (ov, _, _) = overlap(&a, &b.map(|v| v + 10.0), &all);
    assert!(close(ov, 0.834, 1e-3), "B3(a), B + 10: got overlap {ov}, expected 0.834");
    let (_, k1, k2) = overlap(&a, &b.map(|v| 2.0 * v), &all);
    assert!(close(k1, 1.571, 1e-3) && close(k2, 0.583, 1e-3), "B6, B doubled: got k1 = {k1}, k2 = {k2}, expected 1.571 and 0.583");
}

#[test]
fn c3_manders_b2_b3_b4() {
    let (a, b, all) = b1();
    let (m1, m2) = manders(&a, &b, &all, 2.0, 2.0);
    assert!(close(m1, 23.0 / 26.0, 1e-4) && close(m2, 1.0, 1e-4), "B2(b): got M1 = {m1}, M2 = {m2}, expected 0.885 and 1");
    let (m1, m2) = manders(&a, &b, &all, -1.0, -1.0);
    assert!(close(m1, 1.0, 1e-4) && close(m2, 1.0, 1e-4), "B2(c), no thresholds: got {m1}, {m2}, expected 1 and 1");
    let b10 = b.map(|v| v + 10.0);
    let (m1, m2) = manders(&a, &b10, &all, 2.0, 2.0);
    assert!(close(m1, 1.0, 1e-4) && close(m2, 59.0 / 114.0, 1e-4), "B3(b): got M1 = {m1}, M2 = {m2}, expected 1 and 0.518");
    let (m1, m2) = manders(&a, &b10, &all, 2.0, 12.0);
    assert!(close(m1, 23.0 / 26.0, 1e-4) && close(m2, 1.0, 1e-4), "B3(c), t_B = 12: got {m1}, {m2}, expected 0.885 and 1");
    let a4 = Image::from_rows(&[[0., 0., 0.], [0., 9., 0.], [0., 6., 0.]]);
    let b4 = Image::from_rows(&[[0., 5., 5.], [0., 5., 5.], [0., 5., 5.]]);
    let (m1, m2) = manders(&a4, &b4, &all, 0.0, 0.0);
    assert!(close(m1, 1.0, 1e-4) && close(m2, 1.0 / 3.0, 1e-4), "B4: got M1 = {m1}, M2 = {m2}, expected 1 and 0.333");
}

#[test]
fn c4_ranks_b5() {
    // A of B1, row by row
    let r = ranks(&[0., 2., 8., 1., 6., 9., 0., 1., 3.]);
    let expected = [1.5, 5.0, 8.0, 3.5, 7.0, 9.0, 1.5, 3.5, 6.0];
    assert_eq!(r, expected, "B5: ranks of A, got {r:?}, expected {expected:?}");
    let r = ranks(&[1., 1., 6., 0., 5., 8., 1., 2., 0.]);
    let expected = [4.0, 4.0, 8.0, 1.5, 7.0, 9.0, 4.0, 6.0, 1.5];
    assert_eq!(r, expected, "B5: ranks of B, got {r:?}, expected {expected:?}");
}

#[test]
fn c4_spearman_b5() {
    let (mut a, b, all) = b1();
    let s = spearman(&a, &b, &all);
    assert!(close(s, 0.622, 1e-3), "B5(b): got {s}, expected 0.622");
    a[(2, 1)] = 90.0;
    let s = spearman(&a, &b, &all);
    assert!(close(s, 0.622, 1e-3), "B5(c), 9 → 90: got {s}, expected still 0.622");
}

#[test]
fn c5_orthogonal_line() {
    // points on B = 2A + 1 (no noise): the line itself
    let a = Image::from_rows(&[[0., 1., 2.], [3., 4., 5.]]);
    let b = a.map(|v| 2.0 * v + 1.0);
    let (slope, icpt) = orthogonal_line(&a, &b, &sim::everywhere(3, 2));
    assert!(close(slope, 2.0, 1e-4) && close(icpt, 1.0, 1e-4), "B = 2A + 1: got slope {slope}, intercept {icpt}");
    // a cloud symmetric about the diagonal: slope 1
    // (0, 1), (1, 0), (2, 3), (3, 2): ordinary least squares of B on A would give 0.6
    let a = Image::from_rows(&[[0., 1., 2., 3.]]);
    let b = Image::from_rows(&[[1., 0., 3., 2.]]);
    let (slope, _) = orthogonal_line(&a, &b, &sim::everywhere(4, 1));
    assert!(close(slope, 1.0, 1e-4), "symmetric cloud: got slope {slope}, expected 1");
}

#[test]
fn c5_costes() {
    let (a, b, cells) = sim::scene(&Scene::default());
    let (ta, tb) = costes(&a, &b, &cells, 200).expect("colocalised channels: Costes must be defined");
    // the scene has offset 0.04 and noise 0.015: the thresholds stop just above the background
    assert!(ta > 0.04 && ta < 0.15 && tb > 0.04 && tb < 0.15, "got t_A = {ta}, t_B = {tb}, expected just above the background 0.04");
    let below = Image::from_fn(a.width(), a.height(), |x, y| cells[(x, y)] && (a[(x, y)] <= ta || b[(x, y)] <= tb));
    let r = pearson(&a, &b, &below);
    assert!(r <= 0.0, "r of the pixels below the thresholds: got {r}, expected ≤ 0");
    // anticorrelated channels: no rising line
    let neg = b.map(|v| 1.0 - v);
    assert!(costes(&a, &neg, &cells, 200).is_none(), "r < 0: Costes should give None");
}
