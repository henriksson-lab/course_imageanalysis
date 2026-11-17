// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p radon algebraic_recon::

use radon::algebraic_recon::tomo::{ray_matrix, Row};
use radon::common::{angles, sinogram};
use radon::algebraic_recon::*;
use imgcourse::Image;

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

fn assert_vec(got: &[f32], expected: &[f32], what: &str) {
    assert!(got.iter().zip(expected).all(|(g, e)| close(*g, *e, 1e-3)), "{what}: got {got:?}, expected {expected:?}");
}

/// The 2 × 2 scan of B1: x = (a, b, c, d), rays row 1, row 2, column 1, column 2.
fn rows_and_columns() -> Vec<Row> {
    vec![vec![(0, 1.0), (1, 1.0)], vec![(2, 1.0), (3, 1.0)], vec![(0, 1.0), (2, 1.0)], vec![(1, 1.0), (3, 1.0)]]
}

#[test]
fn c1_forward_b1() {
    let a = rows_and_columns();
    assert_vec(&forward(&a, &[1.0, 2.0, 3.0, 4.0]), &[3.0, 7.0, 4.0, 6.0], "B1(a): data of [1 2; 3 4]");
    assert_vec(&forward(&a, &[2.0, 1.0, 2.0, 5.0]), &[3.0, 7.0, 4.0, 6.0], "B1(b): data of [2 1; 2 5]");
    assert_vec(&forward(&a, &[1.0, -1.0, -1.0, 1.0]), &[0.0; 4], "B1(b): the null image is invisible");
    let mut with_diagonal = a.clone();
    with_diagonal.push(vec![(0, 1.0), (3, 1.0)]);
    assert_vec(&forward(&with_diagonal, &[1.0, -1.0, -1.0, 1.0])[4..], &[2.0], "B2(b): the diagonal ray a + d sees the null image");
}

#[test]
fn c1_back_b4() {
    let a = rows_and_columns();
    assert_vec(&back(&a, &[3.0, 7.0, 4.0, 6.0], 4), &[7.0, 9.0, 11.0, 13.0], "B4(b): Aᵀb");
}

#[test]
fn c1_back_is_the_transpose() {
    // ⟨A x, y⟩ = ⟨x, Aᵀ y⟩ for the projector of a small image
    let n = 9;
    let a = ray_matrix(n, &angles(5));
    let x: Vec<f32> = (0..n * n).map(|i| ((i * 7) % 11) as f32).collect();
    let y: Vec<f32> = (0..a.len()).map(|i| ((i * 5) % 7) as f32 - 3.0).collect();
    let left: f32 = forward(&a, &x).iter().zip(&y).map(|(p, q)| p * q).sum();
    let right: f32 = x.iter().zip(back(&a, &y, n * n)).map(|(p, q)| p * q).sum();
    assert!(close(left, right, 1e-2 * left.abs().max(1.0)), "⟨Ax, y⟩ = {left} but ⟨x, Aᵀy⟩ = {right}");
}

#[test]
fn c2_art_by_hand_b3() {
    let a = rows_and_columns();
    let b = [3.0, 7.0, 4.0, 6.0];
    // step by step: one ray at a time
    let expected = [[1.5, 1.5, 0.0, 0.0], [1.5, 1.5, 3.5, 3.5], [1.0, 1.5, 3.0, 3.5], [1.0, 2.0, 3.0, 4.0]];
    let mut x = vec![0.0; 4];
    for i in 0..4 {
        art_sweep(&a[i..i + 1], &b[i..i + 1], &mut x, 1.0);
        assert_vec(&x, &expected[i], &format!("B3(a), after step {}", i + 1));
    }
    let mut x = vec![0.0; 4];
    art_sweep(&a, &b, &mut x, 1.0);
    assert_vec(&x, &[1.0, 2.0, 3.0, 4.0], "B3(b): one sweep from 0 (the smallest image)");
}

#[test]
fn c2_art_two_pixels_b5_b6() {
    let a = vec![vec![(0, 1.0), (1, 1.0)], vec![(0, 1.0), (1, -1.0)]];
    let mut x = vec![0.0; 2];
    art_sweep(&a, &[5.0, -1.0], &mut x, 1.0);
    assert_vec(&x, &[2.0, 3.0], "B5(a): perpendicular lines, one sweep");
    let a = vec![vec![(0, 1.0), (1, 1.0)], vec![(0, 1.0), (1, 1.2)]];
    let mut x = vec![0.0; 2];
    art_sweep(&a, &[5.0, 5.6], &mut x, 1.0);
    assert_vec(&x, &[2.541, 2.549], "B5(b): nearly parallel lines, one sweep");
    // B6: three inconsistent lines x2 = 4, x1 = 2, x1 + x2 = 5.5: ART goes round in a cycle
    let a = vec![vec![(1, 1.0)], vec![(0, 1.0)], vec![(0, 1.0), (1, 1.0)]];
    let mut x = vec![1.75, 3.75];
    art_sweep(&a, &[4.0, 2.0, 5.5], &mut x, 1.0);
    assert_vec(&x, &[1.75, 3.75], "B6(b): one sweep from (1.75, 3.75) comes back to the start");
}

#[test]
fn c3_sirt_by_hand_b4() {
    let a = rows_and_columns();
    let b = [3.0, 7.0, 4.0, 6.0];
    let mut x = vec![0.0; 4];
    sirt_step(&a, &b, &mut x, 1.0);
    assert_vec(&x, &[1.75, 2.25, 2.75, 3.25], "B4(a): one SIRT step from 0");
    for _ in 0..200 {
        sirt_step(&a, &b, &mut x, 1.0);
    }
    assert_vec(&forward(&a, &x), &b, "after many SIRT steps the data fit");
}

#[test]
fn c3_sirt_reconstructs_a_small_image() {
    let n = 12;
    let img = Image::from_fn(n, n, |x, y| if (3..9).contains(&x) && (4..8).contains(&y) { 1.0 } else { 0.0 });
    let thetas = angles(18);
    let a = ray_matrix(n, &thetas);
    let sino = sinogram(&img, &thetas);
    let b: Vec<f32> = (0..thetas.len()).flat_map(|t| (0..n).map(move |k| (t, k))).map(|(t, k)| sino[(t, k)]).collect();
    assert_vec(&forward(&a, img.pixels()), &b, "the ray matrix gives the same numbers as the projector");
    let mut x = vec![0.0; n * n];
    for _ in 0..300 {
        sirt_step(&a, &b, &mut x, 1.0);
    }
    let err = x.iter().zip(img.pixels()).map(|(p, q)| (p - q).powi(2)).sum::<f32>() / (n * n) as f32;
    assert!(err.sqrt() < 0.1, "SIRT, 300 steps on a 12 × 12 block image: RMSE {:.3}, expected below 0.1", err.sqrt());
}
