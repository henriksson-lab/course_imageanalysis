// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p day13 morph_reconstruct::
// The 1D signals are images of height 1; the grey levels are those of the exercises.

use imgcourse::Image;
use day13::morph_reconstruct::*;

/// The signal of the exercises (samples 0–11).
fn signal() -> Image<f32> {
    Image::from_rows(&[[7., 1., 4., 3., 4., 6., 5., 1., 6., 4., 2., 3.]])
}

fn check(name: &str, got: &Image<f32>, expected: &[f32]) {
    let g = got.pixels().to_vec();
    let ok = g.len() == expected.len() && g.iter().zip(expected).all(|(a, b)| (a - b).abs() < 1e-5);
    assert!(ok, "{name}:\n  got      {g:?}\n  expected {expected:?}");
}

/// The samples where a 1D mask is true.
fn where_true(a: &Image<bool>) -> Vec<usize> {
    (0..a.width()).filter(|&x| a[(x, 0)]).collect()
}

#[test]
fn k1_h_maxima_by_hand_b1() {
    let i = signal();
    let (r, iterations) = reconstruct(&i.map(|v| v - 2.0), &i, true);
    check("B1 (b): reconstruction of I − 2", &r, &[5., 1., 3., 3., 4., 4., 4., 1., 4., 4., 2., 2.]);
    assert_eq!(iterations, 3, "B1 (b): the number of geodesic dilations that change something");
}

#[test]
fn k1_single_seed_b3() {
    let mut m = Image::new(12, 1, 0.0);
    m[(5, 0)] = 6.0;
    let (r, iterations) = reconstruct(&m, &signal(), true);
    check("B3: one seed at sample 5", &r, &[1., 1., 3., 3., 4., 6., 5., 1., 1., 1., 1., 1.]);
    assert_eq!(iterations, 6, "B3: the number of iterations");
}

#[test]
fn k2_h_minima_by_hand_b2() {
    let i = signal();
    let (r, iterations) = reconstruct_by_erosion(&i.map(|v| v + 2.0), &i, true);
    check("B2: reconstruction by erosion of I + 2", &r, &[7., 3., 4., 4., 4., 6., 5., 3., 6., 4., 4., 4.]);
    assert_eq!(iterations, 3, "B2: the number of geodesic erosions that change something");
}

#[test]
fn k3_h_transforms_b1_b2() {
    check("B1: h_maxima(I, 2)", &h_maxima(&signal(), 2.0), &[5., 1., 3., 3., 4., 4., 4., 1., 4., 4., 2., 2.]);
    check("B2: h_minima(I, 2)", &h_minima(&signal(), 2.0), &[7., 3., 4., 4., 4., 6., 5., 3., 6., 4., 4., 4.]);
}

#[test]
fn k3_regional_extrema_b1_b2() {
    let i = signal();
    assert_eq!(where_true(&regional_maxima(&i, 1.0)), vec![0, 2, 5, 8, 11], "B1 (c): the regional maxima of I");
    assert_eq!(where_true(&regional_maxima(&h_maxima(&i, 2.0), 1.0)), vec![0, 4, 5, 6, 8, 9], "B1 (c): the regional maxima after h = 2 (sample 0 and the plateaus 4–6 and 8–9)");
    assert_eq!(where_true(&regional_minima(&i, 1.0)), vec![1, 3, 7, 10], "B2: the regional minima of I");
    assert_eq!(where_true(&regional_minima(&h_minima(&i, 2.0), 1.0)), vec![1, 7, 9, 10, 11], "B2: the regional minima after h = 2");
}

fn mask<const W: usize>(rows: &[[u8; W]]) -> Image<bool> {
    Image::from_rows(rows).map(|v| v == 1)
}

/// The 7 × 7 mask of B5.
fn b5() -> Image<bool> {
    mask(&[
        [1, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0],
        [0, 0, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ])
}

fn show(a: &Image<bool>) -> String {
    format!("\n{}", a.map(|b| b as u8))
}

#[test]
fn k4_fill_holes_b5() {
    let got = fill_holes(&b5());
    let mut expected = b5();
    expected[(4, 4)] = true; // the hole inside the ring
    assert_eq!(got, expected, "B5 (a): only row 4, column 4 is a hole. Got:{}", show(&got));
}

#[test]
fn k4_clear_border_b5() {
    let got = clear_border(&b5());
    let mut expected = b5();
    for (x, y) in [(0, 0), (1, 0), (0, 1), (1, 1), (2, 1), (1, 2)] {
        expected[(x, y)] = false;
    }
    assert_eq!(got, expected, "B5 (b): the 6 pixels of the top-left object go; the ring and the single pixel stay. Got:{}", show(&got));
}

#[test]
fn k4_hole_closed_by_a_diagonal_b10() {
    // B10: a ring closed only through corners is still an 8-connected object around its centre
    let ring = mask(&[[0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 1, 0, 1, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0]]);
    assert!(fill_holes(&ring)[(2, 2)], "B10: the centre of the diamond ring is a hole (background is 4-connected)");
}

#[test]
fn k5_impose_minima_b4() {
    let mut markers = Image::new(12, 1, false);
    markers[(3, 0)] = true;
    markers[(10, 0)] = true;
    let got = impose_minima(&signal(), &markers);
    check("B4 (b): minima imposed at samples 3 and 10", &got, &[8., 5., 5., 0., 5., 7., 7., 7., 7., 5., 0., 4.]);
    assert_eq!(where_true(&regional_minima(&got, 1.0)), vec![3, 10], "B4 (b): the only regional minima are the markers");
}
