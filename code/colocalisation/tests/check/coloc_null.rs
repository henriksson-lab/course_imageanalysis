// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p colocalisation coloc_null::

use colocalisation::common::{self, Scene};
use colocalisation::coloc_null::*;
use imgcourse::{Image, Rng};

fn close(got: f32, expected: f32, tol: f32) -> bool {
    (got - expected).abs() < tol
}

#[test]
fn c1_shift_curve_finds_a_shift() {
    let (a, _, cells) = common::scene(&Scene::default());
    // B is A moved 2 pixels to the right: B(x, y) = A(x − 2, y)
    let b = Image::from_fn(a.width(), a.height(), |x, y| if x >= 2 { a[(x - 2, y)] } else { a[(0, y)] });
    let curve = shift_curve(&a, &b, &cells, 5);
    assert_eq!(curve.len(), 11, "shifts −5 … 5 give 11 values");
    let best = (0..11).max_by(|&i, &j| curve[i].total_cmp(&curve[j])).unwrap() as isize - 5;
    assert_eq!(best, 2, "the peak should be at dx = +2, got {best}; curve {curve:?}");
    assert!(close(curve[7], 1.0, 1e-3), "at dx = +2 the images match: got r = {}, expected 1", curve[7]);
    assert!(curve[5] < 0.9, "at dx = 0 the images do not match: got r = {}", curve[5]);
}

#[test]
fn c2_subpixel_peak_b10() {
    // B3: r = 0.78, 0.90, 0.84 at −1, 0, 1
    let x = subpixel_peak(0.78, 0.90, 0.84);
    assert!(close(x, 1.0 / 6.0, 1e-3), "B10: got {x}, expected +0.17");
    let x = subpixel_peak(0.5, 1.0, 0.5);
    assert!(close(x, 0.0, 1e-6), "a symmetric peak: got {x}, expected 0");
    // a parabola with its vertex at 0.3: r(x) = 1 − (x − 0.3)²
    let r = |x: f32| 1.0 - (x - 0.3) * (x - 0.3);
    let x = subpixel_peak(r(-1.0), r(0.0), r(1.0));
    assert!(close(x, 0.3, 1e-5), "vertex at 0.3: got {x}");
}

#[test]
fn c3_block_shuffle_keeps_the_pieces() {
    let (_, b, cells) = common::scene(&Scene::default());
    let mut rng = Rng::new(7);
    let s = block_shuffle(&b, &cells, 8, &mut rng);
    assert_ne!(s, b, "the shuffled image should differ from the original");
    // the same values, in another order
    let mut v1: Vec<f32> = b.pixels().to_vec();
    let mut v2: Vec<f32> = s.pixels().to_vec();
    v1.sort_by(f32::total_cmp);
    v2.sort_by(f32::total_cmp);
    assert_eq!(v1, v2, "shuffling must keep every pixel value, only move them");
    // the corner square (centre outside the cells) stays where it is
    for y in 0..8 {
        for x in 0..8 {
            assert_eq!(s[(x, y)], b[(x, y)], "pixel ({x}, {y}) is in a square outside the cells and should not move");
        }
    }
    // every 8 × 8 square of the result is a whole square of the original
    let square = |img: &Image<f32>, x0: usize, y0: usize| img.crop(x0, y0, 8, 8);
    let originals: Vec<Image<f32>> = (0..16).flat_map(|by| (0..16).map(move |bx| (bx * 8, by * 8))).map(|(x, y)| square(&b, x, y)).collect();
    for by in 0..16 {
        for bx in 0..16 {
            let piece = square(&s, bx * 8, by * 8);
            assert!(originals.contains(&piece), "the square at ({}, {}) is not a square of the original: blocks were cut", bx * 8, by * 8);
        }
    }
}

#[test]
fn c4_permutation_p_b1() {
    let null = [-0.08, 0.03, -0.11, 0.05, 0.14, -0.02, 0.07, -0.05, 0.01, -0.09, 0.10, -0.03, 0.04, -0.06, 0.02, -0.01, 0.08, -0.04, 0.00];
    let p = permutation_p(0.12, &null);
    assert!(close(p, 0.10, 1e-6), "B1(a): got p = {p}, expected 0.10");
    let p = permutation_p(0.5, &null);
    assert!(close(p, 0.05, 1e-6), "B1(b), no shuffle reaches it: got p = {p}, expected 1/20");
    let p = permutation_p(0.14, &null);
    assert!(close(p, 0.10, 1e-6), "a tie counts as 'at least as large': got p = {p}, expected 0.10");
}

#[test]
fn c5_pdm_and_icq_b2() {
    let a = Image::from_rows(&[[0., 2., 8.], [1., 6., 9.], [0., 1., 3.]]);
    let b = Image::from_rows(&[[1., 1., 6.], [0., 5., 8.], [1., 2., 0.]]);
    let all = common::everywhere(3, 3);
    let d = pdm(&a, &b, &all);
    let expected = [5.56, 2.22, 15.56, 6.22, 6.22, 30.22, 5.56, 1.56, 0.89];
    for (k, (x, y)) in d.coords().enumerate() {
        assert!(close(d[(x, y)], expected[k], 0.01), "B2(c): PDM at row {}, column {}: got {:.2}, expected {}", y + 1, x + 1, d[(x, y)], expected[k]);
    }
    assert!(close(d.sum(), 74.0, 0.01), "B2(c): the PDM values should sum to 74, got {}", d.sum());
    let q = icq(&a, &b, &all);
    assert!(close(q, 0.5, 1e-6), "B2(b): got ICQ {q}, expected 0.5");
    // B upside down: some pixels now vary in opposite directions
    let flipped = Image::from_fn(3, 3, |x, y| b[(x, 2 - y)]);
    let q = icq(&a, &flipped, &all);
    assert!(close(q, 7.0 / 9.0 - 0.5, 1e-6), "B flipped: got ICQ {q}, expected 7/9 − 0.5");
}
