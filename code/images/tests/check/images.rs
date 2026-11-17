// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p images images::

use images::images::*;
use imgcourse::{samples, Image};
use std::f32::consts::FRAC_PI_2;

fn close(a: f32, b: f32) -> bool {
    (a - b).abs() < 1e-4
}

/// The four pixels of B4, written as I(x, y): I(0,0) = 10, I(1,0) = 20, I(0,1) = 30, I(1,1) = 60.
fn b4() -> Image<f32> {
    Image::from_rows(&[[10., 20.], [30., 60.]])
}

#[test]
fn c1_bilinear_b4() {
    let v = bilinear(&b4(), 0.25, 0.5);
    assert!(close(v, 25.0), "I(0.25, 0.5): got {v}, expected 25");
    assert!(close(bilinear(&b4(), 1.0, 0.0), 20.0), "on a pixel, bilinear must give that pixel");
}

#[test]
fn c1_nearest() {
    assert_eq!(nearest(&b4(), 0.8, 0.1), 20.0, "(0.8, 0.1) is closest to (1, 0)");
    assert_eq!(nearest(&b4(), 0.2, 0.9), 30.0, "(0.2, 0.9) is closest to (0, 1)");
    assert_eq!(nearest(&b4(), 5.0, 0.0), 0.0, "outside the image: 0");
}

#[test]
fn c2_rotate_about_b5_b7() {
    let m = rotate_about(2.0, 1.0, FRAC_PI_2);
    let expected = [[0., -1., 3.], [1., 0., -1.], [0., 0., 1.]];
    for i in 0..3 {
        for j in 0..3 {
            assert!(close(m[i][j], expected[i][j]), "rotate_about(2, 1, 90°): got {m:?}, expected {expected:?}");
        }
    }
    let (x, y) = apply(&m, (3.0, 1.0));
    assert!(close(x, 2.0) && close(y, 2.0), "(3, 1) should go to (2, 2), got ({x}, {y})");
    let (x, y) = apply(&m, (2.0, 1.0));
    assert!(close(x, 2.0) && close(y, 1.0), "the centre (2, 1) should stay, got ({x}, {y})");
}

#[test]
fn c2_order_matters_b6() {
    let stretch = [[2., 0., 0.], [0., 1., 0.], [0., 0., 1.]];
    let turn = rotation(FRAC_PI_2);
    let (x, y) = apply(&mat3_mul(&turn, &stretch), (1.0, 0.0)); // stretch first, then turn
    assert!(close(x, 0.0) && close(y, 2.0), "stretch then turn: (1, 0) -> (0, 2), got ({x}, {y})");
    let (x, y) = apply(&mat3_mul(&stretch, &turn), (1.0, 0.0)); // turn first, then stretch
    assert!(close(x, 0.0) && close(y, 1.0), "turn then stretch: (1, 0) -> (0, 1), got ({x}, {y})");
}

#[test]
fn c3_invert_affine() {
    let m = mat3_mul(&translation(5.0, -2.0), &[[2., 1., 0.], [0., 0.5, 0.], [0., 0., 1.]]);
    let inv = invert_affine(&m).expect("this transform is invertible");
    let back = mat3_mul(&inv, &m);
    for i in 0..3 {
        for j in 0..3 {
            assert!(close(back[i][j], IDENTITY[i][j]), "inverse times m should be the identity, got {back:?}");
        }
    }
    let collapse = [[1., 2., 0.], [0.5, 1., 0.], [0., 0., 1.]];
    assert!(invert_affine(&collapse).is_none(), "determinant 0: no inverse");
}

#[test]
fn c4_warp() {
    let img = samples::letter_f(40);
    // The identity changes nothing.
    assert_eq!(warp(&img, &IDENTITY, nearest), img, "warp with the identity must return the same image");
    // A shift by (3, 2): output pixel (x, y) comes from input (x - 3, y - 2).
    let shifted = warp(&img, &translation(3.0, 2.0), bilinear);
    for (x, y) in img.coords().filter(|&(x, y)| x >= 3 && y >= 2) {
        assert!(close(shifted[(x, y)], img[(x - 3, y - 2)]), "shift by (3, 2), pixel ({x}, {y})");
    }
    // A quarter turn about the centre of the image moves pixels exactly onto pixels.
    let c = 19.5;
    let turned = warp(&img, &rotate_about(c, c, FRAC_PI_2), nearest);
    for (x, y) in img.coords() {
        // turning (x, y) about c by +90° gives (c - (y - c), c + (x - c))
        let (tx, ty) = ((2.0 * c - y as f32) as usize, x);
        assert_eq!(turned[(tx, ty)], img[(x, y)], "quarter turn: input ({x}, {y}) should land at ({tx}, {ty})");
    }
}
