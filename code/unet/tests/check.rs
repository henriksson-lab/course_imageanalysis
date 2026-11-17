// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p unet

use imgcourse::Image;
use unet::*;

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

/// The probability map and the true mask of B1.
fn b1() -> (Image<f32>, Image<bool>) {
    let p = Image::from_rows(&[[0.9, 0.8, 0.1], [0.7, 0.6, 0.2], [0.1, 0.2, 0.1]]);
    let g = Image::from_rows(&[[true, true, false], [true, true, false], [false, false, false]]);
    (p, g)
}

fn assert_row(got: &Image<f32>, expected: &[f32], what: &str) {
    let row: Vec<f32> = (0..got.width()).map(|x| got[(x, 0)]).collect();
    assert!(row.len() == expected.len() && row.iter().zip(expected).all(|(a, b)| close(*a, *b, 1e-5)), "{what}: got {row:?}, expected {expected:?}");
}

#[test]
fn c1_cross_entropy_b1a() {
    let (p, g) = b1();
    let ce = pixel_cross_entropy(&p, &g);
    assert!(close(ce, 0.218, 1e-3), "B1a: got {ce}, expected 0.218");
}

#[test]
fn c1_cross_entropy_b1c() {
    // 20 × 20, one nucleus of 20 pixels, p = 0.01 everywhere.
    let g = Image::from_fn(20, 20, |x, y| y == 5 && x < 20);
    let p = Image::new(20, 20, 0.01);
    let ce = pixel_cross_entropy(&p, &g);
    assert!(close(ce, 0.240, 1e-3), "B1c: got {ce}, expected 0.240");
}

#[test]
fn c2_soft_dice_b1b() {
    let (p, g) = b1();
    let d = soft_dice(&p, &g);
    assert!(close(d, 6.0 / 7.7, 1e-4), "B1b, soft Dice: got {d}, expected 0.779");
    let mask = p.map(|v| if v > 0.5 { 1.0 } else { 0.0 });
    let d = soft_dice(&mask, &g);
    assert!(close(d, 1.0, 1e-6), "B1b, Dice of the mask at 0.5: got {d}, expected 1");
}

#[test]
fn c2_soft_dice_b1c() {
    let g = Image::from_fn(20, 20, |x, y| y == 5 && x < 20);
    let d = soft_dice(&Image::new(20, 20, 0.01), &g);
    assert!(close(d, 0.4 / 24.0, 1e-4), "B1c, soft Dice: got {d}, expected 0.017");
}

#[test]
fn c3_dice_gradient_b7() {
    let (p, g) = b1();
    let grad = soft_dice_gradient(&p, &g);
    assert!(close(grad[(0, 0)], 0.159, 1e-3), "B7, an object pixel: got {}, expected 0.159", grad[(0, 0)]);
    assert!(close(grad[(2, 2)], -0.101, 1e-3), "B7, a background pixel: got {}, expected -0.101", grad[(2, 2)]);
    // compare with a finite difference at one pixel
    let eps = 1e-3;
    let mut plus = p.clone();
    plus[(1, 1)] += eps;
    let numeric = (soft_dice(&plus, &g) - soft_dice(&p, &g)) / eps;
    assert!(close(grad[(1, 1)], numeric, 2e-3), "finite difference at (1, 1): {numeric}, your gradient {}", grad[(1, 1)]);
}

#[test]
fn c4_upsample_nearest_b2a() {
    let up = upsample_nearest(&Image::from_rows(&[[1.0, 2.0, 3.0]]), 2);
    assert!(up.width() == 6 && up.height() == 2, "a 3 × 1 image upsampled by 2 should be 6 × 2, got {} × {}", up.width(), up.height());
    assert_row(&up, &[1., 1., 2., 2., 3., 3.], "B2a");
}

#[test]
fn c5_transposed_conv_b2c() {
    let x = Image::from_rows(&[[1.0, 2.0, 3.0]]);
    let out = transposed_conv(&x, &Image::from_rows(&[[0.5, 1.0, 0.5]]), 2);
    assert_row(&out, &[0.5, 1., 1.5, 2., 2.5, 3., 1.5], "B2c, kernel (1/2, 1, 1/2), stride 2");
    let out = transposed_conv(&x, &Image::from_rows(&[[1.0, 1.0]]), 2);
    assert_row(&out, &[1., 1., 2., 2., 3., 3.], "B2d, kernel (1, 1), stride 2 = nearest neighbour");
}

#[test]
fn c5_transposed_conv_2d() {
    // A 2 × 2 image and a 2 × 2 kernel of ones with stride 2: nearest-neighbour upsampling.
    let x = Image::from_rows(&[[1.0, 2.0], [3.0, 4.0]]);
    let out = transposed_conv(&x, &Image::new(2, 2, 1.0), 2);
    assert_eq!(out, upsample_nearest(&x, 2), "a kernel of ones with stride 2 should equal upsample_nearest");
}
