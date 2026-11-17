// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p images kernels::

use imgcourse::{samples, Border, Image};
use images::kernels::*;

/// The 5 × 5 image of Part A: 2 on the left, 8 on the right, a noisy 11 at row 2, column 2.
/// The exercise counts rows and columns from 1; the code counts from 0, so the noisy
/// pixel is at (x, y) = (1, 1).
fn part_a() -> Image<f32> {
    let mut img = Image::from_rows(&[[2., 2., 2., 8., 8.]; 5]);
    img[(1, 1)] = 11.0;
    img
}

fn close(a: f32, b: f32) -> bool {
    (a - b).abs() < 1e-4
}

#[test]
fn c1_box_blur_b1() {
    let out = filter3x3(&part_a(), &BOX, Border::Clamp);
    // row 3, columns 2, 3, 4  ->  y = 2, x = 1, 2, 3
    assert!(close(out[(1, 2)], 3.0), "row 3, column 2: got {}, expected 3", out[(1, 2)]);
    assert!(close(out[(2, 2)], 5.0), "row 3, column 3: got {}, expected 5", out[(2, 2)]);
    assert!(close(out[(3, 2)], 6.0), "row 3, column 4: got {}, expected 6", out[(3, 2)]);
    assert!(close(out[(1, 1)], 3.0), "the noisy pixel: got {}, expected 3", out[(1, 1)]);
}

#[test]
fn c1_gaussian_b2() {
    let out = filter3x3(&part_a(), &GAUSSIAN, Border::Clamp);
    assert!(close(out[(1, 1)], 4.25), "the noisy pixel: got {}, expected 4.25", out[(1, 1)]);
    assert!(close(out[(2, 2)], 65.0 / 16.0), "row 3, column 3: got {}, expected 65/16", out[(2, 2)]);
}

#[test]
fn c1_laplacian_b4() {
    let out = filter3x3(&part_a(), &LAPLACIAN, Border::Clamp);
    let got = [out[(1, 2)], out[(2, 2)], out[(3, 2)], out[(1, 1)]];
    let expected = [9.0, 6.0, -6.0, -36.0];
    for (g, e) in got.iter().zip(expected) {
        assert!(close(*g, e), "Laplacian: got {got:?}, expected {expected:?}");
    }
}

#[test]
fn c1_no_flip_sobel_x_b5() {
    // Without flipping, Sobel x is positive where the image gets brighter to the right.
    let out = filter3x3(&part_a(), &SOBEL_X, Border::Clamp);
    assert!(close(out[(2, 3)], 24.0), "row 4, column 3: got {}, expected 24", out[(2, 3)]);
    assert!(close(out[(2, 2)], 15.0), "row 3, column 3: got {}, expected 15", out[(2, 2)]);
}

#[test]
fn c1_border_b8() {
    // Row 5, column 1 (bottom left) is (x, y) = (0, 4).
    let zero = filter3x3(&part_a(), &BOX, Border::Zero);
    let clamp = filter3x3(&part_a(), &BOX, Border::Clamp);
    assert!(close(zero[(0, 4)], 8.0 / 9.0), "zeros outside: got {}, expected 8/9", zero[(0, 4)]);
    assert!(close(clamp[(0, 4)], 2.0), "copy the edge: got {}, expected 2", clamp[(0, 4)]);
}

#[test]
fn c2_gradient_of_a_step() {
    let step = samples::step(8, 8, 0.0, 1.0); // columns 0-3 are 0, columns 4-7 are 1
    let g = gradient_magnitude(&step, Border::Clamp);
    assert!(close(g[(3, 4)], 4.0) && close(g[(4, 4)], 4.0), "at the step: got {} and {}, expected 4", g[(3, 4)], g[(4, 4)]);
    assert!(close(g[(1, 4)], 0.0), "on the flat part: got {}, expected 0", g[(1, 4)]);
}

#[test]
fn c3_separable_equals_3x3() {
    let img = samples::cell().crop(100, 100, 20, 15);
    let (column, row) = ([1.0, 2.0, 1.0], [-1.0, 0.0, 1.0]); // Sobel x = (1 2 1) down times (-1 0 1) across
    for border in [Border::Zero, Border::Clamp, Border::Mirror, Border::Wrap] {
        let a = filter_separable(&img, column, row, border);
        let b = filter3x3(&img, &SOBEL_X, border);
        for (x, y) in img.coords() {
            assert!(close(a[(x, y)], b[(x, y)]), "{border:?}, pixel ({x}, {y}): separable {} but 3 × 3 {}", a[(x, y)], b[(x, y)]);
        }
    }
}
