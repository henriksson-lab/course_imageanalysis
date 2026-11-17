// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p day22 integral_image::

use imgcourse::{samples, Image};
use day22::integral_image::*;

/// The 5 × 5 image of the exercise. The exercise counts rows and columns from 1, the code
/// from 0: pixel "column 3, row 3" is (x, y) = (2, 2). Table entries keep their numbers:
/// S(3, 3) of the exercise is `table[(3, 3)]`.
fn exercise_image() -> Image<f32> {
    Image::from_rows(&[
        [2., 0., 1., 3., 1.],
        [1., 4., 2., 0., 2.],
        [3., 1., 5., 2., 0.],
        [0., 2., 1., 4., 3.],
        [1., 3., 0., 2., 5.],
    ])
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-9
}

#[test]
fn c1_running_total_b1() {
    let s = running_total(&[3., 1., 4., 1., 5., 9., 2., 6.]);
    let expected = [0., 3., 4., 8., 9., 14., 23., 25., 31.];
    assert_eq!(s.len(), 9, "the running total of 8 pixels has 9 entries (with S(0) = 0)");
    for (i, (g, e)) in s.iter().zip(expected).enumerate() {
        assert!(close(*g, e), "S({i}): got {g}, expected {e}");
    }
}

#[test]
fn c1_range_sums_b1() {
    let s = running_total(&[3., 1., 4., 1., 5., 9., 2., 6.]);
    // pixels 3-7, 2-5 and 1-8 of the exercise are a..=b = 2..=6, 1..=4, 0..=7 in the code
    for (a, b, e) in [(2, 6, 21.), (1, 4, 11.), (0, 7, 31.)] {
        let g = range_sum(&s, a, b);
        assert!(close(g, e), "pixels {}-{} (counted from 1): got {g}, expected {e}", a + 1, b + 1);
    }
}

#[test]
fn c2_table_b2() {
    let s = integral_image(&exercise_image());
    assert_eq!((s.width(), s.height()), (6, 6), "the table of a 5 × 5 image is 6 × 6");
    let expected = [
        [0., 0., 0., 0., 0., 0.],
        [0., 2., 2., 3., 6., 7.],
        [0., 3., 7., 10., 13., 16.],
        [0., 6., 11., 19., 24., 27.],
        [0., 6., 13., 22., 31., 37.],
        [0., 7., 17., 26., 37., 48.],
    ];
    for (y, row) in expected.iter().enumerate() {
        for (x, &e) in row.iter().enumerate() {
            assert!(close(s[(x, y)], e), "table entry S({x}, {y}): got {}, expected {e}", s[(x, y)]);
        }
    }
}

#[test]
fn c3_rectangles_b2_b3() {
    let s = integral_image(&exercise_image());
    // (columns, rows) counted from 1 in the exercise; from 0 in the code
    let cases = [
        ("B2(b), columns 2-4, rows 2-4", (1, 1, 3, 3), 21.),
        ("B2(c), columns 3-5, rows 2-5", (2, 1, 4, 4), 26.),
        ("B2(d), the pixel I(3, 3)", (2, 2, 2, 2), 5.),
        ("B3, columns 1-3, rows 1-2", (0, 0, 2, 1), 10.),
        ("the whole image", (0, 0, 4, 4), 48.),
    ];
    for (name, (x0, y0, x1, y1), e) in cases {
        let g = rect_sum(&s, x0, y0, x1, y1);
        assert!(close(g, e), "{name}: got {g}, expected {e}");
    }
}

#[test]
fn c3_every_rectangle_equals_direct_sum() {
    let img = samples::cell().crop(60, 80, 12, 9);
    let s = integral_image(&img);
    for y0 in 0..9 {
        for y1 in y0..9 {
            for x0 in 0..12 {
                for x1 in x0..12 {
                    let mut direct = 0.0;
                    for y in y0..=y1 {
                        for x in x0..=x1 {
                            direct += img[(x, y)] as f64;
                        }
                    }
                    let g = rect_sum(&s, x0, y0, x1, y1);
                    assert!((g - direct).abs() < 1e-6, "columns {x0}-{x1}, rows {y0}-{y1}: got {g}, expected {direct}");
                }
            }
        }
    }
}

#[test]
fn c4_two_passes_give_the_same_table_b8() {
    let img = samples::cell().crop(10, 20, 17, 13);
    let a = integral_image(&img);
    let b = integral_two_pass(&img);
    assert_eq!((b.width(), b.height()), (18, 14), "the table must be one larger than the image");
    for (x, y) in a.coords() {
        assert!((a[(x, y)] - b[(x, y)]).abs() < 1e-6, "entry ({x}, {y}): two passes {}, recursion {}", b[(x, y)], a[(x, y)]);
    }
}

#[test]
fn c5_window_sums_agree() {
    let img = samples::cell().crop(0, 0, 30, 25);
    for k in [1, 3, 7] {
        let a = window_sums_direct(&img, k);
        let b = window_sums_table(&img, k);
        assert_eq!((a.width(), a.height()), (31 - k, 26 - k), "k = {k}: wrong number of windows");
        for (x, y) in a.coords() {
            assert!((a[(x, y)] - b[(x, y)]).abs() < 1e-6, "k = {k}, window at ({x}, {y}): direct {}, table {}", a[(x, y)], b[(x, y)]);
        }
    }
}
