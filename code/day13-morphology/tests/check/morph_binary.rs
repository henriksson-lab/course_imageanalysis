// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p day13 morph_binary::

use imgcourse::Image;
use day13::morph_binary::*;

fn mask<const W: usize>(rows: &[[u8; W]]) -> Image<bool> {
    Image::from_rows(rows).map(|v| v == 1)
}

fn count(a: &Image<bool>) -> usize {
    a.pixels().iter().filter(|&&b| b).count()
}

/// Prints a mask as 0/1 rows, for the failure messages.
fn show(a: &Image<bool>) -> String {
    format!("\n{}", a.map(|b| b as u8))
}

/// The 7 × 7 image A of the exercise (17 pixels).
fn a() -> Image<bool> {
    mask(&[
        [0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 0, 0],
        [0, 1, 1, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ])
}

#[test]
fn c1_erosion_b1() {
    let got = erode(&a(), L, false);
    let expected = mask(&[
        [0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ]);
    assert_eq!(got, expected, "B1: A eroded by the L (7 pixels). Got:{}", show(&got));
}

#[test]
fn c2_dilation_b2() {
    let got = dilate(&a(), L, false);
    let expected = mask(&[
        [0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 1, 1, 0],
    ]);
    assert_eq!(got, expected, "B2: A dilated by the L (28 pixels). Got:{}", show(&got));
}

#[test]
fn c2_single_pixel_b3() {
    let mut p = Image::new(5, 5, false);
    p[(2, 2)] = true;
    let got = dilate(&p, L, false);
    let expected = mask(&[[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 1, 1, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0]]);
    assert_eq!(got, expected, "B3: a single pixel dilates to a copy of the L. Got:{}", show(&got));
}

#[test]
fn c1_c2_duality_b4() {
    // (A ⊖ B)ᶜ = Aᶜ ⊕ B̌, with the outside 0 for A and 1 for Aᶜ
    let lhs = complement(&erode(&a(), L, false));
    let rhs = dilate(&complement(&a()), &reflect(L), true);
    assert_eq!(count(&rhs), 42, "B4: Aᶜ dilated by the reflected L has 42 pixels. Got:{}", show(&rhs));
    assert_eq!(lhs, rhs, "B4: duality");
}

#[test]
fn c1_c2_duality_random() {
    let mut rng = imgcourse::Rng::new(1);
    let img = Image::from_fn(20, 15, |_, _| rng.uniform() < 0.6);
    for se in [CROSS, SQUARE, L] {
        let lhs = complement(&erode(&img, se, false));
        let rhs = dilate(&complement(&img), &reflect(se), true);
        assert_eq!(lhs, rhs, "duality on a random image, SE {se:?}");
    }
}

/// The disc-like object D of B5 (21 pixels).
fn d() -> Image<bool> {
    mask(&[
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ])
}

#[test]
fn c3_boundaries_b5() {
    let cross = inner_boundary(&d(), CROSS);
    let square = inner_boundary(&d(), SQUARE);
    assert_eq!(count(&cross), 12, "B5: the boundary with the cross has 12 pixels. Got:{}", show(&cross));
    assert_eq!(count(&square), 16, "B5: the boundary with the square has 16 pixels. Got:{}", show(&square));
    // with the square, the corner pixel at row 2, column 2 is boundary; with the cross it is not
    assert!(square[(2, 2)] && !cross[(2, 2)], "B5: row 2, column 2");
}

#[test]
fn c1_c2_origin_outside_b6() {
    // B' = {(1, 0)}: erosion shifts A one pixel to the left, dilation one to the right
    let b = [(1isize, 0isize)];
    let left = erode(&a(), &b, false);
    let right = dilate(&a(), &b, false);
    for (x, y) in a().coords() {
        let a_right = x + 1 < 7 && a()[(x + 1, y)];
        let a_left = x >= 1 && a()[(x - 1, y)];
        assert_eq!(left[(x, y)], a_right, "B6: erosion by {{(1, 0)}} at ({x}, {y})");
        assert_eq!(right[(x, y)], a_left, "B6: dilation by {{(1, 0)}} at ({x}, {y})");
    }
}

#[test]
fn c1_border_b7() {
    let block = Image::from_fn(6, 6, |x, y| x < 4 && y < 4);
    let zero = erode(&block, SQUARE, false);
    let one = erode(&block, SQUARE, true);
    assert_eq!(count(&zero), 4, "B7 (a): outside = 0 leaves 4 pixels. Got:{}", show(&zero));
    assert_eq!(count(&one), 9, "B7 (b): outside = 1 leaves 9 pixels. Got:{}", show(&one));
}

#[test]
fn c2_row_then_column_is_square() {
    let row = [(-1isize, 0isize), (0, 0), (1, 0)];
    let column = [(0isize, -1isize), (0, 0), (0, 1)];
    let mut rng = imgcourse::Rng::new(2);
    let img = Image::from_fn(16, 12, |_, _| rng.uniform() < 0.15);
    let two = dilate(&dilate(&img, &row, false), &column, false);
    assert_eq!(two, dilate(&img, SQUARE, false), "a row of 3 then a column of 3 = the 3 × 3 square");
}
