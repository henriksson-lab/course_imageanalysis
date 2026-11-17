// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p day13 morph_openclose::

use imgcourse::Image;
use day13::morph_openclose::*;

fn mask<const W: usize>(rows: &[[u8; W]]) -> Image<bool> {
    Image::from_rows(rows).map(|v| v == 1)
}

fn count(a: &Image<bool>) -> usize {
    a.pixels().iter().filter(|&&b| b).count()
}

fn show(a: &Image<bool>) -> String {
    format!("\n{}", a.map(|b| b as u8))
}

/// The 9 × 9 image A of B1 (20 pixels).
fn b1() -> Image<bool> {
    mask(&[
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 0, 1, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ])
}

#[test]
fn c1_opening_b1() {
    let got = open(&b1(), SQUARE);
    let expected = Image::from_fn(9, 9, |x, y| (2..=4).contains(&x) && (2..=5).contains(&y));
    assert_eq!(got, expected, "B1 (a): the opening is the block of rows 2–5, columns 2–4. Got:{}", show(&got));
}

#[test]
fn c1_closing_b1() {
    let got = close(&b1(), SQUARE);
    let mut expected = b1();
    expected[(5, 4)] = true; // the hole, row 4, column 5
    assert_eq!(got, expected, "B1 (b): the closing fills the hole and keeps the rest (21 pixels). Got:{}", show(&got));
}

#[test]
fn c1_idempotent_b1() {
    let once = open(&b1(), SQUARE);
    assert_eq!(open(&once, SQUARE), once, "B1 (c): opening twice changes nothing");
    let once = close(&b1(), SQUARE);
    assert_eq!(close(&once, SQUARE), once, "closing twice changes nothing");
}

#[test]
fn c1_order_matters_b2() {
    let oc = close(&open(&b1(), SQUARE), SQUARE);
    let co = open(&close(&b1(), SQUARE), SQUARE);
    assert_eq!((count(&oc), count(&co)), (12, 20), "B2: open-then-close 12 pixels, close-then-open 20");
}

#[test]
fn c2_granulometry_b4() {
    // runs of lengths 1, 2, 2, 4, 5, separated by 2 zeros
    let row = [0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0];
    let a = Image::from_rows(&[row]).map(|v| v == 1);
    let ses: Vec<_> = (1..=6).map(line).collect();
    let (area, spectrum) = granulometry(&a, &ses);
    assert_eq!(area, vec![14, 13, 9, 9, 5, 0], "B4 (a): area left for lines of length 1–6");
    assert_eq!(spectrum, vec![1, 4, 0, 4, 5], "B4 (b): pattern spectrum");
}

/// The 8 × 8 mask M of B5 (32 pixels).
fn m() -> Image<bool> {
    mask(&[
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 1, 1, 1, 1, 0],
        [0, 0, 0, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ])
}

#[test]
fn c3_city_block_b5() {
    let got = distance_transform(&m(), false);
    let expected = Image::from_rows(&[
        [0u32, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 2, 2, 2, 2, 1, 0],
        [0, 1, 2, 3, 3, 2, 1, 0],
        [0, 1, 1, 2, 3, 2, 1, 0],
        [0, 0, 0, 1, 2, 2, 1, 0],
        [0, 0, 0, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ]);
    assert_eq!(got, expected, "B5 (a): the city-block distance transform. Got:\n{got}");
}

#[test]
fn c3_chessboard_b6() {
    let got = distance_transform(&m(), true);
    let expected = Image::from_rows(&[
        [0u32, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 2, 2, 2, 2, 1, 0],
        [0, 1, 2, 2, 2, 2, 1, 0],
        [0, 1, 1, 1, 2, 2, 1, 0],
        [0, 0, 0, 1, 2, 2, 1, 0],
        [0, 0, 0, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ]);
    assert_eq!(got, expected, "B6: the chessboard distance transform. Got:\n{got}");
}

#[test]
fn c4_erosion_by_threshold_b5_b6() {
    assert_eq!(count(&erode_by_distance(&m(), 2, false)), 3, "B5 (b): DT > 2 has 3 pixels");
    assert_eq!(count(&erode_by_distance(&m(), 1, true)), 12, "B6: the erosion by the 3 × 3 square has 12 pixels");
}

#[test]
fn c4_threshold_equals_erosion() {
    let mut rng = imgcourse::Rng::new(3);
    let blobs = Image::from_fn(30, 20, |_, _| rng.uniform() < 0.8);
    for r in 0..4 {
        assert_eq!(erode_by_distance(&blobs, r, false), erode(&blobs, &diamond(r as usize)), "{{city-block DT > {r}}} = erosion by the diamond of radius {r}");
    }
    assert_eq!(erode_by_distance(&blobs, 1, true), erode(&blobs, SQUARE), "{{chessboard DT > 1}} = erosion by the 3 × 3 square");
}
