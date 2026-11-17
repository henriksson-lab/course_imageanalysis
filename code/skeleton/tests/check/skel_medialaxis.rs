// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p skeleton skel_medialaxis::
//
// The exercise writes pixels as (row, column); the code indexes img[(x, y)] = img[(column, row)].

use imgcourse::Image;
use skeleton::skel_medialaxis::*;

/// The object pixels as (row, column), sorted.
fn pixels(img: &Image<bool>) -> Vec<(usize, usize)> {
    let mut v: Vec<(usize, usize)> = img.coords().filter(|&p| img[p]).map(|(x, y)| (y, x)).collect();
    v.sort();
    v
}

/// The 5 × 9 rectangle of B1 (rows 1–5, columns 1–9 of a 7 × 11 image).
fn rectangle() -> Image<bool> {
    Image::from_fn(11, 7, |x, y| (1..=9).contains(&x) && (1..=5).contains(&y))
}

/// A 5 × 5 square in a 7 × 7 image (B2).
fn square() -> Image<bool> {
    Image::from_fn(7, 7, |x, y| (1..=5).contains(&x) && (1..=5).contains(&y))
}

#[test]
fn c1_distance_transform_b1() {
    let dt = distance_transform(&rectangle(), Metric::CityBlock);
    let expected = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 2, 2, 2, 2, 2, 2, 2, 1],
        [1, 2, 3, 3, 3, 3, 3, 2, 1],
        [1, 2, 2, 2, 2, 2, 2, 2, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
    ];
    for (r, row) in expected.iter().enumerate() {
        for (c, &e) in row.iter().enumerate() {
            let got = dt[(c + 1, r + 1)];
            assert_eq!(got, e, "B1 city-block DT at (row, column) = ({}, {}): got {got}, expected {e}", r + 1, c + 1);
        }
    }
    assert_eq!(dt[(0, 0)], 0, "the background has DT 0");
    let dt = distance_transform(&square(), Metric::Chessboard);
    assert!(dt[(3, 3)] == 3 && dt[(2, 2)] == 2 && dt[(1, 3)] == 1, "chessboard DT of the 5 × 5 square: centre 3, inner ring 2, outer ring 1");
}

#[test]
fn c1_object_touching_the_image_edge() {
    let full = Image::new(5, 3, true); // outside the image counts as background
    let dt = distance_transform(&full, Metric::CityBlock);
    assert_eq!(dt.row(1), &[1, 2, 2, 2, 1], "a full 5 × 3 image: middle row");
}

#[test]
fn c2_medial_axis_city_block_b1() {
    let rect = rectangle();
    let axis = medial_axis(&distance_transform(&rect, Metric::CityBlock), Metric::CityBlock);
    let mut expected = vec![(1, 1), (1, 9), (5, 1), (5, 9), (2, 2), (2, 8), (4, 2), (4, 8)];
    expected.extend((3..=7).map(|c| (3, c)));
    expected.sort();
    assert_eq!(pixels(&axis), expected, "B1(b) the 13 maximal-disc centres, as (row, column)");
}

#[test]
fn c2_medial_axis_chessboard_b1() {
    let axis = medial_axis(&distance_transform(&rectangle(), Metric::Chessboard), Metric::Chessboard);
    let expected: Vec<(usize, usize)> = (3..=7).map(|c| (3, c)).collect();
    assert_eq!(pixels(&axis), expected, "B1(d) chessboard: only the middle row (3, 3) to (3, 7)");
}

#[test]
fn c3_reconstruct_b1() {
    for metric in [Metric::CityBlock, Metric::Chessboard] {
        let dt = distance_transform(&rectangle(), metric);
        let back = reconstruct(&medial_axis(&dt, metric), &dt, metric);
        assert_eq!(pixels(&back), pixels(&rectangle()), "{metric:?}: axis and radii must give back the rectangle");
    }
}

#[test]
fn c3_reconstruct_b3() {
    // Skeleton points with chessboard radii, (row, column) = (4, 4) r 3, (4, 5) r 3, (4, 6) r 2, (4, 7) r 1.
    let mut axis = Image::new(10, 9, false);
    let mut dt = Image::new(10, 9, 0u32);
    for (c, r) in [(4, 3), (5, 3), (6, 2), (7, 1)] {
        axis[(c, 4)] = true;
        dt[(c, 4)] = r;
    }
    let shape = reconstruct(&axis, &dt, Metric::Chessboard);
    let expected: Vec<(usize, usize)> = (2..=6).flat_map(|r| (2..=7).map(move |c| (r, c))).collect();
    assert_eq!(pixels(&shape), expected, "B3(a) the union of the four squares: rows 2–6, columns 2–7 (30 pixels)");
}

#[test]
fn c4_lantuejoul_square_b2() {
    let s = lantuejoul(&square(), false);
    let centre: Vec<(usize, usize)> = s.coords().filter(|&p| s[p] != 0).map(|(x, y)| (y, x)).collect();
    assert_eq!(centre, [(3, 3)], "B2(a) with the square: S0 and S1 empty, S2 = the centre");
    assert_eq!(s[(3, 3)], 3, "B2(a) the centre is in S2 (stored as r + 1 = 3)");
}

#[test]
fn c4_lantuejoul_cross_b2() {
    let s = lantuejoul(&square(), true);
    let in_s = |r: u32| -> Vec<(usize, usize)> {
        let mut v: Vec<(usize, usize)> = s.coords().filter(|&p| s[p] == r + 1).map(|(x, y)| (y, x)).collect();
        v.sort();
        v
    };
    assert_eq!(in_s(0), [(1, 1), (1, 5), (5, 1), (5, 5)], "B2(b) S0 = the 4 corners");
    assert_eq!(in_s(1), [(2, 2), (2, 4), (4, 2), (4, 4)], "B2(b) S1 = the 4 diagonal pixels");
    assert_eq!(in_s(2), [(3, 3)], "B2(b) S2 = the centre");
}

#[test]
fn c4_lantuejoul_is_the_maximal_disc_test_b7() {
    // With the cross, S_r = pixels with DT = r + 1 that are maximal-disc centres (B7).
    let f = imgcourse::samples::letter_f(24).map(|v| v > 0.5);
    let s = lantuejoul(&f, true);
    let dt = distance_transform(&f, Metric::CityBlock);
    let axis = medial_axis(&dt, Metric::CityBlock);
    for (x, y) in f.coords() {
        let expected = if axis[(x, y)] { dt[(x, y)] } else { 0 };
        assert_eq!(s[(x, y)], expected, "B7 letter F, pixel (x, y) = ({x}, {y}): Lantuéjoul and the maximal-disc test differ");
    }
}

#[test]
fn c5_reconstruct_lantuejoul_b2() {
    for cross in [false, true] {
        let back = reconstruct_lantuejoul(&lantuejoul(&square(), cross), cross);
        assert_eq!(pixels(&back), pixels(&square()), "B2(c) cross = {cross}: the square must come back");
    }
    let f = imgcourse::samples::letter_f(24).map(|v| v > 0.5);
    assert_eq!(reconstruct_lantuejoul(&lantuejoul(&f, true), true), f, "the letter F must come back");
}
