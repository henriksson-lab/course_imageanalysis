// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p ws-distance
// The exercise writes pixels as (row, column); the code uses (x, y) = (column, row).

use imgcourse::Image;
use ws_distance::*;

fn b1() -> Image<bool> {
    let rows = ["011101110", "111111111", "111111111", "111111111", "011101110"];
    Image::from_fn(9, 5, |x, y| rows[y].as_bytes()[x] == b'1')
}

/// Two discs of radius r, centres d apart, on a grid with room around them.
fn two_discs(r: f32, d: f32) -> Image<bool> {
    let w = (2.0 * r + d) as usize + 10;
    let h = 2 * r as usize + 10;
    let (cy, c1) = (h as f32 / 2.0, 5.0 + r);
    let c2 = c1 + d;
    Image::from_fn(w, h, |x, y| {
        let (x, y) = (x as f32, y as f32);
        (x - c1).powi(2) + (y - cy).powi(2) <= r * r || (x - c2).powi(2) + (y - cy).powi(2) <= r * r
    })
}

#[test]
fn c1_city_block_b1() {
    let dt = distance_city_block(&b1());
    let expected = Image::from_rows(&[
        [0., 1., 1., 1., 0., 1., 1., 1., 0.],
        [1., 2., 2., 2., 1., 2., 2., 2., 1.],
        [1., 2., 3., 3., 2., 3., 3., 2., 1.],
        [1., 2., 2., 2., 1., 2., 2., 2., 1.],
        [0., 1., 1., 1., 0., 1., 1., 1., 0.],
    ]);
    for (x, y) in dt.coords() {
        assert_eq!(dt[(x, y)], expected[(x, y)], "B1 (a): row {y}, column {x}\ngot:\n{dt}expected:\n{expected}");
    }
}

#[test]
fn c2_chamfer() {
    // A 7 × 7 block: the centre is 4 steps of 3 from the outside (12/3 = 4); a pixel whose
    // nearest background is one diagonal step away gets 4/3.
    let block = Image::new(7, 7, true);
    let dt = distance_chamfer34(&block);
    assert!((dt[(3, 3)] - 4.0).abs() < 1e-5, "centre of a 7 × 7 block: got {}, expected 4", dt[(3, 3)]);
    let mut m = Image::new(7, 7, true);
    m[(2, 2)] = false;
    let d = distance_chamfer34(&m);
    assert!((d[(3, 3)] - 4.0 / 3.0).abs() < 1e-5, "a diagonal step: got {}, expected 4/3", d[(3, 3)]);
    assert!((d[(3, 2)] - 1.0).abs() < 1e-5, "an edge step: got {}, expected 1", d[(3, 2)]);
}

#[test]
fn c3_hmaxima_b1() {
    let mask = b1();
    let dt = distance_city_block(&mask);
    let (_, n) = regional_maxima(&h_maxima(&dt, 0.5), &mask);
    assert_eq!(n, 2, "B1 (d): h = 0.5 keeps both peaks (depth 1); got {n} markers");
    let (_, n) = regional_maxima(&h_maxima(&dt, 1.0), &mask);
    assert_eq!(n, 1, "B1 (d): h = 1 lowers both peaks to the saddle: one marker; got {n}");
    let m = h_maxima(&dt, 1.0);
    assert!(m.pixels().iter().zip(dt.pixels()).all(|(a, b)| a <= b), "HMAX must never go above dt");
}

#[test]
fn c4_split_b1() {
    let mask = b1();
    let dt = distance_city_block(&mask);
    let (labels, n) = split(&dt, &mask, 0.5, true);
    assert_eq!(n, 2, "B1 (c): two objects");
    let line: Vec<(usize, usize)> = labels.coords().filter(|&p| mask[p] && labels[p] == 0).map(|(x, y)| (y, x)).collect();
    assert_eq!(line, [(1, 4), (2, 4), (3, 4)], "B1 (c): the line is the neck column (row, column)");
    assert!(labels[(1, 2)] != labels[(7, 2)] && labels[(1, 2)] > 0 && labels[(7, 2)] > 0, "left and right of the line: two labels");
    assert_eq!(labels[(0, 0)], 0, "the background stays 0");
}

#[test]
fn c4_two_discs_b2() {
    // r = 10, d = 16: neck depth 10 − 6 = 4 (B2 (b)). Split for small h, merged for large h.
    let mask = two_discs(10.0, 16.0);
    let dt = distance_euclidean(&mask);
    assert_eq!(split(&dt, &mask, 1.0, false).1, 2, "r = 10, d = 16, h = 1: two objects");
    assert_eq!(split(&dt, &mask, 6.0, false).1, 1, "r = 10, d = 16, h = 6: one object");
    // r = 10, d = 6: depth 0.46 (B2 (b)): merged already at h = 1
    let mask = two_discs(10.0, 6.0);
    let dt = distance_euclidean(&mask);
    assert_eq!(split(&dt, &mask, 1.0, false).1, 1, "r = 10, d = 6, h = 1: one object");
}
