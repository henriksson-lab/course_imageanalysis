// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p day49 stereo::

use imgcourse::Image;
use day49::stereo::*;

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

#[test]
fn c1_sad_b2() {
    // one row each; rows outside the images read 0 in both, so they add nothing
    let left = Image::from_rows(&[[1., 1., 5., 9., 5., 1., 1., 1.]]);
    let right = Image::from_rows(&[[5., 9., 5., 1., 1., 1., 1., 1.]]);
    let costs = sad_cost_volume(&left, &right, 2, 1);
    let got = [costs[0][(3, 0)], costs[1][(3, 0)], costs[2][(3, 0)]];
    assert!(close(got[0], 12.0, 1e-5) && close(got[1], 12.0, 1e-5) && close(got[2], 0.0, 1e-5), "SAD at position 3 for d = 0, 1, 2: got {got:?}, expected (12, 12, 0)");
    // a constant row: every disparity costs the same (B2 b)
    let flat = Image::new(8, 1, 4.0f32);
    let costs = sad_cost_volume(&flat, &flat, 2, 1);
    let got = [costs[0][(4, 0)], costs[1][(4, 0)], costs[2][(4, 0)]];
    assert!(got.iter().all(|&c| close(c, 0.0, 1e-5)), "a constant row: got {got:?}, expected all 0");
}

#[test]
fn c2_census_b3() {
    let w1 = Image::from_rows(&[[10., 20., 30.], [40., 25., 15.], [5., 35., 50.]]);
    let w2 = Image::from_rows(&[[12., 30., 28.], [41., 26., 14.], [6., 33., 49.]]);
    let (c1, c2) = (census(&w1)[(1, 1)], census(&w2)[(1, 1)]);
    assert_eq!(c1, 0b11001100, "census of W1: got {c1:08b}, expected 11001100");
    assert_eq!(c2, 0b10001100, "census of W2: got {c2:08b}, expected 10001100");
    assert_eq!(hamming(c1, c2), 1, "Hamming distance: got {}, expected 1", hamming(c1, c2));
    let w3 = w1.map(|v| 2.0 * v + 10.0);
    assert_eq!(census(&w3)[(1, 1)], c1, "2 W1 + 10 must have the same code as W1 (B3 b)");
    assert_eq!(hamming(0b1011, 0b0110), 3, "hamming(1011, 0110)");
}

#[test]
fn c3_winner_take_all_b2() {
    let left = Image::from_rows(&[[1., 1., 5., 9., 5., 1., 1., 1.]]);
    let right = Image::from_rows(&[[5., 9., 5., 1., 1., 1., 1., 1.]]);
    let d = winner_take_all(&sad_cost_volume(&left, &right, 2, 1));
    assert_eq!(d[(3, 0)], 2.0, "disparity at position 3: got {}, expected 2", d[(3, 0)]);
    let costs = vec![Image::from_rows(&[[3.0f32, 1.0]]), Image::from_rows(&[[2.0, 1.0]]), Image::from_rows(&[[5.0, 0.5]])];
    let d = winner_take_all(&costs);
    assert_eq!((d[(0, 0)], d[(1, 0)]), (1.0, 2.0), "the lowest cost per pixel");
}

/// The cost slice of B5, `costs[x][d]`.
fn b5() -> Vec<Vec<f32>> {
    let by_d = [[5., 4., 6., 1., 5.], [2., 3., 1., 2., 2.], [6., 2., 4., 6., 6.], [4., 5., 5., 4., 7.]];
    (0..5).map(|x| (0..4).map(|d| by_d[d][x]).collect()).collect()
}

#[test]
fn c4_scanline_dp_b5() {
    let path = scanline_dp(&b5(), 1.0, 2.0);
    assert_eq!(path, [1, 1, 1, 1, 1], "P1 = 1, P2 = 2: got {path:?}, expected the straight profile d = 1 (energy 10)");
    let path = scanline_dp(&b5(), 0.25, 2.0);
    assert_eq!(path, [1, 2, 1, 0, 1], "P1 = 0.25: got {path:?}, expected the winner-take-all profile (energy 9)");
    let path = scanline_dp(&b5(), 0.0, 0.0);
    assert_eq!(path, [1, 2, 1, 0, 1], "P1 = P2 = 0 must give winner-take-all: got {path:?}");
}

#[test]
fn c5_subpixel_b6() {
    let o = subpixel_offset(4.0, 1.0, 2.0);
    assert!(close(o, 0.25, 1e-6), "costs 4, 1, 2: got {o}, expected +0.25");
    let o = subpixel_offset(2.0, 1.0, 4.0);
    assert!(close(o, -0.25, 1e-6), "costs 2, 1, 4: got {o}, expected −0.25");
    // exact for a parabola with its minimum at 0.3
    let c = |d: f32| (d - 0.3) * (d - 0.3);
    let o = subpixel_offset(c(-1.0), c(0.0), c(1.0));
    assert!(close(o, 0.3, 1e-5), "a parabola with its minimum at 0.3: got {o}");
}
