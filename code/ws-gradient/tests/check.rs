// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p ws-gradient

use imgcourse::Image;
use ws_gradient::*;

fn b1_image() -> Image<f32> {
    let mut img = Image::from_rows(&[[2., 2., 2., 8., 8., 8.]; 6]);
    img[(1, 1)] = 3.0; // the noisy pixel at row 1, column 1
    img
}

fn b1_gradient() -> Image<f32> {
    Image::from_rows(&[
        [2., 2., 24., 24., 0., 0.],
        [2., 0., 22., 24., 0., 0.],
        [2., 2., 24., 24., 0., 0.],
        [0., 0., 24., 24., 0., 0.],
        [0., 0., 24., 24., 0., 0.],
        [0., 0., 24., 24., 0., 0.],
    ])
}

const B4: [f32; 11] = [2., 0., 3., 1., 6., 0., 4., 1., 5., 0., 2.];

fn row(v: &[f32]) -> Image<f32> {
    Image::from_vec(v.len(), 1, v.to_vec())
}

fn count_minima(g: &Image<f32>) -> u32 {
    regional_minima(g).1
}

#[test]
fn c1_gradient_b1() {
    let g = gradient_l1(&b1_image());
    let expected = b1_gradient();
    for (x, y) in g.coords() {
        assert!((g[(x, y)] - expected[(x, y)]).abs() < 1e-4, "B1: row {y}, column {x}: got {}, expected {}", g[(x, y)], expected[(x, y)]);
    }
}

#[test]
fn c2_hminima_b1() {
    let g = b1_gradient();
    assert_eq!(count_minima(&g), 3, "B1 (b): 3 regional minima (this uses the given regional_minima)");
    assert_eq!(count_minima(&h_minima(&g, 1.0)), 3, "B1 (c): with h = 1 the noisy pixel's basin (depth 2) survives");
    assert_eq!(count_minima(&h_minima(&g, 2.0)), 2, "B1 (c): with h = 2 it is filled: 2 regions");
}

#[test]
fn c2_hminima_b4() {
    let g = row(&B4);
    let counts: Vec<u32> = [0.0, 2.0, 3.0, 5.0, 6.0].iter().map(|&h| count_minima(&h_minima(&g, h))).collect();
    assert_eq!(counts, [5, 4, 3, 2, 1], "B4 (c): minima left for h = 0, 2, 3, 5, 6");
    let m = h_minima(&g, 2.0);
    assert!(m.pixels().iter().zip(g.pixels()).all(|(a, b)| a >= b), "HMIN must never go below g");
}

#[test]
fn c3_dams_b4() {
    let g = row(&B4);
    let labels = watershed(&g, &regional_minima(&g).0);
    assert_eq!(labels.pixels(), [1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5], "B4 (a): the five basins (this uses the given watershed)");
    assert_eq!(dam_heights(&labels, &g), [(1, 2, 3.), (2, 3, 6.), (3, 4, 4.), (4, 5, 5.)], "B4 (a): dams (a, b, height)");
}

#[test]
fn c3_lowest_dam_wins() {
    // Two regions that touch in two places: the lower pass counts.
    let labels = Image::from_rows(&[[1, 2], [1, 2], [1, 2]]);
    let g = Image::from_rows(&[[5., 7.], [1., 3.], [9., 9.]]);
    assert_eq!(dam_heights(&labels, &g), [(1, 2, 3.)], "the lowest dam between 1 and 2 is max(1, 3) = 3");
}

#[test]
fn c4_merge_b4() {
    let g = row(&B4);
    let labels = watershed(&g, &regional_minima(&g).0);
    let dams = dam_heights(&labels, &g);
    let counts: Vec<u32> = [2.0, 3.0, 4.0, 5.0, 6.0].iter().map(|&t| merge_below(&labels, &dams, t).1).collect();
    assert_eq!(counts, [5, 4, 3, 2, 1], "B4 (b): regions after merging across dams ≤ 2, 3, 4, 5, 6");
    let (merged, _) = merge_below(&labels, &dams, 4.0);
    assert_eq!(merged.pixels(), [1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3], "B4 (b): after the dams 3 and 4");
}
