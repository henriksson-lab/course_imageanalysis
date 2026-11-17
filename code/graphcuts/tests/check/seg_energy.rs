// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p graphcuts seg_energy::

use imgcourse::{Image, Rng};
use graphcuts::seg_energy::*;

/// The 1 × 5 image of B1–B3 with D(ℓ) = (I − μℓ)²: μ0 = 0, μ1 = 3 and 2s² = 1.
fn b1() -> (Image<f32>, Image<f32>) {
    data_costs(&Image::from_rows(&[[3.0, 3.0, 1.0, 3.0, 0.0]]), 0.0, 3.0, 0.5f32.sqrt())
}

fn labels(s: &str) -> Image<u32> {
    Image::from_vec(s.len(), 1, s.chars().map(|c| c.to_digit(10).unwrap()).collect())
}

fn as_string(l: &[u32]) -> String {
    l.iter().map(|v| v.to_string()).collect()
}

#[test]
fn c1a_data_costs_b1() {
    let (d0, d1) = b1();
    let (e0, e1) = ([9.0, 9.0, 1.0, 9.0, 0.0], [0.0, 0.0, 4.0, 0.0, 9.0]);
    for x in 0..5 {
        assert!((d0[(x, 0)] - e0[x]).abs() < 1e-4, "B1: D(0) of pixel {}: got {}, expected {}", x + 1, d0[(x, 0)], e0[x]);
        assert!((d1[(x, 0)] - e1[x]).abs() < 1e-4, "B1: D(1) of pixel {}: got {}, expected {}", x + 1, d1[(x, 0)], e1[x]);
    }
}

#[test]
fn c1b_energy_b2() {
    let (d0, d1) = b1();
    for (l, e1, e5) in [("11010", 4.0, 16.0), ("11110", 5.0, 9.0), ("11111", 13.0, 13.0), ("00000", 28.0, 28.0)] {
        let (a, b) = (energy(&labels(l), &d0, &d1, 1.0), energy(&labels(l), &d0, &d1, 5.0));
        assert!((a - e1).abs() < 1e-3, "B2: E({l}) at λ = 1: got {a}, expected {e1}");
        assert!((b - e5).abs() < 1e-3, "B2: E({l}) at λ = 5: got {b}, expected {e5}");
    }
}

#[test]
fn c1b_energy_2d_b5() {
    // B5: a pixel of 120 among background pixels (0), model μ0 = 60, μ1 = 160, s = 30.
    let mut img = Image::new(3, 3, 0.0f32);
    img[(1, 1)] = 120.0;
    let (d0, d1) = data_costs(&img, 60.0, 160.0, 30.0);
    let background = Image::new(3, 3, 0u32);
    let mut speck = background.clone();
    speck[(1, 1)] = 1;
    // the speck is cheaper by 2 − 0.889 = 1.111 in data and costs 4 boundaries
    let diff = energy(&speck, &d0, &d1, 0.0) - energy(&background, &d0, &d1, 0.0);
    assert!((diff + 1.111).abs() < 1e-3, "B5: data difference {diff}, expected −1.111");
    let diff = energy(&speck, &d0, &d1, 1.0) - energy(&background, &d0, &d1, 1.0);
    assert!((diff - 2.889).abs() < 1e-3, "B5: with λ = 1 the speck costs 4 boundaries more: got {diff}, expected 2.889");
}

#[test]
fn c2_threshold_b1() {
    let (d0, d1) = b1();
    let t = threshold(&d0, &d1);
    assert_eq!(as_string(t.pixels()), "11010", "B1: thresholding");
}

#[test]
fn c3_chain_dp_b2_b3() {
    let (d0, d1) = b1();
    for (lambda, best, e) in [(1.0, "11010", 4.0), (2.0, "11110", 6.0), (5.0, "11110", 9.0), (10.0, "11111", 13.0)] {
        let (l, energy) = chain_dp(d0.pixels(), d1.pixels(), lambda);
        assert_eq!(as_string(&l), best, "B2/B3: the best labelling at λ = {lambda}");
        assert!((energy - e).abs() < 1e-3, "B2/B3: energy at λ = {lambda}: got {energy}, expected {e}");
    }
}

#[test]
fn c3_chain_dp_thin_structure_b4() {
    // B4: two samples of 140 and 221 between background (60); model 60 / 160, s = 30.
    let img = Image::from_rows(&[[60.0, 60.0, 140.0, 221.0, 60.0, 60.0]]);
    let (d0, d1) = data_costs(&img, 60.0, 160.0, 30.0);
    let (keep, _) = chain_dp(d0.pixels(), d1.pixels(), 7.7);
    assert_eq!(as_string(&keep), "001100", "B4: at λ = 7.7 (< 7.83) the structure is kept");
    let (drop, _) = chain_dp(d0.pixels(), d1.pixels(), 7.9);
    assert_eq!(as_string(&drop), "000000", "B4: at λ = 7.9 (> 7.83) the structure is dropped");
}

#[test]
fn c3_chain_dp_equals_trying_all() {
    let mut rng = Rng::new(4);
    for _ in 0..20 {
        let n = 10;
        let d0: Vec<f32> = (0..n).map(|_| 5.0 * rng.uniform()).collect();
        let d1: Vec<f32> = (0..n).map(|_| 5.0 * rng.uniform()).collect();
        let lambda = 2.0 * rng.uniform();
        let (_, e) = chain_dp(&d0, &d1, lambda);
        let mut best = f32::INFINITY;
        for bits in 0..(1u32 << n) {
            let l: Vec<u32> = (0..n).map(|k| (bits >> k) & 1).collect();
            let img = Image::from_vec(n, 1, l);
            let (a, b) = (Image::from_vec(n, 1, d0.clone()), Image::from_vec(n, 1, d1.clone()));
            best = best.min(energy(&img, &a, &b, lambda));
        }
        assert!((e - best).abs() < 1e-3, "dynamic programming gives {e}, but trying all 1024 labellings gives {best}");
    }
}

#[test]
fn c4_rows_dp() {
    let img = Image::from_rows(&[[3.0, 3.0, 1.0, 3.0, 0.0], [0.0, 1.0, 3.0, 0.0, 0.0]]);
    let (d0, d1) = data_costs(&img, 0.0, 3.0, 0.5f32.sqrt());
    let out = rows_dp(&d0, &d1, 5.0);
    assert_eq!(as_string(out.row(0)), "11110", "row 1 at λ = 5 (B2)");
    assert_eq!(as_string(out.row(1)), "00000", "row 2 at λ = 5: the lone bright pixel (data gain 9) costs 2 boundaries (10)");
}
