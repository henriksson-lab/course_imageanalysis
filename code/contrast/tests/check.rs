// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p contrast

use contrast::*;
use imgcourse::{samples, Image};

/// The 64-pixel image of B5 (counts 6, 22, 18, 8, 5, 3, 1, 1 at levels 0 … 7), as one row.
fn b5_image() -> Image<u32> {
    let counts = [6, 22, 18, 8, 5, 3, 1, 1];
    let values: Vec<u32> = counts.iter().enumerate().flat_map(|(v, &c)| std::iter::repeat_n(v as u32, c)).collect();
    Image::from_vec(64, 1, values)
}

/// The micrograph as 8-bit whole numbers.
fn cell8() -> Image<u32> {
    samples::cell().map(|v| (255.0 * v).round() as u32)
}

#[test]
fn c1_gamma_and_log_b2_b3() {
    // B2 on 101 levels (0.00, 0.01, …, 1.00): γ = 0.5 maps 0.01, 0.04, 0.25, 0.81 to 0.1, 0.2, 0.5, 0.9.
    let lut = gamma_lut(101, 101, 0.5);
    for (v, s) in [(1, 10), (4, 20), (25, 50), (81, 90)] {
        assert_eq!(lut[v], s, "B2, γ = 0.5: {:.2} should map to {:.2}, got {:.2}", v as f32 / 100.0, s as f32 / 100.0, lut[v] as f32 / 100.0);
    }
    // B3: 12 bits to 8 bits with c = 255 / ln 4096.
    let lut = log_lut(4096, 256);
    for (v, s) in [(10, 74), (100, 141), (1000, 212), (4095, 255), (0, 0)] {
        assert_eq!(lut[v], s, "B3: log curve, raw {v}: got {}, expected {s}", lut[v]);
    }
    let img = Image::from_rows(&[[0u32, 10, 100], [1000, 4095, 10]]);
    let out = apply_lut(&img, &lut);
    assert_eq!(out.pixels(), &[0, 74, 141, 212, 255, 74], "apply_lut with the log curve");
}

#[test]
fn c2_equalise_b5() {
    let h = histogram(&b5_image(), 8);
    assert_eq!(h, [6, 22, 18, 8, 5, 3, 1, 1], "B5: the histogram");
    let lut = equalise_lut(&h);
    assert_eq!(lut, [1, 3, 5, 6, 6, 7, 7, 7], "B5: the equalisation mapping");
    let out = histogram(&apply_lut(&b5_image(), &lut), 8);
    assert_eq!(out, [0, 6, 0, 22, 0, 18, 13, 5], "B5: the output histogram");
    // B7(b): equalising twice changes nothing.
    let again = equalise_lut(&out);
    for k in [1, 3, 5, 6, 7] {
        assert_eq!(again[k], k as u32, "B7(b): the second equalisation should map {k} to itself, got {}", again[k]);
    }
}

#[test]
fn c2_gamma_first_changes_nothing_b7() {
    // B7(a) / B15: equalisation only uses the order of the values. A gamma on 12 bits keeps
    // the order (no two 8-bit levels merge), so equalising after it gives the same picture.
    let cell = cell8();
    let direct = apply_lut(&cell, &equalise_lut(&histogram(&cell, 256)));
    let wide = apply_lut(&cell, &gamma_lut(256, 4096, 0.5));
    let after = apply_lut(&wide, &equalise_lut(&histogram(&wide, 4096)));
    // The second result is on 4096 levels: compare the order of the pixels. Sorted by one
    // result, the other must never go down (and the reverse).
    let mut pairs: Vec<(u32, u32)> = direct.pixels().iter().copied().zip(after.pixels().iter().copied()).collect();
    pairs.sort();
    for w in pairs.windows(2) {
        assert!(w[1].1 >= w[0].1, "B7(a): equalising directly gives {} < {} but after γ = 0.5 {} > {}", w[0].0, w[1].0, w[0].1, w[1].1);
    }
}

#[test]
fn c3_match_b6() {
    let h = [6, 22, 18, 8, 5, 3, 1, 1];
    let target = [0, 0, 8, 8, 16, 16, 8, 8];
    let lut = match_lut(&h, &target);
    assert_eq!(lut, [2, 4, 5, 6, 7, 7, 7, 7], "B6: the matching curve");
    // Matching to a flat target is equalisation (up to rounding).
    let flat = [8; 8];
    let m = match_lut(&h, &flat);
    let e = equalise_lut(&h);
    for k in 0..8 {
        assert!(m[k].abs_diff(e[k]) <= 1, "match to flat at level {k}: {} but equalisation {}", m[k], e[k]);
    }
}

#[test]
fn c4_clip_histogram_b9() {
    let h = clip_histogram(&[1.0, 10.0, 4.0, 1.0], 1.5);
    let expected = [7.0 / 3.0, 6.0, 16.0 / 3.0, 7.0 / 3.0];
    for k in 0..4 {
        assert!((h[k] - expected[k]).abs() < 1e-3, "B9(c): after repeated clip-and-spread got {h:?}, expected about [2.33, 6, 5.33, 2.33]");
    }
    let sum: f32 = h.iter().sum();
    assert!((sum - 16.0).abs() < 1e-3, "B9: clipping must keep the number of pixels (16), got {sum}");
}

#[test]
fn c5_clahe_limits() {
    let cell = cell8();
    // One tile and no clipping: global equalisation.
    let global = apply_lut(&cell, &equalise_lut(&histogram(&cell, 256)));
    let one = clahe(&cell, 256, 1, 1000.0);
    let worst = one.pixels().iter().zip(global.pixels()).map(|(a, b)| a.abs_diff(*b)).max().unwrap();
    assert!(worst <= 1, "CLAHE with 1 tile and no clipping should be global equalisation; largest difference {worst}");
    // β = 1: every clipped histogram is flat, the curve is a straight line: nothing happens.
    let flat = clahe(&cell, 256, 8, 1.0);
    let worst = flat.pixels().iter().zip(cell.pixels()).map(|(a, b)| a.abs_diff(*b)).max().unwrap();
    assert!(worst <= 1, "CLAHE with β = 1 should leave the image (almost) unchanged; largest change {worst}");
}
