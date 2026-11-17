// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p measurements intensity_features::

use imgcourse::Image;
use measurements::intensity_features::*;

/// The 5 × 5 object of the exercise (B1), with x = column and y = row.
const BLOCK: [[f32; 5]; 5] = [
    [1., 2., 2., 2., 1.],
    [2., 3., 3., 4., 2.],
    [2., 3., 4., 8., 6.],
    [2., 3., 5., 9., 7.],
    [1., 2., 2., 3., 2.],
];

/// The block as a 5 × 5 image whose every pixel is object 1.
fn block() -> (Image<f32>, Image<u32>) {
    (Image::from_rows(&BLOCK), Image::new(5, 5, 1))
}

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

#[test]
fn c1_quantiles_b1() {
    let v: Vec<f32> = BLOCK.iter().flatten().copied().collect();
    let got = [quantile(&v, 0.25), quantile(&v, 0.5), quantile(&v, 0.75)];
    assert!(close(got[0], 2.0, 1e-5) && close(got[1], 2.0, 1e-5) && close(got[2], 4.0, 1e-5), "B1(b) quartiles and median: got {got:?}, expected [2, 2, 4]");
    // Interpolation between neighbours: 1 2 3 4, p = 0.5 -> position 1.5 -> 2.5
    let q = quantile(&[4.0, 1.0, 3.0, 2.0], 0.5);
    assert!(close(q, 2.5, 1e-5), "median of 1 2 3 4: got {q}, expected 2.5");
    assert!(close(quantile(&[4.0, 1.0, 3.0], 1.0), 4.0, 1e-5), "quantile 1 should be the maximum");
    let m = mad(&v);
    assert!(close(m, 1.0, 1e-5), "B1(c) MAD: got {m}, expected 1");
}

#[test]
fn c1_robust_z_b7() {
    let controls = [98.0, 102.0, 100.0, 95.0, 130.0];
    let treated = [101.0, 140.0, 99.0, 160.0, 120.0];
    let expected = [0.34, 13.49, -0.34, 20.23, 6.74];
    for (t, e) in treated.iter().zip(expected) {
        let z = robust_z(*t, &controls);
        assert!(close(z, e, 0.01), "B7(b) robust z of {t}: got {z:.2}, expected {e}");
    }
}

#[test]
fn c2_object_stats_b1() {
    let (img, labels) = block();
    let s = object_stats(&img, &labels, 1);
    assert_eq!(s.n, 25, "number of pixels");
    assert!(close(s.integrated, 81.0, 1e-4), "B1(a) integrated: got {}, expected 81", s.integrated);
    assert!(close(s.mean, 3.24, 1e-4), "B1(a) mean: got {}, expected 3.24", s.mean);
    assert!(s.min == 1.0 && s.max == 9.0, "B1(a) min, max: got {}, {}, expected 1, 9", s.min, s.max);
    assert!(close(s.sd, 2.12, 0.005), "B3 standard deviation: got {}, expected 2.12", s.sd);
    assert!(s.median == 2.0 && s.q1 == 2.0 && s.q3 == 4.0 && s.mad == 1.0, "B1: got {s:?}");
}

#[test]
fn c2_only_the_object_counts() {
    // The block inside a 7 × 7 image, background 0 with label 0.
    let img = Image::from_fn(7, 7, |x, y| if (1..6).contains(&x) && (1..6).contains(&y) { BLOCK[y - 1][x - 1] } else { 0.0 });
    let labels = img.map(|v| if v > 0.0 { 1 } else { 0 });
    let s = object_stats(&img, &labels, 1);
    assert!(s.n == 25 && close(s.mean, 3.24, 1e-4), "the background must not count: got n = {}, mean {}", s.n, s.mean);
}

#[test]
fn c2_offset_and_gain_b3() {
    let (img, labels) = block();
    let s = object_stats(&img.map(|v| v + 10.0), &labels, 1);
    assert!(close(s.integrated, 331.0, 1e-3) && close(s.mean, 13.24, 1e-4) && s.median == 12.0, "B3(a) offset 10: got {s:?}");
    assert!(close(s.sd, 2.12, 0.005) && s.mad == 1.0, "B3(a) the spread must not change: got sd {}, MAD {}", s.sd, s.mad);
    let g = object_stats(&img.map(|v| 2.0 * v), &labels, 1);
    assert!(close(g.mean, 6.48, 1e-4) && g.mad == 2.0 && close(g.sd, 4.24, 0.01), "B3(c) gain 2: got {g:?}");
}

#[test]
fn c2_mask_too_large_b4() {
    let img = Image::from_fn(7, 7, |x, y| if (1..6).contains(&x) && (1..6).contains(&y) { BLOCK[y - 1][x - 1] } else { 1.0 });
    let s = object_stats(&img, &Image::new(7, 7, 1), 1);
    assert!(close(s.mean, 105.0 / 49.0, 1e-4) && s.median == 1.0, "B4: got mean {}, median {}, expected 2.14 and 1", s.mean, s.median);
}

#[test]
fn c3_edge_band_b2() {
    let (img, labels) = block();
    let band = edge_band(&labels, 1);
    let count = band.pixels().iter().filter(|&&b| b).count();
    assert_eq!(count, 16, "B2: the edge band of the 5 × 5 block is its outer ring of 16 pixels");
    assert!(!band[(2, 2)] && band[(0, 2)] && band[(4, 4)], "B2: the centre is not in the band, the sides are");
    let sum: f32 = img.coords().filter(|&p| band[p]).map(|p| img[p]).sum();
    assert!(close(sum / 16.0, 2.4375, 1e-4), "B2(a) edge mean: got {}, expected 39/16 = 2.44", sum / 16.0);
    // Another label next to the object also makes an edge.
    let labels = Image::from_rows(&[[1, 1, 1, 2], [1, 1, 1, 2], [1, 1, 1, 2]]);
    let band = edge_band(&labels, 1);
    assert!(band[(2, 1)] && !band[(1, 1)], "a pixel next to object 2 is on the edge of object 1");
}

#[test]
fn c4_mass_displacement_b5() {
    let (img, labels) = block();
    let (g, w, d) = mass_displacement(&img, &labels, 1);
    assert!(close(g[0], 2.0, 1e-4) && close(g[1], 2.0, 1e-4), "B5(a) geometric centroid: got {g:?}, expected (2, 2)");
    assert!(close(w[0], 195.0 / 81.0, 1e-4) && close(w[1], 178.0 / 81.0, 1e-4), "B5(b) weighted centroid: got {w:?}, expected (2.41, 2.20)");
    assert!(close(d, 0.454, 0.002), "B5(c) mass displacement: got {d}, expected 0.45");
    let (_, _, d) = mass_displacement(&Image::new(5, 5, 7.0), &labels, 1);
    assert!(close(d, 0.0, 1e-5), "B9: a uniform object has displacement 0, got {d}");
}

#[test]
fn c5_rings_b6() {
    let got = [ring_of(1.0, 4.0, 3), ring_of(3.0, 3.0, 3), ring_of(5.0, 1.0, 3)];
    assert_eq!(got, [1, 2, 3], "B6(a) rings for (dc, de) = (1, 4), (3, 3), (5, 1)");
    assert_eq!(ring_of(1.0, 0.0, 4), 4, "a pixel with ρ = 1 belongs to the last ring");
    let values = [10., 10., 6., 6., 6., 2., 2., 2., 2., 2.];
    let rings = [1, 1, 2, 2, 2, 3, 3, 3, 3, 3];
    let (frac, mean_frac) = radial_distribution(&values, &rings, 3);
    let (ef, em) = ([20. / 48., 18. / 48., 10. / 48.], [2.0833, 1.25, 0.4167]);
    for j in 0..3 {
        assert!(close(frac[j], ef[j], 1e-3), "B6(b) FracAtD: got {frac:?}, expected [0.417, 0.375, 0.208]");
        assert!(close(mean_frac[j], em[j], 1e-3), "B6(b) MeanFrac: got {mean_frac:?}, expected [2.08, 1.25, 0.42]");
    }
}
