// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p day35 gmm_images::

use day35::gmm_images::*;
use imgcourse::{Image, Rng};

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

/// B1: three classes, π = (0.50, 0.35, 0.15), μ = (70, 130, 185), σ = (12, 13, 13).
fn b1() -> Mixture {
    Mixture { weights: vec![0.50, 0.35, 0.15], means: vec![70.0, 130.0, 185.0], variances: vec![144.0, 169.0, 169.0] }
}

#[test]
fn c1_histogram() {
    let img = Image::from_rows(&[[0.0, 1.0, 0.5], [1.0, 1.0, 0.501]]);
    let h = histogram(&img);
    assert_eq!(h.len(), 256, "256 bins");
    assert_eq!((h[0], h[255], h[128]), (1.0, 3.0, 2.0), "counts at 0, 255 and 128: got {}, {}, {}, expected 1, 3, 2 (0.5 · 255 = 127.5 rounds to 128)", h[0], h[255], h[128]);
    assert_eq!(h.iter().sum::<f32>(), 6.0, "the counts must add up to the number of pixels");
}

#[test]
fn c2_log_sum_exp_b8() {
    let v = log_sum_exp(&[-800.0, -802.0]);
    assert!(close(v, -799.873, 1e-3), "B8: log(e^−800 + e^−802) = {v}, expected −799.873");
    let v = log_sum_exp(&[0.0, 0.0]);
    assert!(close(v, 2.0f32.ln(), 1e-6), "log(1 + 1) = {v}, expected ln 2");
}

#[test]
fn c3_posteriors_b1() {
    let cases = [(95.0, [0.87, 0.13, 0.00]), (155.0, [0.00, 0.84, 0.16]), (165.0, [0.00, 0.17, 0.83]), (100.0, [0.49, 0.51, 0.00])];
    for (g, e) in cases {
        let r = posteriors(g, &b1());
        for k in 0..3 {
            assert!(close(r[k], e[k], 0.01), "B1: posteriors at {g}: got {r:?}, expected {e:?}");
        }
    }
    // Far from every class: still finite (thanks to log_sum_exp), and the widest class wins.
    let r = posteriors(-500.0, &b1());
    assert!(r.iter().all(|v| v.is_finite()) && close(r.iter().sum(), 1.0, 1e-4), "posteriors at −500 should be finite and add up to 1, got {r:?}");
}

#[test]
fn c5_labels_b1() {
    // B1(b): the boundary between classes 2 and 3 is at 160.1, not at the midpoint 157.5.
    let img = Image::from_rows(&[[95.0 / 255.0, 158.0 / 255.0, 160.0 / 255.0, 161.0 / 255.0, 170.0 / 255.0]]);
    let labels = label_image(&img, &b1());
    let got: Vec<u32> = (0..5).map(|x| labels[(x, 0)]).collect();
    assert_eq!(got, vec![1, 2, 2, 3, 3], "B1(b): labels at grey levels 95, 158, 160, 161, 170");
    let p = probability_map(&img, &b1(), 0);
    assert!(close(p[(0, 0)], 0.87, 0.01), "probability of class 1 at 95: got {}, expected 0.87", p[(0, 0)]);
}

/// EM on all pixels, one by one, for the check of C4.
fn em_per_pixel(values: &[f32], start: &Mixture, iterations: usize) -> Mixture {
    let mut m = start.clone();
    for _ in 0..iterations {
        let kk = m.weights.len();
        let r: Vec<Vec<f32>> = values.iter().map(|&g| posteriors(g, &m)).collect();
        for k in 0..kk {
            let nk: f32 = r.iter().map(|ri| ri[k]).sum();
            let mean: f32 = values.iter().zip(&r).map(|(g, ri)| ri[k] * g).sum::<f32>() / nk;
            let var: f32 = values.iter().zip(&r).map(|(g, ri)| ri[k] * (g - mean) * (g - mean)).sum::<f32>() / nk;
            m.weights[k] = nk / values.len() as f32;
            m.means[k] = mean;
            m.variances[k] = var;
        }
    }
    m
}

#[test]
fn c4_em_histogram_equals_em_per_pixel_b6() {
    // 3000 grey levels from two classes; integer grey levels so that the histogram is exact.
    let mut rng = Rng::new(2);
    let values: Vec<f32> = (0..3000).map(|i| if i % 3 == 0 { 180.0 + 15.0 * rng.normal() } else { 80.0 + 25.0 * rng.normal() }).map(|g: f32| g.round().clamp(0.0, 255.0)).collect();
    let img = Image::from_vec(3000, 1, values.iter().map(|g| g / 255.0).collect());
    let hist = histogram(&img);
    let start = Mixture { weights: vec![0.5, 0.5], means: vec![50.0, 200.0], variances: vec![400.0, 400.0] };
    let (m, trace) = em_histogram(&hist, &start, 1.0, 20);
    let p = em_per_pixel(&values, &start, 20);
    for k in 0..2 {
        assert!(close(m.means[k], p.means[k], 0.05), "B6: histogram EM means {:?}, per-pixel EM {:?}", m.means, p.means);
        assert!(close(m.variances[k], p.variances[k], 0.5), "B6: histogram EM variances {:?}, per-pixel EM {:?}", m.variances, p.variances);
        assert!(close(m.weights[k], p.weights[k], 1e-3), "B6: histogram EM weights {:?}, per-pixel EM {:?}", m.weights, p.weights);
    }
    assert_eq!(trace.len(), 20, "one log-likelihood per iteration");
    for w in trace.windows(2) {
        assert!(w[1] >= w[0] - 0.5, "the log-likelihood went down: {trace:?}");
    }
    assert!(close(m.weights[1], 1.0 / 3.0, 0.03) && close(m.means[1], 180.0, 2.0), "the bright class: weight {}, mean {}, expected about 1/3 and 180", m.weights[1], m.means[1]);
}
