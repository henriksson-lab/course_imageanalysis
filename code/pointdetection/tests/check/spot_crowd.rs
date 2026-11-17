// Checks for the coding exercises, using the numbers of the pen-and-paper exercises and the lecture.
// Run: cargo test -p pointdetection spot_crowd::

use pointdetection::spot_crowd::*;

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() <= tol
}

/// Two equal Gaussians of width `w`, `d` apart, sampled every 0.01 px.
fn pair_profile(d: f32, w: f32) -> Vec<f32> {
    (0..2000)
        .map(|i| {
            let x = -10.0 + 0.01 * i as f32;
            let g = |c: f32| (-(x - c).powi(2) / (2.0 * w * w)).exp();
            g(-d / 2.0) + g(d / 2.0)
        })
        .collect()
}

#[test]
fn c1_merging_b1() {
    let s: f32 = 1.5;
    assert_eq!(count_peaks(&pair_profile(4.0, s)), 2, "B1(a): σ = 1.5, d = 4 > 2σ: two maxima in the raw image");
    let matched = (s * s + s * s).sqrt();
    assert_eq!(count_peaks(&pair_profile(4.0, matched)), 1, "B1(b): after the matched filter (width 2.12) only one");
    let narrower = (s * s + 1.3f32 * 1.3).sqrt();
    assert_eq!(count_peaks(&pair_profile(4.0, narrower)), 2, "B1(c): with σ_f = 1.3 < 1.32 they stay apart");
    assert_eq!(count_peaks(&pair_profile(2.9, 1.5)), 1, "d = 2.9 < 2σ = 3: one maximum");
}

/// B2: truths T1..T3 and detections D1..D3 on a line, gate 1.8.
fn b2() -> (Vec<[f32; 2]>, Vec<[f32; 2]>) {
    (vec![[0.0, 0.0], [3.0, 0.0], [6.0, 0.0]], vec![[1.4, 0.0], [4.4, 0.0], [-1.2, 0.0]])
}

#[test]
fn c2_greedy_b2() {
    let (t, d) = b2();
    let mut pairs = match_greedy(&t, &d, 1.8);
    pairs.sort();
    assert_eq!(pairs, vec![(0, 2), (1, 1)], "B2(b): greedy pairs D3–T1 and D2–T2 (as (truth, detection), from 0)");
    let s = scores(&t, &d, &pairs);
    assert_eq!((s.tp, s.fp, s.fn_), (2, 1, 1), "B2(b): TP, FP, FN");
    assert!(close(s.f1, 0.667, 0.001) && close(s.jaccard, 0.5, 1e-4), "B2(b): F1 {} (expected 0.67), Jaccard {} (expected 0.50)", s.f1, s.jaccard);
    assert!(close(s.rmse, 1.30, 0.005), "B2(b): RMSE got {}, expected 1.30", s.rmse);
}

#[test]
fn c3_optimal_b2() {
    let (t, d) = b2();
    let pairs = match_optimal(&t, &d, 1.8);
    assert_eq!(pairs, vec![(0, 2), (1, 0), (2, 1)], "B2(c): optimal pairs T1–D3, T2–D1, T3–D2");
    let s = scores(&t, &d, &pairs);
    assert!(s.tp == 3 && close(s.f1, 1.0, 1e-6), "B2(c): TP {} F1 {}, expected 3 and 1", s.tp, s.f1);
    assert!(close(s.rmse, 1.48, 0.005), "B2(c): RMSE got {}, expected 1.48", s.rmse);
}

#[test]
fn c3_lecture_example() {
    // T1 = (0, 0), T2 = (2.5, 0); D1 = (1, 0), D2 = (−2, 0); gate 2.5
    let t = [[0.0, 0.0], [2.5, 0.0]];
    let d = [[1.0, 0.0], [-2.0, 0.0]];
    let g = scores(&t, &d, &match_greedy(&t, &d, 2.5));
    assert!(g.tp == 1 && close(g.f1, 0.5, 1e-6), "greedy: TP {} F1 {}, expected 1 and 0.5", g.tp, g.f1);
    let o = scores(&t, &d, &match_optimal(&t, &d, 2.5));
    assert!(o.tp == 2 && close(o.rmse, 1.77, 0.005), "optimal: TP {} RMSE {}, expected 2 and 1.77", o.tp, o.rmse);
    // equal counts: the smaller total distance wins
    let t = [[0.0, 0.0], [1.0, 0.0]];
    let d = [[0.1, 0.0], [1.1, 0.0]];
    assert_eq!(match_optimal(&t, &d, 2.0), vec![(0, 0), (1, 1)], "two matchings of 2 pairs: the shorter one");
}

#[test]
fn c5_pr_curve_b3() {
    let ranked = [true, true, false, true, false, true];
    let (curve, ap) = pr_curve(&ranked, 5);
    let expected = [(1.0, 0.2), (1.0, 0.4), (0.667, 0.4), (0.75, 0.6), (0.6, 0.6), (0.667, 0.8)];
    for (k, (&(p, r), &(pe, re))) in curve.iter().zip(&expected).enumerate() {
        assert!(close(p, pe, 0.001) && close(r, re, 1e-4), "B3(a): after detection {}: got ({p}, {r}), expected ({pe}, {re})", k + 1);
    }
    assert!(close(ap, 0.683, 0.002), "B3(c): AP got {ap}, expected 0.2 · (1 + 1 + 0.75 + 0.67) = 0.68");
}
