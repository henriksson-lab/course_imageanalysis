// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p graphcuts graphcut_seg::
//
// Grey levels 0 to 3, μ0 = 0, μ1 = 3, D(0) = I², D(1) = (3 − I)², as in the exercise.

use graphcuts::graphcut_seg::*;
use imgcourse::Image;

const INF: f32 = f32::INFINITY;

fn costs(img: &Image<f32>) -> (Image<f32>, Image<f32>) {
    (img.map(|v| v * v), img.map(|v| (3.0 - v) * (3.0 - v)))
}

fn cut<const W: usize>(rows: &[[f32; W]], lambda: f32, sigma: f32) -> (String, f32) {
    let img = Image::from_rows(rows);
    let (d0, d1) = costs(&img);
    let (labels, flow) = graph_cut(&img, &d0, &d1, lambda, sigma, Neighbours::Four);
    (labels.pixels().iter().map(|l| l.to_string()).collect(), flow)
}

#[test]
fn c1_contrast_weight_b2() {
    for (diff, w) in [(2.0, 0.135), (1.0, 0.607), (3.0, 0.011)] {
        let got = contrast_weight(3.0, 3.0 - diff, 1.0);
        assert!((got - w).abs() < 1e-3, "B2(a): difference {diff} with σ = 1: weight {got}, expected {w}");
    }
    assert_eq!(contrast_weight(0.0, 3.0, INF), 1.0, "σ = ∞ gives weight 1");
}

#[test]
fn c2_c3_one_by_three_b1() {
    let (l, e) = cut(&[[3.0, 1.0, 3.0]], 1.0, INF);
    assert_eq!(l, "101", "B1(b): the minimum cut at λ = 1");
    assert!((e - 3.0).abs() < 1e-4, "B1(b): cut cost {e}, expected 3");
    let (l, e) = cut(&[[3.0, 1.0, 3.0]], 2.0, INF);
    assert_eq!(l, "111", "B1(c): the minimum cut at λ = 2");
    assert!((e - 4.0).abs() < 1e-4, "B1(c): cut cost {e}, expected 4");
}

#[test]
fn c2_c3_contrast_b2() {
    let (l, e) = cut(&[[3.0, 1.0, 3.0]], 2.0, 1.0);
    assert_eq!(l, "101", "B2(b): with σ = 1 the dark pixel stays background at λ = 2");
    assert!((e - 1.541).abs() < 1e-3, "B2(b): cut cost {e}, expected 1.541");
    let (l, _) = cut(&[[3.0, 1.0, 3.0]], 11.2, 1.0);
    assert_eq!(l, "111", "B2(b): above λ = 11.08 the dark pixel is filled");
}

#[test]
fn c2_c3_two_by_two_b3() {
    let (l, e) = cut(&[[3.0, 2.0], [1.0, 0.0]], 2.0, INF);
    assert_eq!(l, "1100", "B3(b): the minimum cut (top row object)");
    assert!((e - 6.0).abs() < 1e-4, "B3(b): cut cost {e}, expected 6");
}

#[test]
fn c3_seed_b4() {
    // B4: the bottom-left pixel is an object seed: D(0) = ∞, its D(1) = 4 stays
    let img = Image::from_rows(&[[3.0, 2.0], [1.0, 0.0]]);
    let (mut d0, d1) = costs(&img);
    d0[(0, 1)] = INF;
    let (labels, e) = graph_cut(&img, &d0, &d1, 2.0, INF, Neighbours::Four);
    let l: String = labels.pixels().iter().map(|l| l.to_string()).collect();
    assert_eq!(l, "1110", "B4: with the seed the minimum cut is 1110");
    assert!((e - 9.0).abs() < 1e-4, "B4: cut cost {e}, expected 9");
}

#[test]
fn c3_shrinking_bias_b5() {
    // B5: a 6 × 6 image, a line of 2 in column 2 and a band of 2 in columns 4–6
    let row = [0.0, 2.0, 0.0, 2.0, 2.0, 2.0];
    let line_kept = |lambda: f32| cut(&[row; 6], lambda, INF).0.chars().nth(1) == Some('1');
    let band_kept = |lambda: f32| cut(&[row; 6], lambda, INF).0.chars().nth(4) == Some('1');
    assert!(line_kept(1.4) && !line_kept(1.6), "B5(a): the line should be kept at λ = 1.4 and dropped at 1.6");
    assert!(band_kept(8.9) && !band_kept(9.1), "B5(b): the band should be kept at λ = 8.9 and dropped at 9.1");
}

#[test]
fn c2_eight_neighbours_b6() {
    // B6(b): with 8 neighbours a horizontal and a 45° boundary cost the same per unit length,
    // ½ + 1/√2 = 1.207. Cut costs only (no data, seeds everywhere), for n × n images split
    // horizontally and along the diagonal; the difference between n = 40 and n = 20 removes
    // the effect of the image border.
    let cost_of = |n: usize, object: &dyn Fn(usize, usize) -> bool| {
        let img = Image::new(n, n, 0.0f32);
        let d0 = Image::from_fn(n, n, |x, y| if object(x, y) { INF } else { 0.0 });
        let d1 = Image::from_fn(n, n, |x, y| if object(x, y) { 0.0 } else { INF });
        graph_cut(&img, &d0, &d1, 1.0, INF, Neighbours::Eight).1
    };
    let horizontal = (cost_of(40, &|_, y| y < 20) - cost_of(20, &|_, y| y < 10)) / 20.0;
    let diagonal = (cost_of(40, &|x, y| x + y < 40) - cost_of(20, &|x, y| x + y < 20)) / (20.0 * 2f32.sqrt());
    assert!((horizontal - 1.207).abs() < 0.01, "B6(b): horizontal boundary {horizontal} per unit length, expected 1.207");
    assert!((diagonal - 1.207).abs() < 0.01, "B6(b): 45° boundary {diagonal} per unit length, expected 1.207");
}

#[test]
fn c4_seeded_costs() {
    // object seeds on 0.8 and 0.9, background seeds on 0.1 and 0.3
    let img = Image::from_rows(&[[0.8, 0.9, 0.1, 0.3, 0.5]]);
    let seeds = Image::from_rows(&[[1, 1, 2, 2, 0]]);
    let (m1, s1) = seed_model(&img, &seeds, 1);
    let (m0, s0) = seed_model(&img, &seeds, 2);
    assert!((m1 - 0.85).abs() < 1e-5 && (s1 - 0.05).abs() < 1e-5, "object seeds: mean {m1}, sd {s1}, expected 0.85, 0.05");
    assert!((m0 - 0.2).abs() < 1e-5 && (s0 - 0.1).abs() < 1e-5, "background seeds: mean {m0}, sd {s0}, expected 0.2, 0.1");
    let (d0, d1) = seeded_costs(&img, &seeds);
    assert!(d0[(0, 0)].is_infinite() && d1[(2, 0)].is_infinite(), "seeds are hard constraints: infinite cost for the other label");
    // pixel 0.5: (0.5 − 0.2)²/(2 · 0.01) + log(0.1/0.05) = 4.5 + 0.693; (0.5 − 0.85)²/(2 · 0.0025) = 24.5
    assert!((d0[(4, 0)] - 5.193).abs() < 1e-3, "D(0) of 0.5: got {}, expected 4.5 + log 2 = 5.193", d0[(4, 0)]);
    assert!((d1[(4, 0)] - 24.5).abs() < 1e-2, "D(1) of 0.5: got {}, expected 24.5", d1[(4, 0)]);
}
