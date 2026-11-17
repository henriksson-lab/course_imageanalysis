// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p integral rect_features::

use imgcourse::{samples, Image};
use integral::rect_features::*;

/// The 6 × 6 image of the exercise. The exercise counts rows and columns from 1, the code
/// from 0: "columns 2-3, rows 2-5" start at (x, y) = (1, 1).
fn exercise_image() -> Image<f32> {
    Image::from_rows(&[
        [1., 1., 1., 5., 5., 5.],
        [1., 2., 1., 6., 5., 5.],
        [1., 1., 2., 5., 6., 5.],
        [2., 1., 1., 5., 5., 6.],
        [1., 1., 1., 5., 5., 5.],
        [1., 2., 1., 5., 6., 5.],
    ])
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-6
}

#[test]
fn c1_edge_b1() {
    let s = integral(&exercise_image());
    let g = edge_across(&s, 1, 1, 2, 4);
    assert!(close(g, -32.0), "B1, left (columns 2-3) − right (columns 4-5), rows 2-5: got {g}, expected −32");
}

#[test]
fn c2_line_b2() {
    let s = integral(&exercise_image());
    let g = line_across(&s, 2, 1, 1, 4);
    assert!(close(g, 16.0), "B2, 2 · column 4 − column 3 − column 5, rows 2-5: got {g}, expected 16");
}

#[test]
fn c1_c2_c3_which_template_b3() {
    // 1 everywhere, 9 in columns 3 and 4 (x = 2, 3): a bright vertical bar.
    let bar = Image::from_fn(6, 6, |x, _| if x == 2 || x == 3 { 9.0 } else { 1.0 });
    let s = integral(&bar);
    let cases = [
        ("(i) edge across, 3 × 6 rectangles", edge_across(&s, 0, 0, 3, 6), 0.0),
        ("(ii) edge down, 6 × 3 rectangles", edge_down(&s, 0, 0, 6, 3), 0.0),
        ("(iii) line across, 2 × 6 rectangles", line_across(&s, 0, 0, 2, 6), 192.0),
        ("(iv) diagonal, 3 × 3 quarters", diagonal(&s, 0, 0, 3, 3), 0.0),
    ];
    for (name, g, e) in cases {
        assert!(close(g, e), "B3 {name}: got {g}, expected {e}");
    }
}

#[test]
fn c1_c2_c3_flat_regions_give_zero() {
    let s = integral(&Image::new(12, 12, 0.7f32));
    let got = [
        edge_across(&s, 1, 2, 3, 4),
        edge_down(&s, 1, 2, 3, 4),
        line_across(&s, 1, 2, 3, 4),
        line_down(&s, 1, 2, 3, 2),
        diagonal(&s, 1, 2, 3, 4),
        centre_surround(&s, 1, 2, 3, 3),
    ];
    for (name, g) in ["edge across", "edge down", "line across", "line down", "diagonal", "centre − surround"].iter().zip(got) {
        assert!(g.abs() < 1e-4, "{name} on a flat image: got {g}, expected 0 (the weights must sum to zero)");
    }
}

#[test]
fn c1_edge_on_a_ramp_b8() {
    // I(x, y) = a x + b with a = 0.01: left − right gives −a w² h, top − bottom gives 0.
    let ramp = Image::from_fn(40, 30, |x, _| 0.01 * x as f32 + 0.2);
    let s = integral(&ramp);
    let (w, h) = (5, 7);
    let g = edge_across(&s, 3, 4, w, h);
    let e = -0.01 * (w * w * h) as f64;
    assert!((g - e).abs() < 1e-4, "edge across on a ramp: got {g}, expected −a w² h = {e}");
    let g = edge_down(&s, 3, 4, w, h);
    assert!(g.abs() < 1e-4, "edge down on a ramp: got {g}, expected 0");
}

#[test]
fn c3_centre_surround_is_a_blob_detector() {
    let img = samples::cell().crop(50, 50, 30, 30);
    let s = integral(&img);
    // direct: the 4 × 4 centre minus one ninth of the 12 × 12 template at (5, 6)
    let (mut centre, mut all) = (0.0, 0.0);
    for y in 6..18 {
        for x in 5..17 {
            all += img[(x, y)] as f64;
            if (9..13).contains(&x) && (10..14).contains(&y) {
                centre += img[(x, y)] as f64;
            }
        }
    }
    let g = centre_surround(&s, 5, 6, 4, 4);
    assert!((g - (centre - all / 9.0)).abs() < 1e-4, "centre − surround: got {g}, expected {}", centre - all / 9.0);
}

#[test]
fn c4_pool_b4_b9() {
    let cases = [
        ("B4, edge across in 8 × 8", (2, 1, 8, 8), 576),
        ("B9, edge across in 24 × 24", (2, 1, 24, 24), 43_200),
        ("edge down in 24 × 24", (1, 2, 24, 24), 43_200),
        ("line across in 24 × 24", (3, 1, 24, 24), 27_600),
        ("diagonal in 24 × 24", (2, 2, 24, 24), 20_736),
        ("centre − surround in 24 × 24", (3, 3, 24, 24), 8_464),
    ];
    for (name, (a, b, w, h), e) in cases {
        let g = count_placements(a, b, w, h);
        assert_eq!(g, e, "{name}: got {g}, expected {e}");
    }
}
