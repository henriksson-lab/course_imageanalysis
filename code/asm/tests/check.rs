// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p asm

use asm::*;
use imgcourse::Image;

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

const PROFILE: [f32; 9] = [0.0, 0.6, 0.0, 0.1, 0.3, 0.5, 0.2, 0.0, 0.1];
const MODEL: [f32; 3] = [0.3, 0.5, 0.2];

#[test]
fn c1a_strongest_edge_b1() {
    let o = strongest_edge(&PROFILE);
    assert_eq!(o, -3, "B1(a): got offset {o}, expected −3");
}

#[test]
fn c1b_match_profile_b1() {
    let (best, costs) = match_profile(&PROFILE, &MODEL);
    let expected = [0.14, 0.35, 0.26, 0.17, 0.0, 0.17, 0.27];
    assert_eq!(costs.len(), 7, "B1(b): expected 7 costs (centres −3 to 3), got {costs:?}");
    for (c, e) in costs.iter().zip(expected) {
        assert!(close(*c, e, 1e-5), "B1(b): costs {costs:?}, expected {expected:?}");
    }
    assert_eq!(best, 1, "B1(b): best centre {best}, expected +1");
}

/// The triangle model of B2.
fn triangle() -> ShapeModel {
    let r = std::f32::consts::FRAC_1_SQRT_2;
    ShapeModel {
        mean: vec![-1.0, 0.0, 1.0, 0.0, 0.0, 1.0],
        modes: vec![vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0], vec![-r, 0.0, r, 0.0, 0.0, 0.0]],
        values: vec![0.04, 0.02],
    }
}

#[test]
fn c2_project_clamp_b2() {
    let model = triangle();
    let (b, shape) = project_clamp(&[-1.2, 0.1, 1.2, -0.1, 0.0, 1.3], &model);
    assert!(close(b[0], 0.3, 1e-5) && close(b[1], 0.2828, 1e-4), "B2(b): b = {b:?}, expected [0.3, 0.283]");
    let expected = [-1.2, 0.0, 1.2, 0.0, 0.0, 1.3];
    for (s, e) in shape.iter().zip(expected) {
        assert!(close(*s, e, 1e-4), "B2(b): shape {shape:?}, expected {expected:?}");
    }
    let (b, shape) = project_clamp(&[-1.8, 0.0, 1.8, 0.0, 0.0, 2.5], &model);
    assert!(close(b[0], 0.6, 1e-5) && close(b[1], 0.4243, 1e-4), "B2(c): clamped b = {b:?}, expected [0.6, 0.424]");
    let expected = [-1.3, 0.0, 1.3, 0.0, 0.0, 1.6];
    for (s, e) in shape.iter().zip(expected) {
        assert!(close(*s, e, 1e-4), "B2(c): clamped shape {shape:?}, expected {expected:?}");
    }
}

fn circle(cx: f32, cy: f32, r: f32, n: usize) -> Vec<Point> {
    (0..n)
        .map(|i| {
            let t = 2.0 * std::f32::consts::PI * i as f32 / n as f32;
            [cx + r * t.cos(), cy + r * t.sin()]
        })
        .collect()
}

#[test]
fn c3_normals_of_a_circle() {
    let mut points = circle(5.0, 3.0, 2.0, 16);
    for reversed in [false, true] {
        if reversed {
            points.reverse();
        }
        for (p, n) in points.iter().zip(normals(&points)) {
            let radial = [(p[0] - 5.0) / 2.0, (p[1] - 3.0) / 2.0];
            assert!(close(n[0], radial[0], 1e-4) && close(n[1], radial[1], 1e-4), "at {p:?}: normal {n:?}, expected {radial:?} (outward, length 1)");
        }
    }
}

/// A bright square, pixels 11..=29 in x and y, on a dark 40 × 40 image: its right edge lies
/// between x = 29 and 30.
fn square_image() -> Image<f32> {
    Image::from_fn(40, 40, |x, y| if (11..=29).contains(&x) && (11..=29).contains(&y) { 1.0 } else { 0.0 })
}

#[test]
fn c4_suggest_moves_to_the_edge() {
    // 16 landmarks on a square of half-size 6 around (20, 20); the middle of the right side is
    // landmark 2 at (26, 20), with the normal (1, 0).
    let mut points = Vec::new();
    for i in 0..4 {
        points.push([26.0, 14.0 + 3.0 * i as f32]);
    }
    for i in 0..4 {
        points.push([26.0 - 3.0 * i as f32, 26.0]);
    }
    for i in 0..4 {
        points.push([14.0, 26.0 - 3.0 * i as f32]);
    }
    for i in 0..4 {
        points.push([14.0 + 3.0 * i as f32, 14.0]);
    }
    let model = [-0.25, -0.5, -0.25]; // bright inside, dark outside: the derivative outward is negative
    let y = suggest(&square_image(), &points, 6, &model);
    for (i, edge) in [(2, [29.5, 20.0]), (6, [20.0, 29.5]), (10, [10.5, 20.0]), (14, [20.0, 10.5])] {
        let d = ((y[i][0] - edge[0]).powi(2) + (y[i][1] - edge[1]).powi(2)).sqrt();
        assert!(d < 0.6, "landmark {i} at {:?}: suggested {:?}, expected near the edge {edge:?}", points[i], y[i]);
    }
}

#[test]
fn c5_search_finds_a_disk() {
    // A disk of radius 15 around (32, 32); the model: a circle of radius 1, one mode (its size).
    let img = Image::from_fn(64, 64, |x, y| {
        let r = ((x as f32 - 32.0).powi(2) + (y as f32 - 32.0).powi(2)).sqrt();
        (15.0 - r + 0.5).clamp(0.0, 1.0)
    });
    let mean: Vec<f32> = circle(0.0, 0.0, 1.0, 16).iter().flat_map(|p| [p[0], p[1]]).collect();
    let mode: Vec<f32> = mean.iter().map(|v| v / 4.0).collect(); // length 1
    let model = ShapeModel { mean, modes: vec![mode], values: vec![0.01] };
    let mut pose = Pose { a: 12.0, b: 0.0, t: [35.0, 29.0] };
    let mut b = vec![0.0];
    for _ in 0..8 {
        (pose, b) = asm_iteration(&img, &model, pose, &b, 6, &[-0.25, -0.5, -0.25]);
    }
    let outline = apply(pose, &to_points(&model_shape(&model, &b)));
    let c = centroid(&outline);
    assert!(close(c[0], 32.0, 0.5) && close(c[1], 32.0, 0.5), "after 8 iterations the centre is at {c:?}, expected (32, 32)");
    for p in &outline {
        let r = ((p[0] - 32.0).powi(2) + (p[1] - 32.0).powi(2)).sqrt();
        assert!(close(r, 15.0, 0.8), "landmark {p:?} is at radius {r}, expected about 15");
    }
}
