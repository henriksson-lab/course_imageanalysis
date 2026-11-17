// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p cellpose

use cellpose::*;
use imgcourse::Image;

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

/// A cell of `n` pixels in a row (B1, B5).
fn row_cell(n: usize) -> Image<bool> {
    Image::new(n, 1, true)
}

#[test]
fn c1_heat_in_1d_b1() {
    let cell = row_cell(5);
    let expected = [[0.0, 0.333, 0.333, 0.333, 0.0], [0.167, 0.556, 0.667, 0.556, 0.167], [0.361, 0.796, 0.926, 0.796, 0.361]];
    for (k, e) in expected.iter().enumerate() {
        let heat = heat_target(&cell, k + 1);
        let got: Vec<f32> = (0..5).map(|x| heat[(x, 0)]).collect();
        for x in 0..5 {
            assert!(close(got[x], e[x], 0.002), "after step {}: got {got:.3?}, expected {e:?} (B1a)", k + 1);
        }
    }
}

#[test]
fn c1_heat_spreads_one_pixel_per_step_b1() {
    let heat = heat_target(&row_cell(11), 3);
    let got: Vec<f32> = (0..11).map(|x| heat[(x, 0)]).collect();
    let expected = [0.0, 0.0, 0.04, 0.22, 0.78, 0.93, 0.78, 0.22, 0.04, 0.0, 0.0];
    for x in 0..11 {
        assert!(close(got[x], expected[x], 0.006), "11 pixels after 3 steps: got {got:.2?}, expected {expected:?} (B1c)");
    }
}

#[test]
fn c2_arrows_in_1d_b1() {
    let cell = row_cell(5);
    let a = arrows(&heat_target(&cell, 3), &cell);
    let got: Vec<[f32; 2]> = (0..5).map(|x| a[(x, 0)]).collect();
    let expected = [[1.0, 0.0], [1.0, 0.0], [0.0, 0.0], [-1.0, 0.0], [-1.0, 0.0]];
    for x in 0..5 {
        assert!(close(got[x][0], expected[x][0], 1e-5) && close(got[x][1], 0.0, 1e-5), "arrows: got {got:?}, expected → → none ← ← (B1b)");
    }
    // B1c: in the 11-pixel cell after 3 steps, pixel 2 has an arrow but pixel 1 has none
    let cell = row_cell(11);
    let a = arrows(&heat_target(&cell, 3), &cell);
    assert!(close(a[(1, 0)][0], 1.0, 1e-5), "11 pixels, pixel 2: got {:?}, expected an arrow to the right (B1c)", a[(1, 0)]);
    assert!(a[(0, 0)] == [0.0, 0.0], "11 pixels, pixel 1: got {:?}, expected no arrow (B1c)", a[(0, 0)]);
}

#[test]
fn c2_arrows_of_a_disc_point_inwards() {
    let disc = Image::from_fn(21, 21, |x, y| (x as f32 - 10.0).powi(2) + (y as f32 - 10.0).powi(2) <= 64.0);
    let a = arrows(&heat_target(&disc, 40), &disc);
    for (x, y, want) in [(3, 10, [1.0, 0.0]), (17, 10, [-1.0, 0.0]), (10, 3, [0.0, 1.0]), (10, 17, [0.0, -1.0])] {
        let got = a[(x, y)];
        assert!(close(got[0], want[0], 0.05) && close(got[1], want[1], 0.05), "disc of radius 8, pixel ({x}, {y}): got {got:?}, expected {want:?} (towards the centre)");
    }
}

/// The field of B2: rows top to bottom, columns left to right. In the code a pixel is
/// (x, y) = (column, row); the exercise writes (row, column).
fn b2_field() -> Image<[f32; 2]> {
    let rows = ["→←↓←←", "→↘↓↙←", "→→·←←", "↗↑↑↘↓", "↗↑↑→·"];
    let arrow = |c: char| match c {
        '→' => [1.0, 0.0],
        '←' => [-1.0, 0.0],
        '↓' => [0.0, 1.0],
        '↑' => [0.0, -1.0],
        '↘' => [1.0, 1.0],
        '↙' => [-1.0, 1.0],
        '↗' => [1.0, -1.0],
        _ => [0.0, 0.0],
    };
    let cells: Vec<Vec<[f32; 2]>> = rows.iter().map(|r| r.chars().map(arrow).collect()).collect();
    Image::from_fn(5, 5, |x, y| cells[y][x])
}

#[test]
fn c3_follow_b2() {
    let field = b2_field();
    let mask = Image::new(5, 5, true);
    let ends = follow(&field, &mask, 10);
    // (row 4, column 0) -> centre; (row 3, column 3) -> (4, 4); (row 0, column 0) back and forth
    assert!(ends[(0, 4)] == [2.0, 2.0], "start (row 4, column 0): got {:?}, expected the centre [2, 2] (B2a)", ends[(0, 4)]);
    assert!(ends[(3, 3)] == [4.0, 4.0], "start (row 3, column 3): got {:?}, expected [4, 4] (B2a)", ends[(3, 3)]);
    let e = ends[(0, 0)];
    assert!(e == [0.0, 0.0] || e == [1.0, 0.0], "start (row 0, column 0): got {e:?}, expected to swing between columns 0 and 1 of row 0 (B2a)");
}

#[test]
fn c4_group_b2() {
    let mask = Image::new(5, 5, true);
    for steps in [10, 11] {
        let labels = group_ends(&follow(&b2_field(), &mask, steps), &mask, 1);
        let n = labels.pixels().iter().copied().max().unwrap_or(0) as usize;
        let mut sizes: Vec<usize> = (1..=n as u32).map(|l| labels.pixels().iter().filter(|&&v| v == l).count()).collect();
        sizes.sort();
        assert_eq!(sizes, vec![2, 4, 19], "{steps} steps: object sizes {sizes:?}, expected 2, 4 and 19 pixels (B2b)");
        let small = group_ends(&follow(&b2_field(), &mask, steps), &mask, 3);
        let n_small = small.pixels().iter().copied().max().unwrap_or(0);
        assert_eq!(n_small, 2, "{steps} steps, sinks of at least 3 particles per pixel: got {n_small} objects, expected 2");
    }
}

#[test]
fn c4_one_wrong_arrow_b5() {
    let signs = [1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0];
    let field = Image::from_fn(10, 1, |x, _| [signs[x], 0.0]);
    let mask = Image::new(10, 1, true);
    let labels = group_ends(&follow(&field, &mask, 6), &mask, 1);
    let got: Vec<u32> = (0..10).map(|x| labels[(x, 0)]).collect();
    assert_eq!(got, vec![1, 1, 1, 2, 2, 2, 2, 2, 2, 2], "objects: got {got:?}, expected pixels 1-3 and 4-10 (B5b)");
    let mut fixed = field.clone();
    fixed[(2, 0)] = [1.0, 0.0];
    let labels = group_ends(&follow(&fixed, &mask, 6), &mask, 1);
    let got: Vec<u32> = (0..10).map(|x| labels[(x, 0)]).collect();
    assert_eq!(got, vec![1; 10], "with the arrow at pixel 3 corrected: got {got:?}, expected one object (B5c)");
}

#[test]
fn c5_flow_error_b3() {
    let own = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]];
    let pred = [[1.0, 0.0], [0.8, 0.6], [0.0, 1.0], [-1.0, 0.0]];
    let e = flow_error(&pred, &own);
    assert!(close(e, 1.1, 1e-5), "B3a: got {e}, expected 1.1");
    let pred = [[1.0, 0.0], [0.8, 0.6], [0.0, 1.0], [0.6, 0.8]];
    let e = flow_error(&pred, &own);
    assert!(close(e, 0.3, 1e-5), "B3b: got {e}, expected 0.3");
}
