// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p day23 stitching::

use imgcourse::samples;
use day23::stitching::*;

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() < tol
}

#[test]
fn c1_c2_four_candidates_b1b() {
    let c = candidates(80, 10, 100, 100);
    let expected = [(80, 10), (-20, 10), (80, -90), (-20, -90)];
    for e in expected {
        assert!(c.contains(&e), "B1(b): candidates {c:?} must contain {e:?}");
    }
    // their overlaps for two 100 × 100 tiles
    let tile = samples::cell().crop(0, 0, 100, 100);
    for ((tx, ty), n) in expected.into_iter().zip([1800, 7200, 200, 800]) {
        let (_, got) = ncc_overlap(&tile, &tile, tx, ty);
        assert_eq!(got, n, "B1(b): candidate ({tx}, {ty}) overlaps {got} pixels, expected {n}");
    }
}

#[test]
fn c2_ncc_on_the_overlap() {
    let cell = samples::cell();
    let a = cell.crop(40, 30, 64, 64);
    let b = cell.crop(90, 37, 64, 64).map(|v| 0.7 * v + 0.2); // 50 right, 7 down, darker
    let (c, n) = ncc_overlap(&a, &b, 50, 7);
    assert_eq!(n, 14 * 57, "overlap of ({}, {}): {n} pixels, expected 14 × 57", 50, 7);
    assert!((c - 1.0).abs() < 1e-4, "NCC at the true shift: {c}, expected 1");
    assert_eq!(ncc_overlap(&a, &b, 64, 0).1, 0, "shift (64, 0): the tiles only touch, the overlap must be empty");
}

#[test]
fn c3_register_pair_resolves_the_candidates() {
    let cell = samples::cell();
    let a = cell.crop(10, 10, 128, 128);
    for (tx, ty) in [(100, 5), (3, 104), (-3, 0)] {
        let b = cell.crop((10 + tx) as usize, (10 + ty) as usize, 128, 128);
        let got = register_pair(&a, &b, 1000);
        assert!(matches!(got, Some((x, y, _)) if (x, y) == (tx, ty)), "B at ({tx}, {ty}) from A: got {got:?}");
    }
    let b = cell.crop(0, 10, 128, 128); // B is 10 px left of A: the peak's other candidates overlap less
    let got = register_pair(&a, &b, 1000);
    assert!(matches!(got, Some((-10, 0, _))), "B at (−10, 0) from A: got {got:?}");
}

#[test]
fn c4_parabola_b2() {
    let d = parabola(0.5, 0.9, 0.7);
    assert!(close(d as f64, 1.0 / 6.0, 1e-5), "B2(a): got {d}, expected 0.167");
    let d = parabola(0.6, 0.9, 0.6);
    assert!(d.abs() < 1e-6, "B2(b): got {d}, expected 0");
    let d = parabola(-1.0, 2.0, 2.0);
    assert!(close(d as f64, 0.5, 1e-5), "two equal top samples: got {d}, expected 0.5");
}

#[test]
fn c5_three_tiles_b3() {
    let pair = |i, j, d, w| Pair { i, j, dx: d, dy: 0.0, w };
    let pos = global_positions(3, &[pair(0, 1, 100.0, 1.0), pair(1, 2, 102.0, 1.0), pair(0, 2, 199.0, 1.0)]).expect("solvable");
    assert!(close(pos[1].0, 99.0, 1e-6) && close(pos[2].0, 200.0, 1e-6), "B3(b): got x = {:?}, expected 0, 99, 200", pos.iter().map(|p| p.0).collect::<Vec<_>>());
    let pos = global_positions(3, &[pair(0, 1, 100.0, 1.0), pair(1, 2, 102.0, 1.0), pair(0, 2, 199.0, 4.0)]).unwrap();
    assert!(close(pos[1].0, 98.6667, 1e-3) && close(pos[2].0, 199.3333, 1e-3), "B3(c), w13 = 4: got x = {:?}, expected 0, 98.67, 199.33", pos.iter().map(|p| p.0).collect::<Vec<_>>());
}

#[test]
fn c5_square_of_four_b4() {
    // tiles 0 top left, 1 top right, 2 bottom right, 3 bottom left (exercise: 1, 2, 3, 4)
    let pairs = [
        Pair { i: 0, j: 1, dx: 100.0, dy: 1.0, w: 1.0 },
        Pair { i: 3, j: 2, dx: 102.0, dy: -1.0, w: 1.0 },
        Pair { i: 0, j: 3, dx: 0.0, dy: 80.0, w: 1.0 },
        Pair { i: 1, j: 2, dx: 0.0, dy: 80.0, w: 1.0 },
    ];
    let pos = global_positions(4, &pairs).expect("solvable");
    let expected = [(0.0, 0.0), (100.5, 0.5), (101.0, 80.0), (-0.5, 80.5)];
    for (k, (p, e)) in pos.iter().zip(expected).enumerate() {
        assert!(close(p.0, e.0, 1e-6) && close(p.1, e.1, 1e-6), "B4, tile {}: got {p:?}, expected {e:?}", k + 1);
    }
}

#[test]
fn c5_a_tile_without_pairs_cannot_be_placed() {
    let pairs = [Pair { i: 0, j: 1, dx: 5.0, dy: 5.0, w: 1.0 }];
    assert!(global_positions(3, &pairs).is_none(), "tile 2 has no pair: expected None");
}
