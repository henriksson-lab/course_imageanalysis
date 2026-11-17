// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p day32 livewire::
//
// The exercise writes pixels as (row, column); the code as (x, y) = (column, row).

use imgcourse::{samples, Image};
use day32::livewire::*;

/// The grid of B1.
fn b1() -> Image<f32> {
    Image::from_rows(&[[1., 7., 8., 9.], [1., 5., 7., 5.], [1., 3., 5., 4.], [1., 5., 3., 6.]])
}

/// The "detour" grid of B3: a channel of 1s, all other pixels 9.
fn detour() -> Image<f32> {
    let rows = ["#.######", "#.######", "#.#....#", "#.#.##.#", "#.#.##.#", "#...##.#", "######.#", "######.#"];
    Image::from_fn(8, 8, |x, y| if rows[y].as_bytes()[x] == b'.' { 1.0 } else { 9.0 })
}

fn rc(row: usize, column: usize) -> (usize, usize) {
    (column, row)
}

#[test]
fn c1_distances_b1() {
    let paths = dijkstra(&b1(), &[(0, 0)], false);
    let expected = [[1., 8., 16., 25.], [2., 7., 14., 19.], [3., 6., 11., 15.], [4., 9., 12., 18.]];
    for (y, row) in expected.iter().enumerate() {
        for (x, &e) in row.iter().enumerate() {
            let got = paths.distance[(x, y)];
            assert_eq!(got, e, "d at row {y}, column {x}: got {got}, expected {e}");
        }
    }
}

#[test]
fn c1_order_b1() {
    let paths = dijkstra(&b1(), &[(0, 0)], false);
    let expected: Vec<(usize, usize)> = [(0, 0), (1, 0), (2, 0), (3, 0), (2, 1), (1, 1), (0, 1), (3, 1), (2, 2), (3, 2), (1, 2), (2, 3), (0, 2), (3, 3), (1, 3), (0, 3)]
        .iter()
        .map(|&(r, c)| rc(r, c))
        .collect();
    assert_eq!(paths.order, expected, "B1 (a): the order in which the pixels become final, as (x, y)");
}

#[test]
fn c2_path_b1() {
    let paths = dijkstra(&b1(), &[(0, 0)], false);
    let path = backtrack(&paths, rc(3, 3));
    let expected: Vec<_> = [(0, 0), (1, 0), (2, 0), (3, 0), (3, 1), (3, 2), (3, 3)].iter().map(|&(r, c)| rc(r, c)).collect();
    assert_eq!(path, expected, "B1 (c): the cheapest path to the bottom-right pixel, as (x, y)");
    assert_eq!(paths.distance[rc(3, 3)], 18.0, "B1 (c): its cost");
}

#[test]
fn c2_detour_up_b2() {
    // B2: C(1, 1) = 8. The best path to (row 1, column 3) now comes from below.
    let mut cost = b1();
    cost[rc(1, 1)] = 8.0;
    let paths = dijkstra(&cost, &[(0, 0)], false);
    assert_eq!(paths.distance[rc(1, 3)], 20.0, "B2 (a): d at row 1, column 3: got {}, expected 20", paths.distance[rc(1, 3)]);
    assert_eq!(paths.predecessor[rc(1, 3)], Some(rc(2, 3)), "B2 (a): row 1, column 3 should come from row 2, column 3 (below)");
}

#[test]
fn c3_raster_pass_b2() {
    let mut cost = b1();
    let d = raster_pass(&cost);
    let true_d = dijkstra(&cost, &[(0, 0)], false).distance;
    assert_eq!(d, true_d, "B1 grid: here no best path goes up or left, so one pass is exact");
    cost[rc(1, 1)] = 8.0;
    let d = raster_pass(&cost);
    assert_eq!(d[rc(1, 3)], 22.0, "B2 (b): one raster pass at row 1, column 3: got {}, expected 22 (true: 20)", d[rc(1, 3)]);
    let true_d = dijkstra(&cost, &[(0, 0)], false).distance;
    let wrong = d.pixels().iter().zip(true_d.pixels()).filter(|(a, b)| a != b).count();
    assert_eq!(wrong, 1, "B2 (b): exactly one pixel should be wrong, got {wrong}");
}

#[test]
fn c1_detour_b3() {
    // B3: from anywhere in the top row to anywhere in the bottom row: 19.
    let cost = detour();
    let seeds: Vec<_> = (0..8).map(|x| (x, 0)).collect();
    let paths = dijkstra(&cost, &seeds, false);
    let best = (0..8).map(|x| paths.distance[(x, 7)]).fold(f32::INFINITY, f32::min);
    assert_eq!(best, 19.0, "B3: the cheapest top-to-bottom path: got {best}, expected 19");
    let pos = |p| paths.order.iter().position(|&q| q == p).unwrap();
    assert!(pos(rc(2, 3)) > pos(rc(5, 1)), "B3: (row 2, column 3) must become final after (row 5, column 1)");
    assert_eq!(paths.distance[rc(2, 3)], 11.0, "B3: d at row 2, column 3: got {}, expected 11", paths.distance[rc(2, 3)]);
}

#[test]
fn c1_grid_distances_b6() {
    // B6: cost 1 per step. Subtract the seed's own cost, 1.
    let flat = Image::new(8, 8, 1.0);
    let four = dijkstra(&flat, &[(0, 0)], false).distance;
    let eight = dijkstra(&flat, &[(0, 0)], true).distance;
    let (x, y) = (3, 4); // 3 columns right, 4 rows down
    assert!((four[(x, y)] - 1.0 - 7.0).abs() < 1e-4, "B6, 4 neighbours: got {}, expected 7", four[(x, y)] - 1.0);
    let e = 3.0 * 2f32.sqrt() + 1.0;
    assert!((eight[(x, y)] - 1.0 - e).abs() < 1e-4, "B6, 8 neighbours: got {}, expected 3√2 + 1 = {e}", eight[(x, y)] - 1.0);
    assert!((eight[(5, 5)] - 1.0 - 5.0 * 2f32.sqrt()).abs() < 1e-4, "B6, 8 neighbours to (5, 5): got {}, expected 5√2", eight[(5, 5)] - 1.0);
}

#[test]
fn c4_edge_cost_step() {
    let step = samples::step(10, 4, 0.0, 1.0); // the edge between columns 4 and 5
    let c = edge_cost(&step);
    assert!((c[(4, 2)] - EPSILON).abs() < 1e-5, "on the edge: got {}, expected EPSILON = {EPSILON}", c[(4, 2)]);
    assert!((c[(1, 2)] - (1.0 + EPSILON)).abs() < 1e-5, "on the flat part: got {}, expected 1 + EPSILON", c[(1, 2)]);
}
