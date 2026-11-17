// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p day17 skel_neurovessel::

use day17::skel_neurovessel::*;

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

fn point(x: f32, y: f32, parent: Option<usize>) -> Point {
    Point { x, y, radius: 1.0, parent }
}

/// A point at distance `r` from the soma (at the origin), at angle `deg` degrees.
fn polar(r: f32, deg: f32, parent: usize) -> Point {
    let a = deg.to_radians();
    point(r * a.cos(), r * a.sin(), Some(parent))
}

/// The neuron of B1 with straight edges: soma S (0) at the origin; a (1), c (2), d (3),
/// e (4), f (5), b (6) at the distances r_out of the table.
fn b1() -> Vec<Point> {
    vec![
        point(0.0, 0.0, None),
        polar(36.0, 0.0, 0),   // a
        polar(70.0, 10.0, 1),  // c
        polar(60.0, -10.0, 1), // d
        polar(78.0, -5.0, 3),  // e
        polar(90.0, -20.0, 3), // f
        polar(55.0, 180.0, 0), // b
    ]
}

#[test]
fn c1_counts_b1() {
    let (_, branch_points, tips) = tree_counts(&b1());
    assert_eq!((branch_points, tips), (2, 4), "B1(a) branch points a, d and tips b, c, e, f");
    // lengths: a 3-4-5 triangle path
    let path = vec![point(0.0, 0.0, None), point(3.0, 4.0, Some(0)), point(3.0, 10.0, Some(1))];
    let (length, _, tips) = tree_counts(&path);
    assert!(close(length, 11.0, 1e-4) && tips == 1, "a path of lengths 5 and 6: got length {length}, {tips} tips");
}

#[test]
fn c2_branches_b1() {
    let b = branches(&b1());
    assert_eq!(b.len(), 6, "B1 has 6 edges: S–a, S–b, a–c, a–d, d–e, d–f");
    let mut ends: Vec<(usize, usize)> = b.iter().map(|br| (br.points[0], *br.points.last().unwrap())).collect();
    ends.sort();
    assert_eq!(ends, [(0, 1), (0, 6), (1, 2), (1, 3), (3, 4), (3, 5)], "the branches as (start, end) point numbers");
}

#[test]
fn c2_tortuosity_of_an_arc_b7() {
    // A half circle of radius 50 from (−50, 0) to (50, 0) in 200 small steps (B7: π/2).
    let mut tree = vec![point(-50.0, 0.0, None)];
    for i in 1..=200 {
        let a = std::f32::consts::PI * (1.0 - i as f32 / 200.0);
        tree.push(point(50.0 * a.cos(), 50.0 * a.sin(), Some(i - 1)));
    }
    let b = branches(&tree);
    assert_eq!(b.len(), 1, "an unbranched path is one branch");
    let t = b[0].tortuosity();
    assert!(close(t, std::f32::consts::FRAC_PI_2, 0.002), "B7 half circle: got tortuosity {t}, expected π/2 = 1.571");
    assert!(close(b[0].chord, 100.0, 1e-3), "chord: got {}, expected 100", b[0].chord);
}

#[test]
fn c3_strahler_b1() {
    let s = strahler(&b1());
    // order of the edge from each point to its parent: a 2, c 1, d 2, e 1, f 1, b 1
    assert_eq!(&s[1..], &[2, 1, 2, 1, 1, 1], "B1(c) Strahler orders of the edges S–a, a–c, a–d, d–e, d–f, S–b");
    assert_eq!(s[0], 2, "the soma: orders 2 and 1 meet, the larger continues");
}

#[test]
fn c3_strahler_balanced_b8() {
    // A balanced binary tree with 2^3 = 8 tips: the root edge has order 4 (B8).
    let mut tree = vec![point(0.0, 0.0, None), point(0.0, 1.0, Some(0))];
    let mut frontier = vec![1];
    for _ in 0..3 {
        let mut next = Vec::new();
        for &p in &frontier {
            for _ in 0..2 {
                tree.push(point(0.0, 0.0, Some(p)));
                next.push(tree.len() - 1);
            }
        }
        frontier = next;
    }
    assert_eq!(strahler(&tree)[1], 4, "B8: a balanced tree with 8 tips has order 4 at the root edge");
}

#[test]
fn c4_sholl_b1() {
    let n = sholl(&b1(), 0.0, 0.0, &[20.0, 50.0, 65.0, 80.0, 95.0]);
    assert_eq!(n, [2, 3, 3, 1, 0], "B1(d) Sholl crossings at r = 20, 50, 65, 80, 95");
}

#[test]
fn c5_match_points_b3() {
    let truth = [[20., 30.], [45., 32.], [60., 70.], [80., 40.], [95., 90.]];
    let measured = [[22., 31.], [47., 36.], [81., 43.], [70., 75.], [95., 92.], [30., 80.]];
    let (mut matched, missed, spurious) = match_points(&truth, &measured, 5.0);
    matched.sort();
    assert_eq!(matched, [(0, 0), (1, 1), (3, 2), (4, 4)], "B3(a) matched pairs (true, measured), numbered from 0");
    assert_eq!(missed, [2], "B3(a) the true point (60, 70) is missed");
    assert_eq!(spurious, [3, 5], "B3(a) (70, 75) and (30, 80) are spurious");
    let (matched, missed, _) = match_points(&truth, &measured, 12.0);
    assert!(matched.len() == 5 && missed.is_empty(), "B3(c) with a tolerance of 12 the displaced point is found");
    // closest pairs first: one measured point between two true ones goes to the closer
    let (matched, _, _) = match_points(&[[0., 0.], [10., 0.]], &[[6., 0.]], 8.0);
    assert_eq!(matched, [(1, 0)], "the measured point at 6 belongs to the true point at 10, not 0");
}
