// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p track-linking

use imgcourse::Rng;
use track_linking::*;

const INF: f32 = f32::INFINITY;

/// B1: A(0, 0), B(3, 0) at frame t; P(2, 0), Q(5, 0) at frame t+1.
fn b1() -> Vec<Vec<f32>> {
    cost_matrix(&[[0.0, 0.0], [3.0, 0.0]], &[[2.0, 0.0], [5.0, 0.0]], 100.0)
}

/// B3: rows A, B, C; columns P, Q, R; B–R is outside the gate.
fn b3() -> Vec<Vec<f32>> {
    vec![vec![1.0, 2.0, 8.0], vec![2.0, 6.0, INF], vec![9.0, 3.0, 5.0]]
}

fn total(cost: &[Vec<f32>], links: &[Option<usize>]) -> f32 {
    links.iter().enumerate().filter_map(|(i, l)| l.map(|j| cost[i][j])).sum()
}

fn by_search(cost: &[Vec<f32>]) -> Vec<usize> {
    global_by_search(cost).expect("the augmented matrix always has a finite assignment")
}

#[test]
fn c1_cost_matrix_b1() {
    let c = b1();
    let expected = [[4.0, 25.0], [1.0, 4.0]];
    for i in 0..2 {
        for j in 0..2 {
            assert!((c[i][j] - expected[i][j]).abs() < 1e-5, "B1(a): got {c:?}, expected {expected:?}");
        }
    }
}

#[test]
fn c1_gate() {
    // gate 3: A–Q (distance 5) is not allowed, B–Q (distance 2) is
    let c = cost_matrix(&[[0.0, 0.0], [3.0, 0.0]], &[[2.0, 0.0], [5.0, 0.0]], 3.0);
    assert!(c[0][1].is_infinite(), "A–Q is 5 apart, outside a gate of 3: got {}, expected infinity", c[0][1]);
    assert!((c[1][1] - 4.0).abs() < 1e-5, "B–Q is 2 apart, inside the gate: got {}, expected 4", c[1][1]);
}

#[test]
fn c2_nearest_conflict_b1() {
    let links = nearest(&b1());
    assert_eq!(links, vec![Some(0), Some(0)], "B1(a): both A and B should choose P (a conflict)");
    let gated = nearest(&[vec![INF, INF], vec![3.0, INF]]);
    assert_eq!(gated, vec![None, Some(0)], "a row with only infinite costs has no link");
}

#[test]
fn c3_greedy_b1() {
    let c = b1();
    let links = greedy(&c);
    assert_eq!(links, vec![Some(1), Some(0)], "B1(b): greedy takes B–P first, then A–Q");
    assert!((total(&c, &links) - 26.0).abs() < 1e-4, "B1(b): total {}, expected 26", total(&c, &links));
}

#[test]
fn c3_greedy_b3() {
    let links = greedy(&b3());
    assert_eq!(links, vec![Some(0), None, Some(1)], "B3(b): A–P, C–Q, and B is left without a link");
}

#[test]
fn c4_global_b1_b3() {
    let links = global_by_search(&b1()).expect("B1 has a solution");
    assert_eq!(links, vec![0, 1], "B1(c): A–P and B–Q");
    let links = global_by_search(&b3()).expect("B3 has a solution");
    assert_eq!(links, vec![1, 0, 2], "B3(a): A–Q, B–P, C–R (total 9)");
    assert!(global_by_search(&[vec![INF, 1.0], vec![INF, 2.0]]).is_none(), "both rows need column 2: no solution");
}

#[test]
fn c4_global_agrees_with_fast_solver() {
    let mut rng = Rng::new(7);
    for _ in 0..20 {
        let c: Vec<Vec<f32>> = (0..6).map(|_| (0..6).map(|_| (rng.below(50)) as f32).collect()).collect();
        let mine = global_by_search(&c).expect("a full matrix has a solution");
        let fast = hungarian(&c);
        let t = |a: &[usize]| a.iter().enumerate().map(|(i, &j)| c[i][j]).sum::<f32>();
        assert!((t(&mine) - t(&fast)).abs() < 1e-3, "total {}, but the fast solver finds {}", t(&mine), t(&fast));
    }
}

#[test]
fn c5a_augmented_b7() {
    let m = augmented(&b1(), 3.0);
    let expected = [[4.0, 25.0, 3.0, INF], [1.0, 4.0, INF, 3.0], [3.0, INF, 0.0, 0.0], [INF, 3.0, 0.0, 0.0]];
    for i in 0..4 {
        for j in 0..4 {
            assert_eq!(m[i][j], expected[i][j], "B7: row {}, column {}: got {}, expected {}", i + 1, j + 1, m[i][j], expected[i][j]);
        }
    }
}

#[test]
fn c5b_ends_and_starts_b2() {
    let c = b1();
    // B2(b): b > 3.5 both links; 0.5 < b < 3.5 only B–P; b < 0.5 no links.
    for (b, expected) in [(4.0, vec![Some(0), Some(1)]), (3.0, vec![None, Some(0)]), (2.0, vec![None, Some(0)]), (0.4, vec![None, None])] {
        let links = link_with_ends(&c, b, by_search);
        assert_eq!(links, expected, "B2, b = {b}");
        assert_eq!(link_with_ends(&c, b, hungarian), expected, "B2, b = {b}, with the fast solver");
    }
}
