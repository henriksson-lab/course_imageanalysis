// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p point-patterns

use imgcourse::Rng;
use point_patterns::*;

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

/// The counts of B1, row by row.
const B1: [[usize; 4]; 4] = [[0, 3, 1, 0], [2, 0, 0, 4], [1, 0, 5, 0], [0, 2, 0, 2]];

/// Points in a 40 × 40 µm window that give the counts of B1 in 4 × 4 quadrats of 10 µm.
fn b1_points() -> Vec<Point> {
    let mut pts = Vec::new();
    for (row, r) in B1.iter().enumerate() {
        for (col, &c) in r.iter().enumerate() {
            for k in 0..c {
                pts.push((10.0 * col as f64 + 1.0 + 1.5 * k as f64, 10.0 * row as f64 + 5.0));
            }
        }
    }
    pts
}

#[test]
fn c1_quadrat_counts_b1() {
    let counts = quadrat_counts(&b1_points(), 40.0, 40.0, 4);
    for row in 0..4 {
        assert_eq!(counts[row], B1[row].to_vec(), "row {} of the counts (B1)", row + 1);
    }
    // a point on the far edge of the window belongs to the last quadrat
    let edge = quadrat_counts(&[(40.0, 40.0)], 40.0, 40.0, 4);
    assert_eq!(edge[3][3], 1, "the point (40, 40) should be counted in the last quadrat");
}

#[test]
fn c2_dispersion_b1() {
    let counts: Vec<Vec<usize>> = B1.iter().map(|r| r.to_vec()).collect();
    let (mean, var, vmr, x2) = dispersion(&counts);
    let cases = [("mean", mean, 1.25), ("variance", var, 2.6), ("VMR", vmr, 2.08), ("X²", x2, 31.2)];
    for (name, got, expected) in cases {
        assert!(close(got, expected, 1e-9), "B1(a) and B7, {name}: got {got}, expected {expected}");
    }
    // B1(c): 4 quadrats with 2 cells, 12 with 1
    let even: Vec<Vec<usize>> = vec![vec![2, 2, 2, 2], vec![1; 4], vec![1; 4], vec![1; 4]];
    let (_, _, vmr, x2) = dispersion(&even);
    assert!(close(vmr, 0.16, 1e-9), "B1(c): VMR {vmr}, expected 0.16");
    assert!(close(x2, 2.4, 1e-9), "B8: X² {x2}, expected 2.4");
}

#[test]
fn c3_thomas_intensity_and_vmr_b7() {
    // κ = 0.0005 per µm², μ = 8, σ = 4 µm: ρ = κμ = 0.004 per µm², VMR of large quadrats ≈ 1 + μ = 9
    let mut rng = Rng::new(7);
    let pts = thomas(0.0005, 8.0, 4.0, 1000.0, 1000.0, &mut rng);
    let n = pts.len() as f64;
    assert!((3200.0..4800.0).contains(&n), "expected about 0.004 · 1000² = 4000 points, got {n}");
    assert!(pts.iter().all(|&(x, y)| (0.0..1000.0).contains(&x) && (0.0..1000.0).contains(&y)), "all points must lie inside the window");
    let (_, _, vmr, _) = dispersion(&quadrat_counts(&pts, 1000.0, 1000.0, 20));
    assert!((5.0..13.0).contains(&vmr), "50 µm quadrats: VMR {vmr:.2}, expected about 1 + μ = 9");
    let (_, _, vmr, _) = dispersion(&quadrat_counts(&pts, 1000.0, 1000.0, 500));
    assert!(vmr < 1.6, "2 µm quadrats: VMR {vmr:.2}, expected close to 1");
}

#[test]
fn c4_intensity_with_edge_correction() {
    // points every 5 µm: ρ = 1/25 = 0.04 per µm² everywhere, also in the corners
    let mut pts = Vec::new();
    for i in 0..20 {
        for j in 0..20 {
            pts.push((2.5 + 5.0 * i as f64, 2.5 + 5.0 * j as f64));
        }
    }
    let map = intensity_map(&pts, 100.0, 100.0, 10.0);
    assert_eq!((map.width(), map.height()), (100, 100), "one pixel per µm");
    for (x, y) in [(50, 50), (0, 0), (99, 0), (0, 60), (99, 99)] {
        let v = map[(x, y)] as f64;
        assert!(close(v, 0.04, 0.002), "pixel ({x}, {y}): got {v:.4}, expected 0.04 (without the edge correction a corner gets 0.01)");
    }
    let total: f64 = map.pixels().iter().map(|&v| v as f64).sum();
    assert!(close(total, 400.0, 10.0), "the map should add up to about the number of points (400), got {total:.1}");
}
