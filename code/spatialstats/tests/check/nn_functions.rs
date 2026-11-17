// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p spatialstats nn_functions::

use imgcourse::Rng;
use spatialstats::nn_functions::*;

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

/// The six points of B1, A to F, in a 10 × 10 µm window.
const B1: [Point; 6] = [(2.0, 2.0), (3.0, 2.0), (6.0, 3.0), (6.0, 6.0), (2.0, 7.0), (9.0, 8.0)];

#[test]
fn c1_nn_distances_b1() {
    let d = nn_distances(&B1);
    let expected = [1.0, 1.0, 3.0, 3.0, 17f64.sqrt(), 13f64.sqrt()];
    for (k, name) in ["A", "B", "C", "D", "E", "F"].iter().enumerate() {
        assert!(close(d[k], expected[k], 1e-9), "{name}: nearest neighbour at {:.3}, expected {:.3}", d[k], expected[k]);
    }
}

#[test]
fn c2_g_hat_b1_b4() {
    for (r, expected) in [(1.0, 2.0 / 6.0), (2.0, 2.0 / 6.0), (3.0, 4.0 / 6.0), (4.0, 5.0 / 6.0)] {
        let g = g_hat(&B1, 10.0, 10.0, r, false);
        assert!(close(g, expected, 1e-9), "B1(b), no correction, r = {r}: got {g:.3}, expected {expected:.3}");
    }
    for (r, expected) in [(1.0, 2.0 / 6.0), (2.0, 2.0 / 5.0), (3.0, 1.0)] {
        let g = g_hat(&B1, 10.0, 10.0, r, true);
        assert!(close(g, expected, 1e-9), "B4, border rule, r = {r}: got {g:.3}, expected {expected:.3}");
    }
    assert!(g_hat(&B1, 10.0, 10.0, 6.0, true).is_nan(), "B4: no point is 6 µm from the border, so G cannot be estimated (NaN)");
}

#[test]
fn c3_f_hat_b3() {
    let points = [(1.0, 1.0), (4.0, 2.0), (8.0, 2.0), (5.0, 6.0), (2.0, 8.0)];
    let tests = [(2.0, 2.0), (6.0, 2.0), (9.0, 5.0), (5.0, 4.0), (3.0, 5.0), (7.0, 8.0), (1.0, 5.0), (9.0, 9.0)];
    for (r, expected) in [(1.0, 0.0), (2.0, 3.0 / 8.0), (3.0, 5.0 / 8.0)] {
        let f = f_hat(&points, &tests, 10.0, 10.0, r, false);
        assert!(close(f, expected, 1e-9), "B3(b), r = {r}: got {f:.3}, expected {expected:.3}");
    }
}

#[test]
fn c4_j_reads_the_pattern() {
    let mut rng = Rng::new(3);
    let regular = hard_core(150, 10.0, 200.0, 200.0, &mut rng);
    let clustered = thomas(0.001, 6.0, 4.0, 200.0, 200.0, &mut rng);
    let random = csr(300, 200.0, 200.0, &mut rng);
    let (jr, jc, jcsr) = (j_hat(&regular, 200.0, 200.0, 6.0, 2.0), j_hat(&clustered, 200.0, 200.0, 6.0, 2.0), j_hat(&random, 200.0, 200.0, 4.0, 2.0));
    assert!(jr > 1.2, "hard core 10 µm: J(6) = {jr:.2}, expected well above 1");
    assert!(jc < 0.5, "clusters: J(6) = {jc:.2}, expected well below 1");
    assert!((0.7..1.4).contains(&jcsr), "CSR: J(4) = {jcsr:.2}, expected about 1");
}
