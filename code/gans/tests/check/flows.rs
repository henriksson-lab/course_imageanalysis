// Checks for the coding exercises of "Normalising flows", with the numbers of the
// pen-and-paper exercises. Run: cargo test -p gans flows::

use gans::flows::run::banana;
use gans::flows::*;
use imgcourse::Rng;

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() < tol
}

fn b3() -> Coupling {
    Coupling { d: 1, s: |_| 2f64.ln(), t: |_| 1.0 }
}

#[test]
fn c1_forward_b3() {
    let (y, s) = coupling_forward(&b3(), [0.5, 1.0]);
    assert!(close(y[0], 0.5, 1e-12) && close(y[1], 3.0, 1e-12), "B3: (0.5, 1) -> (0.5, 3); got {y:?}");
    assert!(close(s, 2f64.ln(), 1e-12), "B3: log det = ln 2; got {s}");
    let l = Coupling { d: 0, s: |x1| 0.5 * x1, t: |x1| x1 * x1 };
    let (y, s) = coupling_forward(&l, [1.0, 2.0]);
    assert!(close(y[0], 1f64.exp() + 4.0, 1e-12) && close(y[1], 2.0, 1e-12) && close(s, 1.0, 1e-12), "d = 0 changes x0: got {y:?}, {s}");
}

#[test]
fn c2_inverse() {
    assert_eq!(coupling_inverse(&b3(), [0.5, 3.0]), [0.5, 1.0], "B3 backwards");
    let mut rng = Rng::new(3);
    for l in banana() {
        for _ in 0..50 {
            let x = [rng.normal() as f64, rng.normal() as f64];
            let back = coupling_inverse(&l, coupling_forward(&l, x).0);
            assert!(close(back[0], x[0], 1e-12) && close(back[1], x[1], 1e-12), "inverse(forward(x)) must give x back: {x:?} -> {back:?}");
        }
    }
}

#[test]
fn c3_density_b3_and_integral() {
    // no layers: the standard normal
    let p0 = log_density(&[], [0.0, 0.0]);
    assert!(close(p0, -(2.0 * std::f64::consts::PI).ln(), 1e-12), "empty flow at 0: -ln 2 pi; got {p0}");
    // B3: p(0.5, 1) = N(0.5, 3) * 2
    let p = log_density(&[b3()], [0.5, 1.0]).exp();
    assert!(close(p, 0.003121, 1e-6), "B3: p(0.5, 1) = 2 N((0.5, 3)) = 0.00312; got {p}");
    let flow = banana();
    let (h, mut s) = (0.05, 0.0);
    for i in 0..280 {
        for j in 0..340 {
            s += log_density(&flow, [-7.0 + (i as f64 + 0.5) * h, -5.0 + (j as f64 + 0.5) * h]).exp() * h * h;
        }
    }
    assert!(close(s, 1.0, 1e-3), "the density of the banana flow must integrate to 1; got {s}");
}

#[test]
fn c4_sample_matches_density() {
    let flow = banana();
    let mut rng = Rng::new(8);
    let n = 40000;
    let mut inside = 0;
    for _ in 0..n {
        let x = sample(&flow, &mut rng);
        if x[0].abs() < 1.0 && x[1].abs() < 1.0 {
            inside += 1;
        }
    }
    let (h, mut mass) = (0.02, 0.0);
    for i in 0..100 {
        for j in 0..100 {
            mass += log_density(&flow, [-1.0 + (i as f64 + 0.5) * h, -1.0 + (j as f64 + 0.5) * h]).exp() * h * h;
        }
    }
    let share = inside as f64 / n as f64;
    assert!(close(share, mass, 0.01), "share of samples in the square [-1, 1]^2 ({share:.3}) should match the integral of p there ({mass:.3})");
}
