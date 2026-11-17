// Checks for the coding exercises of "Few labels", using the numbers of the pen-and-paper
// exercises. Run: cargo test -p finetuning few_labels::

use finetuning::few_labels::*;

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

#[test]
fn c1_uncertainty() {
    for (p, e) in [(0.5, 1.0), (0.7, 0.6), (0.9, 0.2), (0.1, 0.2), (0.99, 0.02), (0.0, 0.0)] {
        assert!(close(uncertainty(p), e, 1e-5), "uncertainty({p}): got {}, expected {e}", uncertainty(p));
    }
}

#[test]
fn c2_most_uncertain() {
    let probs = [0.9, 0.45, 0.6, 0.55, 0.02];
    assert_eq!(most_uncertain(&probs, &[false; 5]), Some(1), "0.45 is closest to 0.5 (ties go to the lowest index: 0.45 and 0.55)");
    assert_eq!(most_uncertain(&probs, &[false, true, false, false, false]), Some(3), "0.45 is labelled: next 0.55");
    assert_eq!(most_uncertain(&probs, &[true; 5]), None, "nothing left");
}

#[test]
fn c3_most_different() {
    let c = |x: f32, y: f32| Cell { x, y, class: 0, rare: false };
    let cells = vec![c(0.0, 0.0), c(0.1, 0.0), c(1.0, 1.0), c(0.5, 0.5), c(0.0, 0.9)];
    let lab = [true, false, false, false, false];
    assert_eq!(most_different(&cells, &lab), Some(2), "(1, 1) is farthest from (0, 0)");
    let lab = [true, false, true, false, false];
    assert_eq!(most_different(&cells, &lab), Some(4), "with (0, 0) and (1, 1) labelled, (0, 0.9) is farthest from both");
}

#[test]
fn c4_entropy() {
    for (p, e) in [(0.5, 1.0), (0.9, 0.469), (0.99, 0.0808), (1.0, 0.0), (0.0, 0.0)] {
        assert!(close(entropy(p), e, 1e-3), "entropy({p}): got {}, expected {e}", entropy(p));
    }
}
