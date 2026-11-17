// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p training normalisation::

use imgcourse::Rng;
use training::normalisation::*;

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() <= tol
}

#[test]
fn c1_init_weights_spread() {
    let mut rng = Rng::new(5);
    let w = init_weights(100, 500, 2f32.sqrt(), &mut rng);
    assert_eq!(w.len(), 100 * 500, "fan_in * fan_out weights");
    let mean = w.iter().sum::<f32>() / w.len() as f32;
    let sd = (w.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / w.len() as f32).sqrt();
    // B1: gain sqrt 2 and fan-in 100: sd sqrt(2/100) = 0.1414
    assert!(close(mean, 0.0, 0.003) && close(sd, 0.1414, 0.003), "B1: mean {mean}, sd {sd}; expected 0 and 0.1414");
}

#[test]
fn c2_batch_norm_by_hand_b4() {
    // B4: three examples, two units
    let mut z = vec![vec![1.0, 10.0], vec![3.0, 10.0], vec![5.0, 40.0]];
    let (means, vars) = batch_norm(&mut z, 0.0);
    assert!(close(means[0], 3.0, 1e-5) && close(means[1], 20.0, 1e-4), "B4: means {means:?}, expected [3, 20]");
    assert!(close(vars[0], 8.0 / 3.0, 1e-4) && close(vars[1], 200.0, 1e-3), "B4: variances {vars:?}, expected [2.667, 200]");
    let (r, h) = (std::f32::consts::FRAC_1_SQRT_2, std::f32::consts::SQRT_2);
    let want = [[-1.2247, -r], [0.0, -r], [1.2247, h]];
    for n in 0..3 {
        for j in 0..2 {
            assert!(close(z[n][j], want[n][j], 1e-3), "B4: normalised batch {z:?}, expected {want:?}");
        }
    }
}

#[test]
fn c2_batch_of_one_is_all_zero_b5() {
    // B5: a batch of one example: every value minus its own mean is 0
    let mut z = vec![vec![0.3, 0.8, 0.5]];
    batch_norm(&mut z, 1e-5);
    assert!(z[0].iter().all(|v| v.abs() < 1e-6), "B5: batch of one gives {z:?}, expected all 0");
}

#[test]
fn c3_layer_norm_by_hand_b4() {
    let mut z = vec![1.0, 3.0, 5.0];
    layer_norm(&mut z, 0.0);
    let want = [-1.2247, 0.0, 1.2247];
    for j in 0..3 {
        assert!(close(z[j], want[j], 1e-3), "B4(c): layer norm of (1, 3, 5) gives {z:?}, expected {want:?}");
    }
    // the same example in a different "microscope" (x 2 + 7): the same output
    let mut z2 = vec![9.0, 13.0, 17.0];
    layer_norm(&mut z2, 0.0);
    assert!((0..3).all(|j| close(z2[j], want[j], 1e-3)), "B4(c): layer norm of (9, 13, 17) gives {z2:?}, expected {want:?}");
}

#[test]
fn c4_linear_stack_b2() {
    // B2: a linear stack multiplies the size by about the gain at every layer
    let s = signal_sizes(10, 64, 1.5, Act::Linear, Norm::None, 128, 1);
    assert_eq!(s.len(), 10, "one size per layer");
    let ratio = (s[9] / s[0]).powf(1.0 / 9.0);
    assert!(close(ratio, 1.5, 0.08), "B2: size grows by {ratio} per layer, expected about 1.5 ({s:?})");
    let s = signal_sizes(10, 64, 1.0, Act::Linear, Norm::None, 128, 1);
    assert!(s.iter().all(|&v| v > 0.6 && v < 1.6), "gain 1, linear: the size should stay near 1, got {s:?}");
}

#[test]
fn c4_relu_needs_sqrt2_b3() {
    // B3: ReLU halves the mean square; gain sqrt 2 keeps it, gain 1 shrinks by 1/sqrt 2 per layer
    let he = signal_sizes(20, 64, 2f32.sqrt(), Act::Relu, Norm::None, 128, 1);
    let one = signal_sizes(20, 64, 1.0, Act::Relu, Norm::None, 128, 1);
    assert!(he[19] > 0.3 && he[19] < 1.5, "B3: He initialisation, size after 20 layers {}, expected about 1", he[19]);
    assert!(one[19] < 0.01, "B3: gain 1 with ReLU, size after 20 layers {}, expected about 0.7^20 = 0.001", one[19]);
}

#[test]
fn c4_batch_norm_fixes_the_scale() {
    let s = signal_sizes(20, 64, 2.0, Act::Relu, Norm::Batch, 128, 1);
    // after batch norm each sum has mean 0 and sd 1; after the ReLU the root mean square is sqrt(1/2)
    assert!(s.iter().all(|&v| close(v, std::f32::consts::FRAC_1_SQRT_2, 0.03)), "batch norm: every layer should give about 0.707, got {s:?}");
}
