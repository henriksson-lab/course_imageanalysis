// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p graphics3d volrender::

use imgcourse::Volume;
use graphics3d::volrender::*;

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

/// The transfer function of B1: value → (colour, opacity).
fn tf(v: f32) -> (Rgb, f32) {
    if v < 50.0 {
        ([0.0; 3], 0.0)
    } else if v < 150.0 {
        ([0.9, 0.6, 0.5], 0.2)
    } else if v < 210.0 {
        ([1.0, 1.0, 0.9], 0.5)
    } else {
        ([1.0, 0.2, 0.2], 0.9)
    }
}

fn b1_samples() -> Vec<(Rgb, f32)> {
    [30.0, 80.0, 180.0, 230.0, 90.0].iter().map(|&v| tf(v)).collect()
}

#[test]
fn r1_trilinear() {
    // a linear function of the position is reproduced exactly between the voxels
    let vol = Volume::from_fn(4, 5, 6, |x, y, z| (x + 2 * y + 3 * z) as f32);
    for (x, y, z) in [(1.0, 2.0, 3.0), (1.25, 2.5, 3.75), (0.1, 3.9, 4.5), (2.99, 0.01, 0.5)] {
        let got = trilinear(&vol, x, y, z);
        let want = x + 2.0 * y + 3.0 * z;
        assert!(close(got, want, 1e-4), "at ({x}, {y}, {z}): got {got}, expected {want}");
    }
    // halfway between a voxel of 1 and the outside (0)
    let one = Volume::new(2, 2, 2, 1.0f32);
    assert!(close(trilinear(&one, 1.5, 0.5, 0.5), 0.5, 1e-6), "halfway to the outside: got {}, expected 0.5", trilinear(&one, 1.5, 0.5, 0.5));
    assert!(close(trilinear(&one, 5.0, 0.0, 0.0), 0.0, 1e-6), "far outside: expected 0");
}

#[test]
fn r2_sample_ray_with_spacing() {
    // value = z index; voxels 2 mm apart along z
    let vol = Volume::from_fn(3, 3, 10, |_, _, z| z as f32);
    let v = sample_ray(&vol, [1.0, 1.0, 2.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0], 1.0, 5);
    let want = [0.0, 0.5, 1.0, 1.5, 2.0];
    for k in 0..5 {
        assert!(close(v[k], want[k], 1e-5), "sample {k} (z = {k} mm): got {}, expected {} (z = {k} mm is voxel {})", v[k], want[k], want[k]);
    }
}

#[test]
fn r3_front_to_back_b1() {
    let (c, t, used) = front_to_back(&b1_samples(), 0.0);
    let want = [0.9472, 0.5968, 0.536];
    assert!((0..3).all(|k| close(c[k], want[k], 1e-4)), "C after all 5 samples: got {c:?}, expected {want:?}");
    assert!(close(t, 0.032, 1e-5) && used == 5, "T after all samples: got {t} ({used} samples used), expected 0.032 (5)");
    // early termination at ε = 0.05 (B1 d)
    let (c, t, used) = front_to_back(&b1_samples(), 0.05);
    let want = [0.94, 0.592, 0.532];
    assert_eq!(used, 4, "early termination: got {used} samples used, expected 4");
    assert!((0..3).all(|k| close(c[k], want[k], 1e-4)) && close(t, 0.04, 1e-5), "C and T after sample 4: got {c:?}, {t}, expected {want:?}, 0.04");
}

#[test]
fn r3_back_to_front_b1() {
    let c = back_to_front(&b1_samples());
    let want = [0.9472, 0.5968, 0.536];
    assert!((0..3).all(|k| close(c[k], want[k], 1e-4)), "back to front: got {c:?}, expected the same colour as front to back, {want:?}");
    // only the last two samples: (0.918, 0.192, 0.190) (B1 c, after sample 4)
    let c = back_to_front(&b1_samples()[3..]);
    let want = [0.918, 0.192, 0.190];
    assert!((0..3).all(|k| close(c[k], want[k], 1e-3)), "samples 4 and 5 alone: got {c:?}, expected {want:?}");
}

#[test]
fn r4_opacity_correction_b2_b3() {
    let a0 = 1.0 - (-0.5f32).exp(); // 0.393
    let half = opacity_correct(a0, 0.5);
    assert!(close(half, 0.221, 1e-3), "B2: half a step: got {half}, expected 0.221");
    assert!(close(1.0 - (1.0 - half).powi(2), a0, 1e-5), "B2: two half steps must let through as much light as one step");
    // B3: a slab of 10 voxels with τ = 0.3: opacity 0.950 for any step, with the correction
    let a0 = 1.0 - (-0.3f32).exp();
    for (step, n) in [(2.0, 5), (0.5, 20), (1.0, 10)] {
        let a = opacity_correct(a0, step);
        let slab = 1.0 - (1.0 - a).powi(n);
        assert!(close(slab, 0.950, 1e-3), "B3: step {step} ({n} samples): got slab opacity {slab}, expected 0.950");
    }
}

#[test]
fn r5_iso_crossing_b6() {
    let t = iso_crossing(&[20.0, 50.0, 70.0, 80.0, 140.0, 90.0], 100.0);
    assert!(t.is_some_and(|t| close(t, 3.0 + 1.0 / 3.0, 1e-5)), "crossing of 100 between 80 (sample 3) and 140 (sample 4): got {t:?}, expected 3.333");
    assert_eq!(iso_crossing(&[1.0, 2.0, 3.0], 10.0), None, "a ray that never reaches the isovalue");
    let t = iso_crossing(&[200.0, 50.0, 150.0], 100.0);
    assert!(t.is_some_and(|t| close(t, 1.5, 1e-5)), "only crossings from below count: got {t:?}, expected 1.5");
}
