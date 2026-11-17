// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p day50 pathtrace::

use imgcourse::Rng;
use day50::pathtrace::*;

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

#[test]
fn c1_monte_carlo_b1() {
    let f = |x: f32| x * x;
    let uniform = mc_estimate(f, |_| 1.0, &[0.1, 0.4, 0.6, 0.9]);
    assert!(close(uniform, 0.335, 1e-4), "B1 (a), uniform samples: got {uniform}, expected 0.335");
    let xs: Vec<f32> = [0.1f32, 0.4, 0.6, 0.9].iter().map(|u| u.sqrt()).collect();
    let importance = mc_estimate(f, |x| 2.0 * x, &xs);
    assert!(close(importance, 0.334, 1e-3), "B1 (b), samples from p(x) = 2x: got {importance}, expected 0.334");
}

#[test]
fn c2_cosine_directions() {
    let n = v3(0.0, 0.6, 0.8);
    let mut rng = Rng::new(1);
    let (mut sum_cos, count) = (0.0, 20000);
    for _ in 0..count {
        let w = cosine_direction(n, rng.uniform(), rng.uniform());
        assert!(close(w.length(), 1.0, 1e-4), "a direction must have length 1, got {}", w.length());
        assert!(w.dot(n) >= -1e-6, "the direction {w:?} points below the surface");
        sum_cos += w.dot(n);
    }
    // For the density cos θ / π, the mean of cos θ is 2/3 (uniform directions give 1/2).
    let mean = sum_cos / count as f32;
    assert!(close(mean, 2.0 / 3.0, 0.01), "mean cos θ of cosine-weighted directions: got {mean}, expected 2/3");
    let up = cosine_direction(n, 0.0, 0.3);
    assert!((up - n).length() < 1e-4, "u1 = 0 is the centre of the disc: straight along the normal, got {up:?}");
}

#[test]
fn c3_direct_light_b10() {
    // A matte point (ρ = 0.7) 1.5 below the centre of a square lamp of side 0.5, L_e = 12.
    let d = direct_light(v3(0.0, 0.0, 0.0), v3(0.0, 1.0, 0.0), [0.7; 3], v3(0.0, 1.5, 0.0), v3(0.0, -1.0, 0.0), [12.0; 3], 0.25);
    assert!(close(d[0], 0.297, 1e-3), "B10 (b): got {}, expected 0.297", d[0]);
    let behind = direct_light(v3(0.0, 0.0, 0.0), v3(0.0, -1.0, 0.0), [0.7; 3], v3(0.0, 1.5, 0.0), v3(0.0, -1.0, 0.0), [12.0; 3], 0.25);
    assert_eq!(behind, [0.0; 3], "a lamp behind the surface gives no light");
}

/// The "furnace": the camera inside a ball that emits L_e = 1 and reflects ρ = 0.5. Every
/// path gets 1 + ρ + ρ² + ... , whatever directions it takes.
fn furnace() -> Scene {
    Scene { objects: vec![Object { shape: Shape::Sphere { centre: v3(0.0, 0.0, 0.0), radius: 1.0 }, albedo: [0.5; 3], emission: [1.0; 3] }] }
}

#[test]
fn c4_furnace() {
    let scene = furnace();
    let mut rng = Rng::new(2);
    for (bounces, expected) in [(0, 1.0), (1, 1.5), (3, 1.875)] {
        for _ in 0..20 {
            let ray = Ray::new(v3(0.1, 0.2, 0.0), v3(0.0, 0.6, -0.8));
            let l = radiance(ray, &scene, &mut rng, bounces, false);
            assert!(close(l[0], expected, 1e-4), "inside an emitting ball (L_e = 1, ρ = 0.5), {bounces} bounces: got {}, expected {expected}", l[0]);
        }
    }
}

#[test]
fn c4_light_sampling_is_unbiased() {
    // The same pixel with and without lamp sampling: the same mean, less noise.
    let scene = cornell_box(0.25, 3.0);
    let ray = camera_ray(v3(0.0, 1.0, 2.9), 40.0, 90.0, 80, 60, 40.0); // a point on the floor
    let (n, mut rng) = (20000, Rng::new(3));
    let mut stats = Vec::new();
    for light_sampling in [false, true] {
        let (mut sum, mut sum2) = (0.0f64, 0.0f64);
        for _ in 0..n {
            let l = radiance(ray, &scene, &mut rng, 2, light_sampling)[0] as f64;
            sum += l;
            sum2 += l * l;
        }
        let mean = sum / n as f64;
        stats.push((mean, (sum2 / n as f64 - mean * mean).sqrt()));
    }
    let ((m0, s0), (m1, s1)) = (stats[0], stats[1]);
    let tolerance = 4.0 * s0 / (n as f64).sqrt();
    assert!((m0 - m1).abs() < tolerance, "cosine sampling gives {m0:.3}, lamp sampling {m1:.3}: they should agree (to {tolerance:.3})");
    assert!(s1 < s0, "lamp sampling should have less noise: σ {s1:.3} against {s0:.3}");
}

#[test]
fn c5_render() {
    let img = render(&furnace(), v3(0.0, 0.0, 0.0), 6, 4, Settings { spp: 3, max_bounces: 1, light_sampling: false, seed: 1 });
    assert_eq!((img.width(), img.height()), (6, 4), "the image must have the requested size");
    for (x, y) in img.coords() {
        assert!(close(img[(x, y)][1], 1.5, 1e-4), "pixel ({x}, {y}) inside the furnace: got {:?}, expected 1.5", img[(x, y)]);
    }
}
