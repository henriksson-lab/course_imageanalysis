// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p day50 whitted::

use day50::whitted::*;

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

fn close_vec(a: Vec3, b: Vec3) -> bool {
    (a - b).length() < 1e-3
}

const S: f32 = std::f32::consts::FRAC_1_SQRT_2;

#[test]
fn c1_reflect_b1() {
    let r = reflect(v3(S, -S, 0.0), v3(0.0, 1.0, 0.0));
    assert!(close_vec(r, v3(S, S, 0.0)), "B1, the floor: got {r:?}, expected (0.707, 0.707, 0)");
    let r = reflect(v3(0.0, 0.0, -1.0), v3(0.0, S, S));
    assert!(close_vec(r, v3(0.0, 1.0, 0.0)), "B1, the mirror at 45°: got {r:?}, expected (0, 1, 0)");
}

/// The unit direction that hits a floor with normal (0, 1, 0) at `deg` degrees from the normal.
fn at_angle(deg: f32) -> Vec3 {
    let a = deg.to_radians();
    v3(a.sin(), -a.cos(), 0.0)
}

#[test]
fn c2_refract_b10_b2_b3() {
    let up = v3(0.0, 1.0, 0.0);
    let t = refract(at_angle(30.0), up, 1.0 / 1.5).expect("air into glass always refracts");
    assert!(close_vec(t, v3(0.3333, -0.9428, 0.0)), "B10 (d), air → glass at 30°: got {t:?}, expected (0.333, -0.943, 0)");
    assert!(close(t.length(), 1.0, 1e-4), "the refracted direction must have length 1, got {}", t.length());
    // B2 (c): glass → air at 30°: θ2 = 48.6°.
    let t = refract(at_angle(30.0), up, 1.5).expect("glass → air at 30° is below the critical angle");
    let theta2 = t.x.asin().to_degrees();
    assert!(close(theta2, 48.6, 0.1), "B2 (c): got θ2 = {theta2}°, expected 48.6°");
    // B3 (a): the critical angle glass → air is 41.8°.
    assert!(refract(at_angle(41.0), up, 1.5).is_some(), "B3: glass → air at 41° still refracts");
    assert!(refract(at_angle(42.5), up, 1.5).is_none(), "B3: glass → air at 42.5° is total internal reflection");
}

#[test]
fn c3_schlick_b4_b5() {
    let r0 = schlick(1.0, 1.0, 1.5);
    assert!(close(r0, 0.04, 1e-4), "B4 (a), air–glass head-on: got {r0}, expected 0.04");
    let r0 = schlick(1.0, 1.0, 2.42);
    assert!(close(r0, 0.172, 1e-3), "B4 (e), air–diamond: got {r0}, expected 0.172");
    let r = schlick(60f32.to_radians().cos(), 1.0, 1.5);
    assert!(close(r, 0.070, 1e-3), "B5, 60°: got {r}, expected 0.070");
    let r = schlick(85f32.to_radians().cos(), 1.0, 1.5);
    assert!(close(r, 0.648, 2e-3), "B5, 85°: got {r}, expected 0.648");
    let r = schlick(45f32.to_radians().cos(), 1.5, 1.0);
    assert!(close(r, 1.0, 1e-6), "glass → air at 45° (beyond 41.8°): got {r}, expected 1");
    let r = schlick(1.0, 1.5, 1.0);
    assert!(close(r, 0.04, 1e-4), "glass → air head-on: got {r}, expected 0.04, the same as air → glass");
}

fn floor(material: Material) -> Object {
    Object::new(Shape::Plane { point: v3(0.0, 0.0, 0.0), normal: v3(0.0, 1.0, 0.0) }, material, [0.5, 0.5, 0.5])
}

fn scene(objects: Vec<Object>) -> Scene {
    Scene { objects, light: v3(0.0, 10.0, 0.0), sky: [0.2, 0.4, 0.8], ambient: 0.1 }
}

#[test]
fn c4_shadow_and_leak_b8() {
    // A small ball whose top is 0.12 above the floor, right above the point p (B8 (c)).
    let ball = Object::new(Shape::Sphere { centre: v3(0.0, 0.07, 0.0), radius: 0.05 }, Material::Diffuse, [1.0; 3]);
    let s = scene(vec![floor(Material::Diffuse), ball]);
    let (p, n) = (v3(0.0, 0.0, 0.0), v3(0.0, 1.0, 0.0));
    assert!(in_shadow(p, n, s.light, &s, 1e-4), "the ball hides the light from the point under it");
    assert!(!in_shadow(p, n, s.light, &s, 0.2), "with eps = 0.2 the shadow ray starts above the ball: light leaks");
    assert!(!in_shadow(v3(3.0, 0.0, 0.0), n, s.light, &s, 1e-4), "a point beside the ball sees the light");
    // An object beyond the light does not cast a shadow.
    let high = Object::new(Shape::Sphere { centre: v3(0.0, 20.0, 0.0), radius: 1.0 }, Material::Diffuse, [1.0; 3]);
    let s = scene(vec![floor(Material::Diffuse), high]);
    assert!(!in_shadow(p, n, s.light, &s, 1e-4), "a ball above the light must not shadow the floor");
}

#[test]
fn c5_trace_sky_and_floor() {
    let s = scene(vec![floor(Material::Diffuse)]);
    let mut rays = 0;
    let c = trace(Ray::new(v3(0.0, 1.0, 0.0), v3(0.0, 1.0, 0.0)), &s, 0, 5, 1e-4, &mut rays);
    assert_eq!(c, s.sky, "a ray that hits nothing sees the sky");
    assert_eq!(rays, 1, "one ray for the sky, got {rays}");
    let mut rays = 0;
    let c = trace(Ray::new(v3(0.0, 1.0, 0.0), v3(0.0, -1.0, 0.0)), &s, 0, 5, 1e-4, &mut rays);
    assert!(close(c[0], 0.55, 1e-4), "a lit matte floor under the light: got {c:?}, expected 0.5 · (0.1 + 1) = 0.55");
    assert_eq!(rays, 2, "the primary ray and one shadow ray, got {rays}");
}

#[test]
fn c5_trace_mirror_depth() {
    let s = scene(vec![floor(Material::Mirror(0.8))]);
    let ray = Ray::new(v3(-1.0, 1.0, 0.0), v3(S, -S, 0.0));
    let mut rays = 0;
    let c = trace(ray, &s, 0, 1, 1e-4, &mut rays);
    let expected = s.sky.map(|v| 0.8 * v);
    assert!(close(c[2], expected[2], 1e-5), "the mirror shows the sky, times 0.8: got {c:?}, expected {expected:?}");
    assert_eq!(rays, 2, "the primary and the reflected ray, got {rays}");
    let mut rays = 0;
    let c = trace(ray, &s, 0, 0, 1e-4, &mut rays);
    assert_eq!(c, [0.0; 3], "at the depth limit a mirror is black (depth 0 = ray casting)");
}

#[test]
fn c5_glass_ball_shows_the_sky() {
    // R + T = 1 at every split: a glass ball alone under the sky shows the sky (nearly
    // all of it; only what is left at the depth limit is lost). At depth 0 it is black (E6).
    let ball = Object::new(Shape::Sphere { centre: v3(0.0, 0.0, -5.0), radius: 1.0 }, Material::Glass(1.5), [0.0; 3]);
    let s = scene(vec![ball]);
    let ray = Ray::new(v3(0.0, 0.0, 0.0), v3(0.1, 0.05, -1.0).normalize());
    let mut rays = 0;
    let c = trace(ray, &s, 0, 8, 1e-4, &mut rays);
    assert!(close(c[2], s.sky[2], 0.01), "through a glass ball in an empty sky: got {c:?}, expected about {:?}", s.sky);
    assert!(rays > 3, "glass splits the ray: got only {rays} rays");
    let mut rays = 0;
    let c = trace(ray, &s, 0, 0, 1e-4, &mut rays);
    assert_eq!(c, [0.0; 3], "at depth 0 a glass ball is black");
}
