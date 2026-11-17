// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p raycast

use raycast::*;

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

fn close_vec(a: Vec3, b: Vec3) -> bool {
    (a - b).length() < 1e-3
}

#[test]
fn c1_camera_rays_b1() {
    // 4 × 2 pixels, 90° vertical field of view.
    let r = camera_ray(1, 0, 4, 2, 90.0);
    assert!(close_vec(r.dir, v3(-0.408, 0.408, -0.816)), "pixel (1, 0): got {:?}, expected (-0.408, 0.408, -0.816)", r.dir);
    assert!(close_vec(r.origin, v3(0.0, 0.0, 0.0)), "the ray starts at the eye, the origin: got {:?}", r.origin);
    let r = camera_ray(3, 1, 4, 2, 90.0);
    assert!(close_vec(r.dir, v3(0.802, -0.267, -0.535)), "pixel (3, 1): got {:?}, expected (0.802, -0.267, -0.535)", r.dir);
}

#[test]
fn c2a_roots_b5_and_b7() {
    // B5: roots -1.2 and 3.5 (b = -(3.5 - 1.2)/2, k = -1.2 · 3.5): the eye is inside.
    let t = smaller_positive_root(-1.15, -4.2).unwrap_or(f32::NAN);
    assert!(close(t, 3.5, 1e-4), "roots -1.2 and 3.5: got {t}, expected 3.5");
    // B7: b = -10^4, k = 1. The textbook formula gives 0 in float32; the stable one 5e-5.
    let t = smaller_positive_root(-1e4, 1.0).unwrap_or(f32::NAN);
    assert!(close(t, 5.0e-5, 1e-9), "B7, b = -10^4, k = 1: got {t:e}, expected 5e-5 (compute the large root first)");
    assert!(smaller_positive_root(0.0, 1.0).is_none(), "b² - k < 0 must be a miss");
    assert!(smaller_positive_root(3.0, 5.0).is_none(), "roots -1 and -5 are both behind the eye");
}

#[test]
fn c2b_sphere_b2() {
    let ray = Ray::new(v3(0.0, 0.0, 0.0), v3(0.0, 0.0, -1.0));
    let t = hit_sphere(ray, v3(0.5, 0.0, -5.0), 1.0).unwrap_or(f32::NAN);
    assert!(close(t, 4.134, 1e-3), "B2 (b): got t = {t}, expected 4.134");
    assert!(hit_sphere(ray, v3(1.5, 0.0, -5.0), 1.0).is_none(), "B2 (d): the moved sphere must be a miss");
}

#[test]
fn c3_plane_b3() {
    let floor = (v3(0.0, -1.0, 0.0), v3(0.0, 1.0, 0.0));
    let eye = v3(0.0, 0.0, 0.0);
    assert!(hit_plane(Ray::new(eye, v3(0.0, 0.0, -1.0)), floor.0, floor.1).is_none(), "B3 (a): parallel to the floor, no hit");
    let t = hit_plane(Ray::new(eye, v3(0.0, -0.6, -0.8)), floor.0, floor.1).unwrap_or(f32::NAN);
    assert!(close(t, 1.6667, 1e-3), "B3 (b): got t = {t}, expected 1.667");
    assert!(hit_plane(Ray::new(eye, v3(0.0, 0.6, -0.8)), floor.0, floor.1).is_none(), "B3 (c): the floor is behind the eye");
}

#[test]
fn c4_triangle_b8() {
    let (a, b, c) = (v3(0.0, 0.0, -2.0), v3(1.0, 0.0, -2.0), v3(0.0, 1.0, -2.0));
    let ray = Ray::new(v3(0.0, 0.0, 0.0), v3(0.2, 0.2, -1.0));
    let (t, u, v) = hit_triangle(ray, a, b, c).unwrap_or((f32::NAN, f32::NAN, f32::NAN));
    assert!(close(t, 2.0, 1e-4) && close(u, 0.4, 1e-4) && close(v, 0.4, 1e-4), "B8: got (t, u, v) = ({t}, {u}, {v}), expected (2, 0.4, 0.4)");
    // Outside: u + v > 1 (the point (0.8, 0.8) on the plane).
    assert!(hit_triangle(Ray::new(v3(0.0, 0.0, 0.0), v3(0.4, 0.4, -1.0)), a, b, c).is_none(), "the point (0.8, 0.8, -2) is outside the triangle");
    // Behind the eye.
    assert!(hit_triangle(Ray::new(v3(0.0, 0.0, 0.0), v3(-0.2, -0.2, 1.0)), a, b, c).is_none(), "a triangle behind the eye is not hit");
}

#[test]
fn c5_nearest_hit_b5() {
    // B5 along the ray o = 0, d = (0, 0, -1): a sphere around the eye with roots -1.2 and 3.5
    // (centre (0, 0, -1.15), radius 2.35), a plane at t = 7 and a triangle whose plane is at
    // t = 2.1 but which lies beside the ray.
    let scene = [
        Shape::Plane { point: v3(0.0, 0.0, -7.0), normal: v3(0.0, 0.0, 1.0) },
        Shape::Triangle { a: v3(1.0, 0.0, -2.1), b: v3(2.0, 0.0, -2.1), c: v3(1.0, 1.0, -2.1) },
        Shape::Sphere { centre: v3(0.0, 0.0, -1.15), radius: 2.35 },
    ];
    let ray = Ray::new(v3(0.0, 0.0, 0.0), v3(0.0, 0.0, -1.0));
    let hit = nearest_hit(ray, &scene).expect("B5: the ray must hit something");
    assert_eq!(hit.object, 2, "B5: the pixel shows the sphere (object 2), got object {}", hit.object);
    assert!(close(hit.t, 3.5, 1e-3), "B5: got t = {}, expected 3.5", hit.t);
    // Inside the sphere the outward normal points away from the eye: it must be flipped.
    assert!(close_vec(hit.normal, v3(0.0, 0.0, 1.0)), "the normal must face the eye: got {:?}, expected (0, 0, 1)", hit.normal);
    let miss = nearest_hit(Ray::new(v3(0.0, 0.0, 0.0), v3(0.0, 1.0, 0.0)), &scene[..2]);
    assert!(miss.is_none(), "a ray that hits nothing must give None");
}
