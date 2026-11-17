// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p multiview cameras::

use multiview::cameras::*;

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() < tol
}

const I3: Mat3 = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]];

/// The camera of the exercises: f = 800, (cx, cy) = (320, 240), at the origin looking down +z.
fn b_camera() -> Camera {
    Camera { f: 800.0, cx: 320.0, cy: 240.0, r: I3, t: [0.0; 3] }
}

/// A rotation by the angles (degrees) about x, then y, then z.
fn rotation(ax: f64, ay: f64, az: f64) -> Mat3 {
    let (sx, cx) = ax.to_radians().sin_cos();
    let (sy, cy) = ay.to_radians().sin_cos();
    let (sz, cz) = az.to_radians().sin_cos();
    let rx = [[1., 0., 0.], [0., cx, -sx], [0., sx, cx]];
    let ry = [[cy, 0., sy], [0., 1., 0.], [-sy, 0., cy]];
    let rz = [[cz, -sz, 0.], [sz, cz, 0.], [0., 0., 1.]];
    mat_mul(&rz, &mat_mul(&ry, &rx))
}

/// A hidden camera looking at two checkerboard walls (as in the lecture demo), tilted a little.
fn hidden() -> Camera {
    let (r, t) = look_at([0.55, -0.3, 0.6], [0.12, -0.05, 0.12], [0.0, 1.0, 0.0]);
    let r = mat_mul(&rotation(0.0, 0.0, 4.0), &r);
    let t = mat_vec(&rotation(0.0, 0.0, 4.0), t);
    Camera { f: 820.0, cx: 335.0, cy: 245.0, r, t }
}

/// The corners of two walls at right angles: 6 × 6 on the plane z = 0, 6 × 6 on x = 0.
fn walls() -> Vec<Vec3> {
    let mut pts = Vec::new();
    for i in 1..=6 {
        for j in 1..=6 {
            pts.push([0.05 * i as f64, 0.05 * j as f64 - 0.2, 0.0]);
            pts.push([0.0, 0.05 * j as f64 - 0.2, 0.05 * i as f64]);
        }
    }
    pts
}

#[test]
fn c1_project_b1() {
    let cam = b_camera();
    for (x, want) in [([0.5, -0.25, 5.0], [400.0, 200.0]), ([0.5, -0.25, 10.0], [360.0, 220.0]), ([0.4, 0.3, 2.0], [480.0, 360.0])] {
        let got = project(&cam, x).expect("the point is in front of the camera");
        assert!(close(got[0], want[0], 1e-9) && close(got[1], want[1], 1e-9), "{x:?}: got {got:?}, expected {want:?}");
    }
    assert!(project(&cam, [0.5, 0.2, -3.0]).is_none(), "a point behind the camera must give None");
    // with a rotation and translation: compare with P = K [R | t]
    let cam = hidden();
    let x = [0.1, 0.05, 0.2];
    let (a, b) = (project(&cam, x).unwrap(), project_p(&camera_matrix(&cam), x));
    assert!(close(a[0], b[0], 1e-9) && close(a[1], b[1], 1e-9), "R and t: got {a:?}, but K [R | t] gives {b:?}");
}

#[test]
fn c2_pixel_ray_b2() {
    let (c, d) = pixel_ray(&b_camera(), 400.0, 200.0);
    assert!(c.iter().all(|v| close(*v, 0.0, 1e-12)), "camera centre: got {c:?}, expected the origin");
    let want = [0.1, -0.05, 1.0];
    assert!((0..3).all(|k| close(d[k], want[k], 1e-12)), "direction: got {d:?}, expected {want:?}");
    // a general camera: every point on the ray projects back onto the pixel
    let cam = hidden();
    let (c, d) = pixel_ray(&cam, 123.0, 321.0);
    for s in [0.5, 1.0, 3.0] {
        let p = [c[0] + s * d[0], c[1] + s * d[1], c[2] + s * d[2]];
        let uv = project(&cam, p).expect("points on the ray are in front of the camera");
        assert!(close(uv[0], 123.0, 1e-6) && close(uv[1], 321.0, 1e-6), "s = {s}: the point projects to {uv:?}, expected (123, 321)");
    }
}

#[test]
fn c3_distortion_b4() {
    let barrel = distort(0.4, 0.0, -0.1, 0.0);
    assert!(close(barrel[0], 0.3936, 1e-12), "k1 = −0.1 at r = 0.4: got {}, expected 0.3936", barrel[0]);
    let pin = distort(0.4, 0.0, 0.1, 0.0);
    assert!(close(pin[0], 0.4064, 1e-12), "k1 = +0.1 at r = 0.4: got {}, expected 0.4064", pin[0]);
    for (x, y) in [(0.3, -0.2), (-0.45, 0.3), (0.0, 0.0)] {
        let d = distort(x, y, -0.3, 0.05);
        let back = undistort(d[0], d[1], -0.3, 0.05);
        assert!(close(back[0], x, 1e-8) && close(back[1], y, 1e-8), "undistort(distort({x}, {y})): got {back:?}");
    }
}

#[test]
fn c4_dlt_exact() {
    let cam = hidden();
    let p_true = camera_matrix(&cam);
    let pts = walls();
    let px: Vec<[f64; 2]> = pts.iter().map(|&x| project(&cam, x).unwrap()).collect();
    for n in [72, 6] {
        // six points: one from each corner of the two walls and two from the middle
        let (pts_n, px_n): (Vec<Vec3>, Vec<[f64; 2]>) = if n == 72 { (pts.clone(), px.clone()) } else { [0, 11, 61, 70, 33, 40].iter().map(|&i| (pts[i], px[i])).unzip() };
        let p = dlt(&pts_n, &px_n);
        for (x, want) in pts.iter().zip(&px) {
            let got = project_p(&p, *x);
            assert!(close(got[0], want[0], 1e-5) && close(got[1], want[1], 1e-5), "{n} points: {x:?} reprojects to {got:?}, expected {want:?}");
        }
        // P is the true one up to scale
        let s = p[2][3] / p_true[2][3];
        for i in 0..3 {
            for j in 0..4 {
                assert!(close(p[i][j], s * p_true[i][j], 1e-6 * s.abs() * 1000.0), "{n} points: P is not a multiple of the true camera matrix (entry {i}, {j})");
            }
        }
    }
}

#[test]
fn c5_decompose() {
    let cam = hidden();
    let p = camera_matrix(&cam).map(|row| row.map(|v| -3.0 * v)); // any scale, even negative
    let (k, r, t) = decompose(&p);
    let want_k = [[820.0, 0.0, 335.0], [0.0, 820.0, 245.0], [0.0, 0.0, 1.0]];
    for i in 0..3 {
        for j in 0..3 {
            assert!(close(k[i][j], want_k[i][j], 1e-6), "K[{i}][{j}]: got {}, expected {}", k[i][j], want_k[i][j]);
            assert!(close(r[i][j], cam.r[i][j], 1e-9), "R[{i}][{j}]: got {}, expected {}", r[i][j], cam.r[i][j]);
        }
        assert!(close(t[i], cam.t[i], 1e-9), "t[{i}]: got {}, expected {}", t[i], cam.t[i]);
    }
}
