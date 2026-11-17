// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p pipeline3d

use pipeline3d::*;

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-9
}

fn close_vec(a: &[f64], b: &[f64]) -> bool {
    a.iter().zip(b).all(|(p, q)| close(*p, *q))
}

/// Corner k of the unit cube, numbered k = x + 2y + 4z as in B2.
fn corner(k: usize) -> Vec3 {
    [(k & 1) as f64, ((k >> 1) & 1) as f64, ((k >> 2) & 1) as f64]
}

#[test]
fn c1_cube_normals_b2() {
    let faces = [("z = 0", [0, 2, 3, 1]), ("z = 1", [4, 5, 7, 6]), ("x = 0", [0, 4, 6, 2]), ("x = 1", [1, 5, 7, 3]), ("y = 0", [0, 1, 5, 4]), ("y = 1", [2, 3, 7, 6])];
    let expected: [Vec3; 6] = [[0., 0., -1.], [0., 0., 1.], [-1., 0., 0.], [-1., 0., 0.], [0., -1., 0.], [0., -1., 0.]];
    let mut wrong = Vec::new();
    for ((name, q), e) in faces.iter().zip(expected) {
        let n = face_normal(corner(q[0]), corner(q[1]), corner(q[2]));
        assert!(close_vec(&n, &e), "face {name}: got the normal {n:?}, expected {e:?}");
        // outwards = pointing away from the cube's centre (0.5, 0.5, 0.5)
        let centre = corner(q[0]).iter().zip(corner(q[2])).map(|(a, b)| (a + b) / 2.0 - 0.5).collect::<Vec<_>>();
        if dot(n, [centre[0], centre[1], centre[2]]) < 0.0 {
            wrong.push(*name);
        }
    }
    assert_eq!(wrong, ["x = 1", "y = 1"], "faces wound the wrong way: got {wrong:?}, expected x = 1 and y = 1");
}

#[test]
fn c2_order_matters_b3() {
    let r = rotation_z(90.0);
    let t = translation([1., 0., 0.]);
    let tr = mat_mul(&t, &r);
    let rt = mat_mul(&r, &t);
    let tr_expected = [[0., -1., 0., 1.], [1., 0., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]];
    for i in 0..4 {
        assert!(close_vec(&tr[i], &tr_expected[i]), "T R, row {i}: got {:?}, expected {:?}", tr[i], tr_expected[i]);
    }
    let p = [1., 0., 0., 1.];
    let (a, b) = (apply(&tr, p), apply(&rt, p));
    assert!(close_vec(&a, &[1., 1., 0., 1.]), "T R (1, 0, 0, 1): got {a:?}, expected (1, 1, 0, 1)");
    assert!(close_vec(&b, &[0., 2., 0., 1.]), "R T (1, 0, 0, 1): got {b:?}, expected (0, 2, 0, 1)");
    let d = [1., 0., 0., 0.];
    let (a, b) = (apply(&tr, d), apply(&rt, d));
    assert!(close_vec(&a, &[0., 1., 0., 0.]) && close_vec(&b, &[0., 1., 0., 0.]), "the direction (1, 0, 0, 0): got {a:?} and {b:?}, expected (0, 1, 0, 0) for both");
}

#[test]
fn c3_look_at_b6() {
    let v = look_at([0., 0., 5.], [0., 0., 0.], [0., 1., 0.]);
    let expected = translation([0., 0., -5.]);
    for i in 0..4 {
        assert!(close_vec(&v[i], &expected[i]), "row {i}: got {:?}, expected {:?}", v[i], expected[i]);
    }
    let origin = apply(&v, [0., 0., 0., 1.]);
    assert!(close_vec(&origin, &[0., 0., -5., 1.]), "the world origin in view space: got {origin:?}, expected (0, 0, −5)");
}

#[test]
fn c3_look_at_general() {
    let (eye, target) = ([3., 2., 4.], [0., 1., -1.]);
    let v = look_at(eye, target, [0., 1., 0.]);
    let e = apply(&v, [eye[0], eye[1], eye[2], 1.]);
    assert!(close_vec(&e, &[0., 0., 0., 1.]), "V e: got {e:?}, expected the origin");
    let g = apply(&v, [target[0], target[1], target[2], 1.]);
    let dist = dot(sub(target, eye), sub(target, eye)).sqrt();
    assert!(close_vec(&g, &[0., 0., -dist, 1.]), "the target must be straight ahead on −z: got {g:?}, expected (0, 0, {:.4})", -dist);
    for i in 0..3 {
        for j in 0..3 {
            let d = dot([v[i][0], v[i][1], v[i][2]], [v[j][0], v[j][1], v[j][2]]);
            let want = if i == j { 1.0 } else { 0.0 };
            assert!(close(d, want), "the rows of R must be orthonormal: row {i} · row {j} = {d}");
        }
    }
}

#[test]
fn c4_perspective_b5() {
    let p = perspective(90.0, 1.0, 1.0, 3.0);
    let expected = [[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., -2., -3.], [0., 0., -1., 0.]];
    for i in 0..4 {
        assert!(close_vec(&p[i], &expected[i]), "row {i}: got {:?}, expected {:?}", p[i], expected[i]);
    }
    // B5 (a): depth at z = −2 is 0.5
    let c = apply(&p, [0., 0., -2., 1.]);
    assert!(close(c[2] / c[3], 0.5), "NDC depth at z = −2: got {}, expected 0.5", c[2] / c[3]);
    // B4: with a field of view of 90°, NDC x is n x / (−z) with n = 1
    let c = apply(&p, [1., 0.5, -4., 1.]);
    assert!(close(c[0] / c[3], 0.25) && close(c[1] / c[3], 0.125), "(1, 0.5, −4): got ({}, {}), expected (0.25, 0.125)", c[0] / c[3], c[1] / c[3]);
}

#[test]
fn c5_to_screen_b5_b8() {
    let p = perspective(90.0, 1.0, 1.0, 3.0);
    let clip = apply(&p, [1., 1., -2., 1.]);
    assert!(close_vec(&clip, &[1., 1., 1., 2.]), "clip coordinates of (1, 1, −2): got {clip:?}, expected (1, 1, 1, 2)");
    let s = to_screen(clip, 600.0, 600.0).expect("the point is in front of the camera");
    assert!(close_vec(&s, &[450., 450., 0.5]), "pixel and depth: got {s:?}, expected (450, 450, 0.5)");
    // B8: a point behind the camera
    let behind = apply(&p, [0.5, 0., 1., 1.]);
    assert!(close(behind[3], -1.0), "w of (0.5, 0, 1): got {}, expected −1", behind[3]);
    assert!(to_screen(behind, 600.0, 600.0).is_none(), "a point with w < 0 must give None");
}
