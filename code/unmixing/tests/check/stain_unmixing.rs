// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p unmixing stain_unmixing::

use imgcourse::Image;
use unmixing::stain_unmixing::*;

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

#[test]
fn c1_optical_density_b1_b5() {
    let img = Image::from_rows(&[[[80.0f32 / 255.0, 120.0 / 255.0, 200.0 / 255.0], [0.0, 1.0, 1.0]]]);
    let od = optical_density(&img, [1.0; 3]);
    let e = [0.50, 0.33, 0.11];
    assert!((0..3).all(|c| close(od[(0, 0)][c], e[c], 0.005)), "B1(a): got {:?}, expected {e:?}", od[(0, 0)]);
    assert!(close(od[(1, 0)][0], 2.41, 0.005), "a black red channel (B5): got {}, expected 2.41", od[(1, 0)][0]);
    assert!(close(od[(1, 0)][1], 0.0, 1e-6), "a white channel: got {}, expected 0", od[(1, 0)][1]);
    // B6(a): background 242 analysed with 205
    let bg = optical_density(&Image::new(1, 1, [242.0f32 / 255.0; 3]), [205.0 / 255.0; 3]);
    assert!(close(bg[(0, 0)][0], -0.072, 1e-3), "B6(a): got {}, expected −0.072", bg[(0, 0)][0]);
}

#[test]
fn c2_transmit_b2c() {
    // red and green only: H (0.65, 0.70), E (0.07, 0.99); blue rows unused
    let s = [[0.65, 0.07, 0.0], [0.70, 0.99, 0.0], [0.0, 0.0, 0.0]];
    let both = transmit([0.4, 0.3, 0.0], &s, [255.0; 3]);
    let h = transmit([0.4, 0.0, 0.0], &s, [255.0; 3]);
    let e = transmit([0.0, 0.3, 0.0], &s, [255.0; 3]);
    assert!(close(h[0], 140.1, 0.1) && close(h[1], 133.8, 0.1), "H alone (B2c): got ({}, {}), expected (140.1, 133.8)", h[0], h[1]);
    assert!(close(e[0], 243.0, 0.1) && close(e[1], 128.7, 0.1), "E alone: got ({}, {}), expected (243.0, 128.7)", e[0], e[1]);
    assert!(close(both[0], 133.5, 0.1) && close(both[1], 67.5, 0.1), "both: got ({}, {}), expected (133.5, 67.5)", both[0], both[1]);
    assert!(close(both[0], 255.0 * (h[0] / 255.0) * (e[0] / 255.0), 0.01), "the transmissions must multiply");
}

#[test]
fn c3_stain_matrix_b3() {
    let s = stain_matrix(HEMATOXYLIN, EOSIN);
    let r = [s[0][2], s[1][2], s[2][2]];
    let e = [-0.332, -0.081, 0.940];
    assert!((0..3).all(|c| close(r[c], e[c], 2e-3)), "B3(a), the residual column: got {r:?}, expected {e:?}");
    let h = [s[0][0], s[1][0], s[2][0]];
    let e = [0.651, 0.701, 0.291];
    assert!((0..3).all(|c| close(h[c], e[c], 2e-3)), "the unit hematoxylin column: got {h:?}, expected {e:?}");
}

#[test]
fn c3_deconvolve_b1c_b6() {
    let s = stain_matrix(HEMATOXYLIN, EOSIN);
    let od = Image::from_rows(&[[[0.5f32, 0.33, 0.11], [-0.072, -0.072, -0.072]]]);
    let c = deconvolve(&od, &s);
    let got = [c[0][(0, 0)], c[1][(0, 0)], c[2][(0, 0)]];
    let e = [0.75, -0.21, -0.09];
    assert!((0..3).all(|k| close(got[k], e[k], 0.01)), "B1(c): got {got:?}, expected {e:?}");
    let got = [c[0][(1, 0)], c[1][(1, 0)], c[2][(1, 0)]];
    let e = [-0.132, 0.018, -0.038];
    assert!((0..3).all(|k| close(got[k], e[k], 2e-3)), "B6(b): got {got:?}, expected {e:?}");
}

#[test]
fn c3_deconvolve_undoes_mixing() {
    let s = stain_matrix(HEMATOXYLIN, DAB);
    let amounts = [0.7, 0.2, 0.0];
    let od: [f32; 3] = [0, 1, 2].map(|c| (0..3).map(|k| s[c][k] * amounts[k]).sum());
    let c = deconvolve(&Image::new(1, 1, od), &s);
    let got = [c[0][(0, 0)], c[1][(0, 0)], c[2][(0, 0)]];
    assert!((0..3).all(|k| close(got[k], amounts[k], 1e-4)), "H 0.7 + DAB 0.2: got {got:?}, expected {amounts:?}");
}

#[test]
fn c4_macenko_finds_the_edges() {
    let s = stain_matrix([0.61, 0.73, 0.31], [0.13, 0.94, 0.34]);
    let (h, e) = ([s[0][0], s[1][0], s[2][0]], [s[0][1], s[1][1], s[2][1]]);
    // every mixture on a grid, the pure stains included, plus pixels too faint to count
    let mut od = vec![];
    for i in 0..=20 {
        for j in 0..=20 {
            let (a, b) = (i as f32 / 20.0, j as f32 / 20.0);
            od.push([0, 1, 2].map(|c| a * h[c] + b * e[c]));
        }
    }
    let [mh, me] = macenko(&od, 0.15, 0.0);
    let angle = |u: [f32; 3], v: [f32; 3]| (u[0] * v[0] + u[1] * v[1] + u[2] * v[2]).clamp(-1.0, 1.0).acos().to_degrees();
    assert!(angle(mh, h) < 0.1, "hematoxylin: got {mh:?}, {} degrees from {h:?}", angle(mh, h));
    assert!(angle(me, e) < 0.1, "eosin: got {me:?}, {} degrees from {e:?}", angle(me, e));
}
