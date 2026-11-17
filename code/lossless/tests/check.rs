// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p lossless

use imgcourse::{samples, Image};
use lossless::*;

#[test]
fn c1_paeth_and_med_b2() {
    let cases = [((100, 120, 110), 110, 110), ((50, 60, 100), 50, 50), ((10, 30, 12), 30, 28)];
    for ((a, b, c), paeth, med) in cases {
        let p = predict(a, b, c, Predictor::Paeth);
        let m = predict(a, b, c, Predictor::Med);
        assert_eq!(p, paeth, "B2, (a, b, c) = ({a}, {b}, {c}): Paeth gave {p}, expected {paeth}");
        assert_eq!(m, med, "B2, (a, b, c) = ({a}, {b}, {c}): MED gave {m}, expected {med}");
    }
}

#[test]
fn c1_edges_b3() {
    // vertical edge: a = 40, b = 100, c = 40, x = 100
    let expected = [(Predictor::Left, 40), (Predictor::Up, 100), (Predictor::Average, 70), (Predictor::Planar, 100), (Predictor::Med, 100)];
    for (p, e) in expected {
        let got = predict(40, 100, 40, p);
        assert_eq!(got, e, "B3, vertical edge, {p:?}: got {got}, expected {e}");
    }
    // horizontal edge: a = 100, b = 40, c = 40
    let expected = [(Predictor::Left, 100), (Predictor::Up, 40), (Predictor::Average, 70), (Predictor::Planar, 100), (Predictor::Med, 100)];
    for (p, e) in expected {
        let got = predict(100, 40, 40, p);
        assert_eq!(got, e, "B3, horizontal edge, {p:?}: got {got}, expected {e}");
    }
    assert_eq!(predict(-3, 0, 0, Predictor::Average), -2, "average of −3 and 0: floor(−1.5) = −2");
}

#[test]
fn c2_residuals_b1() {
    let row = Image::from_rows(&[[100, 102, 104, 103, 105, 107, 106, 108]]);
    let e = residuals(&row, Predictor::Left);
    assert_eq!(e.row(0), &[100, 2, 2, -1, 2, 2, -1, 2], "B1(a): residuals of the left predictor");
}

#[test]
fn c3_decode_b1() {
    let res = Image::from_rows(&[[50, 1, 1, 0, -2, -2, 1]]);
    let x = decode(&res, Predictor::Left);
    assert_eq!(x.row(0), &[50, 51, 52, 52, 50, 48, 49], "B1(c): decoded row");
}

#[test]
fn c3_decode_undoes_residuals_b7() {
    let img = samples::cell().crop(60, 80, 40, 30).map(|v| (v * 255.0).round() as i32);
    for p in ALL {
        let back = decode(&residuals(&img, p), p);
        for (x, y) in img.coords() {
            assert_eq!(back[(x, y)], img[(x, y)], "{p:?}, pixel ({x}, {y}): decoded {} but the image has {}", back[(x, y)], img[(x, y)]);
        }
    }
}

#[test]
fn c4_entropy_b1() {
    let h = entropy(&[100, 102, 104, 103, 105, 107, 106, 108]);
    assert!((h - 3.0).abs() < 1e-5, "B1(b), 8 different values: got {h}, expected 3");
    let h = entropy(&[2, 2, -1, 2, 2, -1, 2]);
    assert!((h - 0.863).abs() < 1e-3, "B1(b), the 7 residuals: got {h}, expected 0.86");
}
