// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p neuralnets cnn::

use neuralnets::cnn::*;
use imgcourse::Image;

/// The 6 × 6 image of B1.
fn b1_image() -> Image<f32> {
    let mut rows = [[0.0f32; 6]; 6];
    for (y, row) in rows.iter_mut().enumerate() {
        let start = if y < 3 { 3 } else { 2 };
        for v in row.iter_mut().skip(start) {
            *v = 1.0;
        }
    }
    Image::from_rows(&rows)
}

fn k1() -> Kernel {
    Kernel { weights: vec![Image::from_rows(&[[-1.0, 0.0, 1.0]; 3])], bias: 0.0 }
}

fn k2() -> Kernel {
    Kernel { weights: vec![Image::from_rows(&[[-1.0, -1.0, -1.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])], bias: 0.0 }
}

fn assert_image<const W: usize>(got: &Image<f32>, expected: &[[f32; W]], what: &str) {
    let e = Image::from_rows(expected);
    assert!(got.width() == e.width() && got.height() == e.height(), "{what}: got a {} × {} map, expected {} × {}", got.width(), got.height(), e.width(), e.height());
    for (x, y) in e.coords() {
        assert!((got[(x, y)] - e[(x, y)]).abs() < 1e-5, "{what}, row {}, column {}: got {}, expected {}\n{got}", y + 1, x + 1, got[(x, y)], e[(x, y)]);
    }
}

#[test]
fn c1_conv_layer_b1a() {
    let maps = conv_layer(&[b1_image()], &[k1(), k2()], true);
    assert_eq!(maps.len(), 2, "two kernels should give two feature maps");
    assert_image(&maps[0], &[[0., 3., 3., 0.], [1., 3., 2., 0.], [2., 3., 1., 0.], [3., 3., 0., 0.]], "B1a, K1");
    assert_image(&maps[1], &[[0., 0., 0., 0.], [1., 1., 1., 0.], [1., 1., 1., 0.], [0., 0., 0., 0.]], "B1a, K2");
}

#[test]
fn c1_conv_layer_relu_and_bias() {
    // -K1 sees only bright-to-dark: everything <= 0, so the ReLU gives 0 (B1c).
    let mut neg = k1();
    neg.weights[0] = neg.weights[0].map(|v| -v);
    let maps = conv_layer(&[b1_image()], &[neg.clone()], true);
    assert!(maps[0].max() == 0.0, "B1c: -K1 after the ReLU should be 0 everywhere, got\n{}", maps[0]);
    neg.bias = 5.0;
    let maps = conv_layer(&[b1_image()], &[neg], false);
    assert!((maps[0][(1, 0)] - 2.0).abs() < 1e-5, "bias 5 and -K1 at row 1, column 2: got {}, expected -3 + 5 = 2", maps[0][(1, 0)]);
}

#[test]
fn c1_conv_layer_sums_channels() {
    // Two input channels, a kernel stack that picks channel 0 plus twice channel 1.
    let a = Image::from_rows(&[[1.0, 2.0], [3.0, 4.0]]);
    let b = Image::from_rows(&[[10.0, 20.0], [30.0, 40.0]]);
    let pick = Image::from_rows(&[[0.0, 0.0], [0.0, 1.0]]);
    let twice = Image::from_rows(&[[0.0, 0.0], [0.0, 2.0]]);
    let maps = conv_layer(&[a, b], &[Kernel { weights: vec![pick, twice], bias: 0.0 }], false);
    assert!((maps[0][(0, 0)] - 84.0).abs() < 1e-4, "4 + 2 · 40: got {}, expected 84", maps[0][(0, 0)]);
}

#[test]
fn c1_equivariance_b5() {
    let x = Image::from_rows(&[[0.0, 0.0, 1.0, 3.0, 1.0, 0.0, 0.0, 0.0]]);
    let shifted = Image::from_rows(&[[0.0, 0.0, 0.0, 1.0, 3.0, 1.0, 0.0, 0.0]]);
    let k = vec![Kernel { weights: vec![Image::from_rows(&[[-1.0, 0.0, 1.0]])], bias: 0.0 }];
    assert_image(&conv_layer(&[x], &k, false)[0], &[[1., 3., 0., -3., -1., 0.]], "B5a");
    assert_image(&conv_layer(&[shifted], &k, false)[0], &[[0., 1., 3., 0., -3., -1.]], "B5b");
}

#[test]
fn c2_max_pool_b1b() {
    let maps = conv_layer(&[b1_image()], &[k1(), k2()], true);
    assert_image(&max_pool(&maps[0], 2), &[[3., 3.], [3., 1.]], "B1b, K1 pooled");
    assert_image(&max_pool(&maps[1], 2), &[[1., 1.], [1., 1.]], "B1b, K2 pooled");
    // Four 2 × 2 blocks with the largest values 4, 2, 2 and 6.
    let m = Image::from_rows(&[[1., 4., 0., 2.], [3., 2., 1., 1.], [0., 1., 5., 6.], [2., 2., 3., 0.]]);
    assert_image(&max_pool(&m, 2), &[[4., 2.], [2., 6.]], "a 4 × 4 map");
    assert_eq!(max_pool(&Image::new(7, 5, 0.0), 2).width(), 3, "a 7 wide map pools to 3 (the last column is dropped)");
}

#[test]
fn c3_weights_b2_b6() {
    assert_eq!(conv_weights(3, 3, 16), 448, "B2a");
    assert_eq!(3 * conv_weights(3, 16, 16), 6960, "B6b, three 3 × 3 layers");
    assert_eq!(conv_weights(7, 16, 16), 12560, "B6b, one 7 × 7 layer");
    assert_eq!(conv_weights(3, 8, 16), 1168, "slide Channels");
}

#[test]
fn c3_output_size_b3() {
    for (n, k, p, s, e, what) in [(28, 3, 1, 1, 28, "B3a"), (28, 5, 0, 1, 24, "B3b"), (64, 3, 1, 2, 32, "B3c"), (28, 2, 0, 2, 14, "B3d"), (7, 3, 0, 2, 3, "B3e")] {
        let got = output_size(n, k, p, s);
        assert_eq!(got, e, "{what}: n = {n}, k = {k}, p = {p}, s = {s}: got {got}, expected {e}");
    }
}

#[test]
fn c3_receptive_field_b4() {
    let conv = (3, 1);
    let pool = (2, 2);
    let rj = receptive_field(&[conv, conv, pool, conv, conv]);
    assert_eq!(rj, vec![(3, 1), (5, 1), (6, 2), (10, 2), (14, 2)], "B4a: (r, j) after each layer");
    let rj = receptive_field(&[conv, conv, pool, conv, conv, pool, conv, conv]);
    assert_eq!(rj.last(), Some(&(32, 4)), "B4b: the receptive field after 8 layers");
    assert_eq!(receptive_field(&[conv, conv, conv]).last(), Some(&(7, 1)), "B6a: three 3 × 3 layers see 7 pixels");
}
