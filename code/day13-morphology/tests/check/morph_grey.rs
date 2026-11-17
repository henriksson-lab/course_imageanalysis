// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p day13 morph_grey::
// The 1D signals are images of height 1; the grey levels are those of the exercises.

use imgcourse::Image;
use day13::morph_grey::*;

fn signal<const W: usize>(v: [f32; W]) -> Image<f32> {
    Image::from_rows(&[v])
}

fn check(name: &str, got: &Image<f32>, expected: &[f32]) {
    let g: Vec<f32> = got.pixels().to_vec();
    let ok = g.len() == expected.len() && g.iter().zip(expected).all(|(a, b)| (a - b).abs() < 1e-5);
    assert!(ok, "{name}:\n  got      {g:?}\n  expected {expected:?}");
}

/// The signal of B1 (samples 0–11).
fn b1() -> Image<f32> {
    signal([2., 2., 7., 2., 3., 3., 6., 6., 6., 1., 2., 2.])
}

#[test]
fn c1_erode_dilate_b1() {
    check("B1: erosion, width 3", &erode(&b1(), &window(3)), &[2., 2., 2., 2., 2., 3., 3., 6., 1., 1., 1., 2.]);
    check("B1: dilation, width 3", &dilate(&b1(), &window(3)), &[2., 7., 7., 7., 3., 6., 6., 6., 6., 6., 2., 2.]);
}

#[test]
fn c2_open_tophat_b1() {
    check("B1: opening, width 3", &open(&b1(), &window(3)), &[2., 2., 2., 2., 3., 3., 6., 6., 6., 1., 2., 2.]);
    check("B1: white top-hat, width 3", &white_tophat(&b1(), &window(3)), &[0., 0., 5., 0., 0., 0., 0., 0., 0., 0., 0., 0.]);
}

#[test]
fn c2_open_tophat_b2() {
    check("B2: opening, width 5", &open(&b1(), &window(5)), &[2., 2., 2., 2., 3., 3., 3., 3., 3., 1., 1., 1.]);
    check("B2: white top-hat, width 5", &white_tophat(&b1(), &window(5)), &[0., 0., 5., 0., 0., 0., 3., 3., 3., 0., 1., 1.]);
}

#[test]
fn c2_close_black_tophat_b3() {
    check("B3: closing, width 3", &close(&b1(), &window(3)), &[2., 2., 7., 3., 3., 3., 6., 6., 6., 2., 2., 2.]);
    check("B3: black top-hat, width 3", &black_tophat(&b1(), &window(3)), &[0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0.]);
}

#[test]
fn c3_gradient_b4() {
    let step = signal([0., 0., 0., 0., 10., 10., 10., 10.]);
    let se = window(3);
    check("B4: gradient of the step", &gradient(&step, &se), &[0., 0., 0., 10., 10., 0., 0., 0.]);
    // inner and outer gradient, from erosion and dilation
    let inner = step.zip_map(&erode(&step, &se), |i, e| i - e);
    let outer = dilate(&step, &se).zip_map(&step, |d, i| d - i);
    check("B4: inner gradient", &inner, &[0., 0., 0., 0., 10., 0., 0., 0.]);
    check("B4: outer gradient", &outer, &[0., 0., 0., 10., 0., 0., 0., 0.]);
    let ramp = signal([0., 2., 4., 6., 8., 10., 12., 14.]);
    check("B4: gradient of the ramp", &gradient(&ramp, &se), &[2., 4., 4., 4., 4., 4., 4., 2.]);
}

#[test]
fn c4_rank_filters_b5() {
    let j = signal([5., 5., 9., 5., 5., 0., 5., 5., 5., 9., 0., 5.]);
    let se = window(3);
    check("B5: median", &median(&j, &se), &[5.; 12]);
    check("B5: opening", &open(&j, &se), &[5., 5., 5., 5., 5., 0., 5., 5., 5., 5., 0., 0.]);
    check("B5: closing", &close(&j, &se), &[5., 5., 9., 5., 5., 5., 5., 5., 5., 9., 5., 5.]);
    check("B5: closing of the opening", &close(&open(&j, &se), &se), &[5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 0., 0.]);
}

#[test]
fn c2_tophat_ignores_constant_and_ramp_b9() {
    // B9: adding a constant changes nothing, and a ramp is its own opening (away from the ends)
    let img = imgcourse::samples::cell().crop(60, 60, 30, 20);
    let se = disc(2);
    let a = white_tophat(&img, &se);
    let b = white_tophat(&img.map(|v| v + 0.3), &se);
    assert!(a.zip_map(&b, |p, q| (p - q).abs() < 1e-5).pixels().iter().all(|&ok| ok), "B9: top-hat of I + 0.3 differs from the top-hat of I");
    let ramp = Image::from_fn(20, 1, |x, _| 3.0 * x as f32);
    let t = white_tophat(&ramp, &window(5));
    assert!((2..18).all(|x| t[(x, 0)].abs() < 1e-5), "B9: top-hat of a ramp: got {:?}, expected 0 away from the ends", t.pixels());
}

#[test]
fn c1_disc_on_2d() {
    // a single bright pixel: dilation by the disc of radius 1 is the 5-pixel cross
    let mut img = Image::new(5, 5, 0.0);
    img[(2, 2)] = 1.0;
    let d = dilate(&img, &disc(1));
    let n = d.pixels().iter().filter(|&&v| v == 1.0).count();
    assert!(n == 5 && d[(2, 1)] == 1.0 && d[(1, 1)] == 0.0, "dilation of a single pixel by the disc of radius 1: expected the 5-pixel cross, got\n{d}");
    assert_eq!(erode(&img, &disc(1)).max(), 0.0, "erosion removes a single bright pixel");
}
