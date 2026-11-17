// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p day02 scale_space::

use imgcourse::Image;
use day02::scale_space::*;

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() <= tol
}

/// B1: a 1D box of half-width 3 and height 100, as an image one row high. The two edge
/// pixels at ±3 are half covered by the box, so they get 50.
fn b1_box() -> Image<f32> {
    Image::from_fn(61, 1, |x, _| match (x as isize - 30).abs() {
        0..=2 => 100.0,
        3 => 50.0,
        _ => 0.0,
    })
}

#[test]
fn c1_normalised_log_box_b1() {
    let table = [(1.0, 2.7), (2.0, 38.9), (3.0, 48.4), (4.0, 45.2), (6.0, 35.2)];
    let mut best = (0.0, f32::MIN);
    for (sigma, expected) in table {
        let got = normalised_log(&b1_box(), sigma)[(30, 0)];
        // The table is for a continuous signal; on pixels, [1 −2 1] is a few percent smaller,
        // and at σ = 1 the pixel grid itself matters (the sampled box is not a sharp box).
        if sigma > 1.0 {
            assert!(close(got, expected, 0.05 * expected), "B1, σ = {sigma}: response at the centre {got}, expected about {expected}");
        }
        if got > best.1 {
            best = (sigma, got);
        }
    }
    assert_eq!(best.0, 3.0, "B1: σ = 3 should win (half-width 3), but σ = {} did", best.0);
}

#[test]
fn c1_disc_size_from_scale() {
    // A disc of radius 6: the normalised LoG at its centre peaks near σ = 6/√2 = 4.24.
    let disc = Image::from_fn(64, 64, |x, y| if (x as f32 - 32.0).powi(2) + (y as f32 - 32.0).powi(2) <= 36.0 { 1.0 } else { 0.0 });
    let sigmas: Vec<f32> = (0..25).map(|i| 2.0 + 0.25 * i as f32).collect();
    let responses: Vec<f32> = sigmas.iter().map(|&s| normalised_log(&disc, s)[(32, 32)]).collect();
    let k = (0..sigmas.len()).max_by(|&a, &b| responses[a].total_cmp(&responses[b])).unwrap();
    assert!(close(sigmas[k], 4.24, 0.5), "disc of radius 6: best σ = {}, expected about 4.24", sigmas[k]);
}

#[test]
fn c1_zero_crossing_b6() {
    let got = zero_crossings(&[3.1, 2.4, 0.9, -0.9, -2.4, -3.1]);
    assert!(got.len() == 1 && close(got[0], 2.5, 1e-5), "B6: got {got:?}, expected one crossing at 2.5");
    let got = zero_crossings(&[-1.0, 3.0, 1.0, 0.0, -2.0]);
    assert!(got.len() == 1 && close(got[0], 0.25, 1e-5), "−1, 3: got {got:?}, expected one crossing at 0.25 (0 itself is not a crossing)");
}

#[test]
fn c2_detect_two_discs() {
    // Discs of radius 4 at (20, 24) and radius 10 at (60, 50), on a background that brightens to the right.
    let img = Image::from_fn(96, 80, |x, y| {
        let (xf, yf) = (x as f32, y as f32);
        let d1 = (xf - 20.0).powi(2) + (yf - 24.0).powi(2) <= 16.0;
        let d2 = (xf - 60.0).powi(2) + (yf - 50.0).powi(2) <= 100.0;
        0.002 * xf + if d1 || d2 { 0.8 } else { 0.1 }
    });
    let sigmas: Vec<f32> = (0..12).map(|i| 1.5 * 1.2f32.powi(i)).collect();
    let stack: Vec<Image<f32>> = sigmas.iter().map(|&s| normalised_log(&img, s)).collect();
    let blobs = detect_blobs(&stack, &sigmas, 0.2);
    assert_eq!(blobs.len(), 2, "two discs: expected 2 detections, got {blobs:?}");
    for (cx, cy, r) in [(20, 24, 4.0f32), (60, 50, 10.0)] {
        let b = blobs.iter().find(|b| b.x.abs_diff(cx) <= 1 && b.y.abs_diff(cy) <= 1);
        let b = b.unwrap_or_else(|| panic!("no detection at the disc ({cx}, {cy}): got {blobs:?}"));
        let radius = 2f32.sqrt() * b.sigma;
        assert!(close(radius, r, 0.25 * r), "disc of radius {r} at ({cx}, {cy}): detected at σ = {}, radius {radius}", b.sigma);
    }
}

#[test]
fn c3_dog_b3() {
    // B3: a Gaussian bump of width 3 and height 100, σ = 2, k = 1.2: DoG at the centre −5.12.
    let bump = Image::from_fn(61, 1, |x, _| 100.0 * (-((x as f32 - 30.0).powi(2)) / 18.0).exp());
    let d = dog(&bump, 2.0, 1.2);
    assert!(close(d[(30, 0)], -5.12, 0.03), "B3, centre: DoG {}, expected −5.12", d[(30, 0)]);
    assert!(close(d[(28, 0)], -3.15, 0.03), "B3, x = −2: DoG {}, expected −3.15", d[(28, 0)]);
    // DoG ≈ (k − 1) σ² ∇²L = −(k − 1) × normalised LoG.
    let approx = -0.2 * normalised_log(&bump, 2.0)[(30, 0)];
    assert!(close(approx, -5.12, 0.1), "B3(b): −(k − 1) × normalised LoG at the centre is {approx}, expected about −5.12");
}

#[test]
fn c4_pyramid_sizes_b4() {
    let img = imgcourse::samples::cell().crop(40, 40, 64, 64);
    let mut g = img.clone();
    let mut total = g.pixels().len();
    let mut sides = vec![g.width()];
    while g.width() > 1 {
        g = downsample(&g);
        total += g.pixels().len();
        sides.push(g.width());
    }
    assert_eq!(sides, [64, 32, 16, 8, 4, 2, 1], "B4: side lengths of the Gaussian pyramid of 64 × 64");
    assert_eq!(total, 5461, "B4: total pixels {total}, expected 5461 (about 4/3 of 4096)");
}

#[test]
fn c4_laplacian_pyramid_is_exact() {
    let img = imgcourse::samples::cell().crop(10, 30, 50, 37); // odd sizes on purpose
    let pyr = laplacian_pyramid(&img, 4);
    assert_eq!(pyr.len(), 4, "4 levels expected");
    assert_eq!((pyr[1].width(), pyr[1].height()), (25, 19), "level 1 of 50 × 37 should be 25 × 19");
    let back = reconstruct(&pyr);
    for (x, y) in img.coords() {
        assert!(close(back[(x, y)], img[(x, y)], 1e-5), "pixel ({x}, {y}): rebuilt {} but the image is {}", back[(x, y)], img[(x, y)]);
    }
    let up = upsample(&Image::from_rows(&[[0.0, 1.0]]), 4, 1);
    let want = [0.0, 0.5, 1.0, 1.0];
    assert!(up.pixels().iter().zip(want).all(|(a, b)| close(*a, b, 1e-6)), "upsample of [0, 1] to 4 pixels: got {:?}, expected {want:?}", up.pixels());
}
