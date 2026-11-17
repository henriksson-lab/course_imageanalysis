// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p day09 backprojection::

use day09::backprojection::tomo::{angles, centre, projection};
use day09::backprojection::*;
use imgcourse::{fft, Image};
use std::f32::consts::PI;

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

#[test]
fn c1_three_points_b1() {
    // ρ from −2 to 2 is bin 0 to 4. p0: 1 at ρ = 2, 2 at ρ = 0, 1 at ρ = −1.
    // p90: 1 at ρ = 0, 2 at ρ = 1, 1 at ρ = −1.
    let p0 = [0.0, 1.0, 2.0, 0.0, 1.0];
    let p90 = [0.0, 1.0, 1.0, 2.0, 0.0];
    let b = backproject_two(&p0, &p90);
    // the table of B1, top row (y = 2) first
    let table = [[0.0, 1.0, 2.0, 0.0, 1.0], [2.0, 3.0, 4.0, 2.0, 3.0], [1.0, 2.0, 3.0, 1.0, 2.0], [1.0, 2.0, 3.0, 1.0, 2.0], [0.0, 1.0, 2.0, 0.0, 1.0]];
    for (py, row) in table.iter().enumerate() {
        for (px, &e) in row.iter().enumerate() {
            assert_eq!(b[(px, py)], e, "B1(b), x = {}, y = {}: got {}, expected {e}", px as i32 - 2, 2 - py as i32, b[(px, py)]);
        }
    }
}

#[test]
fn c1_two_by_two_b3() {
    // [1 2; 3 4]: column sums (4, 6); row sums (3, 7) from the top, so p90 = (7, 3) from the bottom
    let b = backproject_two(&[4.0, 6.0], &[7.0, 3.0]);
    assert_eq!(b.pixels(), &[7.0, 9.0, 11.0, 13.0], "B3: expected 7 9 / 11 13");
}

#[test]
fn c2_two_angles_match_c1() {
    // With the angles 0° and 90°, back-projection is π/2 times the sum of C1.
    let n = 5;
    let mut sino = Image::new(2, n, 0.0);
    let (p0, p90) = ([0.0, 1.0, 2.0, 0.0, 1.0], [0.0, 1.0, 1.0, 2.0, 0.0]);
    for k in 0..n {
        sino[(0, k)] = p0[k];
        sino[(1, k)] = p90[k];
    }
    let b = backproject(&sino, &[0.0, PI / 2.0], n);
    let by_hand = backproject_two(&p0, &p90);
    for (x, y) in b.coords() {
        let e = by_hand[(x, y)] * PI / 2.0;
        assert!(close(b[(x, y)], e, 1e-4), "pixel ({x}, {y}): got {}, expected π/2 · {} = {e}", b[(x, y)], by_hand[(x, y)]);
    }
}

#[test]
fn c2_point_halo_b2() {
    // A point at the centre, 180 angles: the back-projection falls like 1/r.
    let n = 65;
    let c = centre(n) as usize;
    let mut sino = Image::new(180, n, 0.0);
    for a in 0..180 {
        sino[(a, c)] = 1.0; // the projection of a point at the centre: one bin at ρ = 0
    }
    let b = backproject(&sino, &angles(180), n);
    assert!(b[(c, c)] > b[(c + 1, c)], "the peak should be at the point");
    let at = |r: usize| (b[(c + r, c)] + b[(c - r, c)] + b[(c, c + r)] + b[(c, c - r)]) / 4.0;
    let ratio = at(5) / at(20);
    assert!((ratio - 4.0).abs() < 0.4, "B2(b): the halo at r = 5 should be about 4 times that at r = 20, got {ratio:.2}");
}

#[test]
fn c3_slice_theorem_b4() {
    // vertical stripes with period 8 in a 64 × 64 image
    let n = 64;
    let img = Image::from_fn(n, n, |x, _| 0.5 + 0.5 * (2.0 * PI * x as f32 / 8.0).cos());
    let spectrum = fft::fft2_real(&img);
    let mid = n / 2; // ω = 0
    let p0 = projection_spectrum(&projection(&img, 0.0));
    let s0 = spectrum_slice(&spectrum, 0.0);
    for (w, e) in [(0, 2048.0), (8, 1024.0), (-8, 1024.0), (3, 0.0)] {
        let i = (mid as i32 + w) as usize;
        assert!(close(p0[i], e, 0.5), "B4(a), projection at 0°, ω = {w}: got {:.1}, expected {e}", p0[i]);
        assert!(close(s0[i], e, 0.5), "B4(c), 2D spectrum on the 0° line, ω = {w}: got {:.1}, expected {e}", s0[i]);
    }
    let p90 = projection_spectrum(&projection(&img, PI / 2.0));
    let s90 = spectrum_slice(&spectrum, PI / 2.0);
    for (w, e) in [(0, 2048.0), (8, 0.0)] {
        let i = (mid as i32 + w) as usize;
        assert!(close(p90[i], e, 0.5), "B4(b), projection at 90°, ω = {w}: got {:.1}, expected {e}", p90[i]);
        assert!(close(s90[i], e, 0.5), "B4(c), 2D spectrum on the 90° line, ω = {w}: got {:.1}, expected {e}", s90[i]);
    }
}

#[test]
fn c3_horizontal_stripes_b5() {
    // horizontal stripes with period 16: they show at 90° (ω = ±4), not at 0°
    let n = 64;
    let img = Image::from_fn(n, n, |_, y| 0.5 + 0.5 * (2.0 * PI * y as f32 / 16.0).cos());
    let s90 = spectrum_slice(&fft::fft2_real(&img), PI / 2.0);
    let s0 = spectrum_slice(&fft::fft2_real(&img), 0.0);
    assert!(close(s90[n / 2 + 4], 1024.0, 0.5), "B5(a): the 90° line at ω = 4: got {:.1}, expected 1024", s90[n / 2 + 4]);
    assert!(close(s0[n / 2 + 4], 0.0, 0.5), "B5(a): the 0° line at ω = 4: got {:.1}, expected 0", s0[n / 2 + 4]);
}

#[test]
fn c3_oblique_stripes_b5b() {
    // stripes whose grey value changes along the direction 30° (x right, y up), 8 periods per 64
    // pixels: the 30° projection and the 30° slice both show them at ω = 8, the 150° ones do not
    let n = 64;
    let c = centre(n);
    let t = 30f32.to_radians();
    let img = Image::from_fn(n, n, |px, py| {
        let (x, y) = (px as f32 - c, c - py as f32);
        0.5 + 0.5 * (2.0 * PI * 8.0 * (x * t.cos() + y * t.sin()) / n as f32).cos()
    });
    let spectrum = fft::fft2_real(&img);
    let i = n / 2 + 8;
    let (p30, p150) = (projection_spectrum(&projection(&img, t))[i], projection_spectrum(&projection(&img, PI - t))[i]);
    let (s30, s150) = (spectrum_slice(&spectrum, t)[i], spectrum_slice(&spectrum, PI - t)[i]);
    assert!(p30 > 4.0 * p150, "projection spectra at ω = 8: 30° gave {p30:.0}, 150° gave {p150:.0}; 30° should be much larger");
    assert!(s30 > 4.0 * s150, "2D slices at ω = 8: 30° gave {s30:.0}, 150° gave {s150:.0}; 30° should be much larger (rows go down: v = −ω sin θ)");
}
