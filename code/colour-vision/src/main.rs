// Computes the responses of an eye and a camera to spectra, makes a metamer of daylight-like
// light, and draws the chromaticity diagram with the sRGB gamut. Results in out/colour-vision/.
//   cargo run -p colour-vision                      (your code)
//   cargo run -p colour-vision --features solution  (the model solution)

use colour_vision::*;
use imgcourse::*;

/// The CIE 1931 colour-matching functions x̄, ȳ, z̄ at wavelength `l` (nm), from the
/// sum-of-Gaussians fit of Wyman, Sloan and Shirley (2013); close to the tabulated values.
fn cmf(l: f32) -> [f32; 3] {
    let g = |mu: f32, s1: f32, s2: f32| {
        let s = if l < mu { s1 } else { s2 };
        (-0.5 * ((l - mu) / s).powi(2)).exp()
    };
    [
        1.056 * g(599.8, 37.9, 31.0) + 0.362 * g(442.0, 16.0, 26.7) - 0.065 * g(501.1, 20.4, 26.2),
        0.821 * g(568.8, 46.9, 40.5) + 0.286 * g(530.9, 16.3, 31.1),
        1.217 * g(437.0, 11.8, 36.0) + 0.681 * g(459.0, 26.0, 13.8),
    ]
}

/// A synthetic camera: three broad filters (red, green, blue), not a change of basis of x̄ȳz̄.
fn camera(l: f32) -> [f32; 3] {
    let g = |mu: f32, s: f32| (-0.5 * ((l - mu) / s).powi(2)).exp();
    [g(605.0, 35.0), g(535.0, 35.0), g(460.0, 28.0)]
}

/// The light of a hot body (Planck's law) at temperature `t` kelvin, scaled to 1 at 560 nm.
fn blackbody(l: f32, t: f32) -> f32 {
    let planck = |l: f32| {
        let lm = l as f64 * 1e-9;
        1.0 / (lm.powi(5) * ((0.014388 / (lm * t as f64)).exp() - 1.0))
    };
    (planck(l) / planck(560.0)) as f32
}

/// Linear sRGB from XYZ (the inverse of the sRGB matrix), and the sRGB encoding.
fn xyz_to_srgb(c: [f32; 3]) -> [f32; 3] {
    let m = [[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]];
    m.map(|row| row[0] * c[0] + row[1] * c[1] + row[2] * c[2])
}
fn encode(v: f32) -> f32 {
    let v = v.clamp(0.0, 1.0);
    if v <= 0.0031308 { 12.92 * v } else { 1.055 * v.powf(1.0 / 2.4) - 0.055 }
}

/// Is the point inside the polygon? (Count crossings of a ray to the right.)
fn inside(p: [f32; 2], poly: &[[f32; 2]]) -> bool {
    let mut c = false;
    for i in 0..poly.len() {
        let (a, b) = (poly[i], poly[(i + 1) % poly.len()]);
        if (a[1] > p[1]) != (b[1] > p[1]) && p[0] < a[0] + (p[1] - a[1]) / (b[1] - a[1]) * (b[0] - a[0]) {
            c = !c;
        }
    }
    c
}

/// Draw curves (one colour each) over the wavelengths 380..730 nm into a plot.
fn plot(curves: &[(&[f32], [f32; 3])], wavelengths: &[f32]) -> Image<[f32; 3]> {
    let (w, h, pad) = (480usize, 240usize, 10usize);
    let mut img = Image::new(w, h, [1.0f32; 3]);
    let top = curves.iter().flat_map(|(c, _)| c.iter()).fold(0.0f32, |m, &v| m.max(v)) * 1.05;
    let (l0, l1) = (wavelengths[0], wavelengths[wavelengths.len() - 1]);
    let to_px = |l: f32, v: f32| {
        let x = pad as f32 + (l - l0) / (l1 - l0) * (w - 2 * pad) as f32;
        let y = (h - pad) as f32 - v / top * (h - 2 * pad) as f32;
        (x, y)
    };
    for x in pad..w - pad {
        img[(x, h - pad)] = [0.0; 3];
    }
    for (c, colour) in curves {
        for i in 0..c.len() - 1 {
            for k in 0..=400 {
                let f = k as f32 / 400.0;
                let (x, y) = to_px(wavelengths[i] * (1.0 - f) + wavelengths[i + 1] * f, c[i] * (1.0 - f) + c[i + 1] * f);
                for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                    let (px, py) = (x as usize + dx, y as usize + dy);
                    if px < w && py < h {
                        img[(px, py)] = *colour;
                    }
                }
            }
        }
    }
    img
}

fn main() {
    let dir = "out/colour-vision";
    // Sample every 10 nm from 380 to 730 nm: 36 bins, as in the lecture.
    let wl: Vec<f32> = (0..36).map(|j| 380.0 + 10.0 * j as f32).collect();
    let eye: Vec<Vec<f32>> = (0..3).map(|i| wl.iter().map(|&l| cmf(l)[i]).collect()).collect();
    let cam: Vec<Vec<f32>> = (0..3).map(|i| wl.iter().map(|&l| camera(l)[i]).collect()).collect();

    // A daylight-like spectrum (a 6500 K hot body) and a metamer of it for the eye.
    let s1: Vec<f32> = wl.iter().map(|&l| blackbody(l, 6500.0)).collect();
    let mut rng = Rng::new(1);
    let v: Vec<f32> = (0..36).map(|_| rng.normal()).collect();
    let s2 = metamer(&eye, &s1, &v, 0.9);
    let rel = |a: &[Vec<f32>], s: &[f32]| {
        let (c, c1) = (responses(a, s), responses(a, &s1));
        [c[0] / c1[0], c[1] / c1[1], c[2] / c1[2]]
    };
    println!("each relative to the response to the 6500 K light:");
    println!("  eye (X, Y, Z):    light {:.3?}   metamer {:.3?}", rel(&eye, &s1), rel(&eye, &s2));
    println!("  camera (R, G, B): light {:.3?}   metamer {:.3?}", rel(&cam, &s1), rel(&cam, &s2));
    let xy = chromaticity(responses(&eye, &s1).try_into().unwrap());
    println!("  chromaticity of the 6500 K light: ({:.3}, {:.3})", xy[0], xy[1]);
    println!("  smallest value of the metamer: {:.3}", s2.iter().fold(f32::INFINITY, |m, &x| m.min(x)));
    save_rgb(&plot(&[(&s1, [0.0, 0.0, 0.0]), (&s2, [0.9, 0.4, 0.0])], &wl), format!("{dir}/metamer.png"));

    // The chromaticity diagram: the spectral locus (monochromatic light every 1 nm from 420
    // to 645 nm; outside that range the fitted curves are too rough, and the true locus
    // hardly moves there), closed by the purple line; inside it each point is drawn in its colour, outside the sRGB
    // triangle at reduced brightness.
    let locus: Vec<[f32; 2]> = (420..=645).map(|l| chromaticity(cmf(l as f32))).collect();
    let (r, g, b) = ([0.64, 0.33], [0.30, 0.60], [0.15, 0.06]);
    let scale = 600.0;
    let (w, h) = (500, 560);
    let diagram = Image::from_fn(w, h, |px, py| {
        let p = [(px as f32 - 10.0) / scale, (h as f32 - py as f32 - 10.0) / scale];
        if !inside(p, &locus) {
            return [1.0; 3];
        }
        let bary = barycentric(p, r, g, b);
        let min = bary[0].min(bary[1]).min(bary[2]);
        if min.abs() < 0.004 {
            return [0.0; 3]; // the edge of the triangle
        }
        let rgb = xyz_to_srgb([p[0] / p[1], 1.0, (1.0 - p[0] - p[1]) / p[1]]).map(|v| v.max(0.0));
        let m = rgb[0].max(rgb[1]).max(rgb[2]);
        let dim = if min < 0.0 { 0.55 } else { 1.0 };
        rgb.map(|v| dim * encode(v / m))
    });
    let mut diagram = diagram;
    let white = [0.3127f32, 0.3290];
    for dy in -3i32..=3 {
        for dx in -3i32..=3 {
            if dx * dx + dy * dy <= 9 {
                let (x, y) = ((white[0] * scale) as i32 + 10 + dx, h as i32 - 10 - (white[1] * scale) as i32 + dy);
                diagram[(x as usize, y as usize)] = [0.0; 3];
            }
        }
    }
    save_rgb(&diagram, format!("{dir}/chromaticity.png"));
    // Areas by the shoelace formula.
    let area = |poly: &[[f32; 2]]| {
        let n = poly.len();
        0.5 * (0..n).map(|i| poly[i][0] * poly[(i + 1) % n][1] - poly[(i + 1) % n][0] * poly[i][1]).sum::<f32>().abs()
    };
    let (a_locus, a_srgb) = (area(&locus), area(&[r, g, b]));
    println!("area of the horseshoe {a_locus:.3}, of the sRGB triangle {a_srgb:.3}: sRGB covers {:.1} %", 100.0 * a_srgb / a_locus);
    for l in [460.0, 520.0, 580.0, 620.0] {
        let p = chromaticity(cmf(l));
        let bary = barycentric(p, r, g, b);
        println!("{l} nm: chromaticity ({:.3}, {:.3}), barycentric in sRGB {:.2?}", p[0], p[1], bary);
    }
    println!("wrote {dir}/metamer.png and {dir}/chromaticity.png");
}
