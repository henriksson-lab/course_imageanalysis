// Shows a smooth field with jet and viridis, a signed field with a diverging map, two
// fluorescence channels merged and seen by a simulated deuteranope, and a decorrelation
// stretch of a synthetic H&E image. Pictures in out/colour-display/.
//   cargo run -p day42 -- colour-display                      (your code)
//   cargo run -p day42 --features solution -- colour-display  (the model solution)

use crate::colour_display::*;
use imgcourse::*;

/// A smooth height field with a few hills (no edges at all), values 0..1.
fn hills(size: usize) -> Image<f32> {
    let s = size as f32;
    let hills = [(0.3, 0.35, 0.18, 1.0), (0.7, 0.6, 0.22, 0.8), (0.45, 0.8, 0.12, 0.5), (0.8, 0.2, 0.1, 0.35)];
    let f = Image::from_fn(size, size, |x, y| {
        let (u, v) = (x as f32 / s, y as f32 / s);
        hills.iter().map(|&(cx, cy, r, h)| h * (-((u - cx).powi(2) + (v - cy).powi(2)) / (2.0 * r * r)).exp()).sum()
    });
    let m = f.max();
    f.map(|v| v / m)
}

/// The colour bar of a map, `w × 24` pixels.
fn bar(anchors: &[[f32; 3]], w: usize) -> Rgb {
    let ramp = Image::from_fn(w, 24, |x, _| x as f32 / (w - 1) as f32);
    colour_map(&ramp, 0.0, 1.0, anchors)
}

/// Stack images vertically (same width), with a white gap.
fn stack(images: &[Rgb]) -> Rgb {
    let w = images[0].width();
    let h: usize = images.iter().map(|i| i.height() + 4).sum();
    let mut out = Image::new(w, h, [1.0; 3]);
    let mut y0 = 0;
    for img in images {
        for (x, y) in img.coords() {
            out[(x, y0 + y)] = img[(x, y)];
        }
        y0 += img.height() + 4;
    }
    out
}

/// 3 × 3 matrix helpers.
fn mul(a: &[[f32; 3]; 3], b: &[[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let mut c = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    c
}
fn apply(m: &[[f32; 3]; 3], v: [f32; 3]) -> [f32; 3] {
    m.map(|row| row[0] * v[0] + row[1] * v[1] + row[2] * v[2])
}
fn inverse(m: &[[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let a: Vec<Vec<f64>> = m.iter().map(|r| r.iter().map(|&v| v as f64).collect()).collect();
    let mut inv = [[0.0; 3]; 3];
    for j in 0..3 {
        let e: Vec<f64> = (0..3).map(|i| if i == j { 1.0 } else { 0.0 }).collect();
        let col = linalg::solve(&a, &e).expect("invertible");
        for i in 0..3 {
            inv[i][j] = col[i] as f32;
        }
    }
    inv
}

/// Simulate a colour image (encoded) through a matrix that acts on linear RGB.
fn simulate(img: &Rgb, m: &[[f32; 3]; 3]) -> Rgb {
    img.map(|c| apply(m, c.map(srgb_decode)).map(srgb_encode))
}

/// Two synthetic fluorescence channels: round nuclei, and thin filaments.
fn fluorescence(size: usize) -> (Image<f32>, Image<f32>) {
    let mut rng = Rng::new(3);
    let centres: Vec<(f32, f32)> = (0..9).map(|_| (20.0 + (size as f32 - 40.0) * rng.uniform(), 20.0 + (size as f32 - 40.0) * rng.uniform())).collect();
    let nuclei = Image::from_fn(size, size, |x, y| {
        centres.iter().map(|&(cx, cy)| (-((x as f32 - cx).powi(2) + (y as f32 - cy).powi(2)) / 120.0).exp()).fold(0.0, f32::max)
    });
    let mut rng = Rng::new(8);
    let lines: Vec<(f32, f32, f32)> = (0..7).map(|_| (std::f32::consts::PI * rng.uniform(), size as f32 * rng.uniform(), size as f32 * rng.uniform())).collect();
    let filaments = Image::from_fn(size, size, |x, y| {
        lines
            .iter()
            .map(|&(t, cx, cy)| {
                let d = (x as f32 - cx) * t.sin() - (y as f32 - cy) * t.cos();
                0.8 * (-d * d / 3.0).exp()
            })
            .fold(0.0, f32::max)
    });
    (nuclei, filaments)
}

/// A synthetic H&E slide by Beer–Lambert: hematoxylin in round nuclei, eosin everywhere
/// else at varying strength, two empty (white) holes, a little noise.
fn he_slide(size: usize) -> Rgb {
    let (h_vec, e_vec) = ([0.65, 0.70, 0.29], [0.07, 0.99, 0.11]);
    let mut rng = Rng::new(5);
    let nuclei: Vec<(f32, f32, f32)> = (0..40).map(|_| (size as f32 * rng.uniform(), size as f32 * rng.uniform(), 4.0 + 5.0 * rng.uniform())).collect();
    let s = size as f32;
    Image::from_fn(size, size, |x, y| {
        let (xf, yf) = (x as f32, y as f32);
        let hole = ((xf - 0.25 * s).powi(2) + (yf - 0.7 * s).powi(2)).sqrt() < 0.1 * s || ((xf - 0.75 * s).powi(2) + (yf - 0.3 * s).powi(2)).sqrt() < 0.07 * s;
        let h: f32 = nuclei.iter().map(|&(cx, cy, r)| if (xf - cx).powi(2) + (yf - cy).powi(2) < r * r { 0.9 } else { 0.0 }).fold(0.0, f32::max);
        let e = if hole { 0.0 } else { 0.5 + 0.25 * (xf / 23.0).sin() * (yf / 31.0).cos() };
        let h = if hole { 0.0 } else { h };
        [0, 1, 2].map(|k| srgb_encode(10f32.powf(-(h * h_vec[k] + e * e_vec[k])) + 0.004 * rng.normal()))
    })
}

pub fn main() {
    let dir = "out/colour-display";

    // 1. A smooth field with jet and with viridis, each with its colour bar.
    let field = hills(256);
    for (name, map) in [("jet", &JET[..]), ("viridis", &VIRIDIS[..])] {
        save_rgb(&stack(&[colour_map(&field, 0.0, 1.0, map), bar(map, 256)]), format!("{dir}/hills-{name}.png"));
        let profile: Vec<f32> = (0..=100).map(|i| lightness(colour_map(&Image::new(1, 1, i as f32 / 100.0), 0.0, 1.0, map)[(0, 0)])).collect();
        let (imax, lmax) = profile.iter().enumerate().fold((0, 0.0f32), |b, (i, &l)| if l > b.1 { (i, l) } else { b });
        println!(
            "{name:8}: L* at t = 0, 0.25, 0.5, 0.75, 1: {:.1} {:.1} {:.1} {:.1} {:.1}; largest {lmax:.1} at t = {:.2}; monotone: {}",
            profile[0], profile[25], profile[50], profile[75], profile[100], imax as f32 / 100.0, is_monotone(&profile)
        );
    }

    // 2. A signed field (values −3 .. 7) with the diverging map: range −3 .. 7 and −7 .. 7.
    let signed = field.map_xy(|x, y, v| 10.0 * v - 3.0 + 0.0 * (x + y) as f32);
    for (lo, hi) in [(-3.0, 7.0), (-7.0, 7.0)] {
        save_rgb(&stack(&[colour_map(&signed, lo, hi, &DIVERGING), bar(&DIVERGING, 256)]), format!("{dir}/signed{lo}..{hi}.png"));
    }
    println!("signed field: zero is at t = {:.3} with range −3..7, t = {:.3} with −7..7", position(0.0, -3.0, 7.0), position(0.0, -7.0, 7.0));

    // 3. Two channels merged red + green and magenta + green, for normal vision and as a
    //    deuteranope sees them. T: linear sRGB → LMS (Hunt–Pointer–Estévez of XYZ).
    let (nuc, fil) = fluorescence(192);
    let srgb_xyz = [[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]];
    let hpe = [[0.4002, 0.7076, -0.0808], [-0.2263, 1.1653, 0.0457], [0.0, 0.0, 0.9182]];
    let t = mul(&hpe, &srgb_xyz);
    let d = deutan_lms(apply(&t, [1.0, 1.0, 1.0]), apply(&t, [0.0, 0.0, 1.0]));
    let sim = mul(&inverse(&t), &mul(&d, &t));
    let mut rows = vec![];
    for (name, c1) in [("red", [1.0, 0.0, 0.0]), ("magenta", [1.0, 0.0, 1.0])] {
        let m = merge(&[nuc.clone(), fil.clone()], &[c1, [0.0, 1.0, 0.0]]);
        let s = simulate(&m, &sim);
        let row = Image::from_fn(2 * 192 + 4, 192, |x, y| if x < 192 { m[(x, y)] } else if x < 196 { [1.0; 3] } else { s[(x - 196, y)] });
        rows.push(row);
        let (pc, pg) = (simulate(&Image::new(1, 1, c1), &sim)[(0, 0)], simulate(&Image::new(1, 1, [0.0, 1.0, 0.0]), &sim)[(0, 0)]);
        println!("{name:7} + green, deuteranope: {name} looks like {:.0?}, green like {:.0?} (codes)", pc.map(|v| 255.0 * v), pg.map(|v| 255.0 * v));
    }
    save_rgb(&stack(&rows), format!("{dir}/merge-deutan.png"));

    // 4. Decorrelation stretch of a synthetic H&E image.
    let he = he_slide(256);
    save_rgb(&he, format!("{dir}/he.png"));
    let stretched = decorrelation_stretch(&he, 1.0);
    let clipped = stretched.pixels().iter().filter(|p| p.iter().any(|&v| !(0.0..=1.0).contains(&v))).count();
    save_rgb(&stretched, format!("{dir}/he-stretched.png"));
    // the covariance before and after, from the pixels
    for (name, img) in [("before", &he), ("after", &stretched)] {
        let n = (img.width() * img.height()) as f64;
        let mut mu = [0.0f64; 3];
        img.pixels().iter().for_each(|p| (0..3).for_each(|k| mu[k] += p[k] as f64 / n));
        let mut s = vec![vec![0.0f64; 3]; 3];
        for p in img.pixels() {
            for i in 0..3 {
                for j in 0..3 {
                    s[i][j] += (p[i] as f64 - mu[i]) * (p[j] as f64 - mu[j]) / n;
                }
            }
        }
        let (l, _) = linalg::sym_eigen(&s);
        let sd: Vec<f64> = l.iter().map(|v| v.max(0.0).sqrt()).collect();
        println!("H&E {name:6}: sd along the principal axes {:.3} {:.3} {:.3}, PC1 {:.1} % of the variance", sd[0], sd[1], sd[2], 100.0 * l[0] / (l[0] + l[1] + l[2]));
    }
    println!("stretched: {:.1} % of the pixels need clipping", 100.0 * clipped as f32 / (256 * 256) as f32);
    println!("wrote the pictures to {dir}/");
}
