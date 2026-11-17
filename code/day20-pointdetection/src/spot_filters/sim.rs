//! Test images and drawing for the spot exercises. Not part of the exercise: read it if you like.

use imgcourse::{Image, Rng};

/// A synthetic spot image: `n` Gaussian spots of width `sigma` (pixels) and peak amplitudes
/// between `amp.0` and `amp.1`, on the background `background(x, y)`, with Gaussian noise of
/// standard deviation `noise`. Returns the image and the true spot centres `[x, y]`.
pub fn spot_scene(
    size: usize,
    n: usize,
    sigma: f32,
    amp: (f32, f32),
    noise: f32,
    background: impl Fn(f32, f32) -> f32,
    seed: u64,
) -> (Image<f32>, Vec<[f32; 2]>) {
    let mut rng = Rng::new(seed);
    let margin = 3.0 * sigma;
    let spots: Vec<([f32; 2], f32)> = (0..n)
        .map(|_| {
            let x = margin + (size as f32 - 2.0 * margin) * rng.uniform();
            let y = margin + (size as f32 - 2.0 * margin) * rng.uniform();
            ([x, y], amp.0 + (amp.1 - amp.0) * rng.uniform())
        })
        .collect();
    let mut img = Image::from_fn(size, size, |x, y| background(x as f32, y as f32));
    let reach = (4.0 * sigma).ceil() as isize;
    for &([sx, sy], a) in &spots {
        let (cx, cy) = (sx.round() as isize, sy.round() as isize);
        for y in cy - reach..=cy + reach {
            for x in cx - reach..=cx + reach {
                if img.contains(x, y) {
                    let d2 = (x as f32 - sx).powi(2) + (y as f32 - sy).powi(2);
                    img[(x as usize, y as usize)] += a * (-d2 / (2.0 * sigma * sigma)).exp();
                }
            }
        }
    }
    for v in img.pixels_mut() {
        *v += noise * rng.normal();
    }
    (img, spots.into_iter().map(|(p, _)| p).collect())
}

/// The noise estimate of the next unit: median absolute deviation / 0.6745.
pub fn mad_sigma(values: &[f32]) -> f32 {
    let median = |v: &mut Vec<f32>| {
        v.sort_by(|a, b| a.total_cmp(b));
        let m = v.len() / 2;
        if v.len() % 2 == 1 { v[m] } else { 0.5 * (v[m - 1] + v[m]) }
    };
    let mut v = values.to_vec();
    let med = median(&mut v);
    let mut dev: Vec<f32> = values.iter().map(|x| (x - med).abs()).collect();
    median(&mut dev) / 0.6745
}

/// The noise of an image, estimated from differences of horizontal neighbours (a high-pass
/// image: the background drops out, the noise variance doubles).
pub fn image_noise(img: &Image<f32>) -> f32 {
    let mut d = Vec::new();
    for y in 0..img.height() {
        for x in 0..img.width() - 1 {
            d.push(img[(x + 1, y)] - img[(x, y)]);
        }
    }
    mad_sigma(&d) / 2f32.sqrt()
}

/// Count true positives, false positives and false negatives: each detection is matched to
/// the closest still unmatched true spot within `gate` pixels (closest pairs first).
pub fn score(detections: &[(usize, usize)], truth: &[[f32; 2]], gate: f32) -> (usize, usize, usize) {
    let mut pairs = Vec::new();
    for (i, t) in truth.iter().enumerate() {
        for (j, &(x, y)) in detections.iter().enumerate() {
            let d = ((x as f32 - t[0]).powi(2) + (y as f32 - t[1]).powi(2)).sqrt();
            if d <= gate {
                pairs.push((d, i, j));
            }
        }
    }
    pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
    let (mut used_t, mut used_d) = (vec![false; truth.len()], vec![false; detections.len()]);
    let mut tp = 0;
    for (_, i, j) in pairs {
        if !used_t[i] && !used_d[j] {
            used_t[i] = true;
            used_d[j] = true;
            tp += 1;
        }
    }
    (tp, detections.len() - tp, truth.len() - tp)
}

/// The image in grey with the detections drawn as small crosses (green near a true spot, red
/// otherwise) and the missed true spots as yellow rings.
pub fn draw(img: &Image<f32>, detections: &[(usize, usize)], truth: &[[f32; 2]], gate: f32) -> Image<[f32; 3]> {
    let (lo, hi) = (img.min(), img.max());
    let mut out = img.map(|v| {
        let g = (v - lo) / (hi - lo).max(1e-6);
        [g, g, g]
    });
    let put = |out: &mut Image<[f32; 3]>, x: isize, y: isize, c: [f32; 3]| {
        if out.contains(x, y) {
            out[(x as usize, y as usize)] = c;
        }
    };
    let near = |x: f32, y: f32, pts: &[[f32; 2]]| pts.iter().any(|p| (p[0] - x).hypot(p[1] - y) <= gate);
    let det: Vec<[f32; 2]> = detections.iter().map(|&(x, y)| [x as f32, y as f32]).collect();
    for &(x, y) in detections {
        let ok = near(x as f32, y as f32, truth);
        let c = if ok { [0.1, 0.9, 0.1] } else { [1.0, 0.1, 0.1] };
        for d in -2..=2isize {
            put(&mut out, x as isize + d, y as isize, c);
            put(&mut out, x as isize, y as isize + d, c);
        }
    }
    for t in truth {
        if !near(t[0], t[1], &det) {
            for k in 0..24 {
                let a = k as f32 * std::f32::consts::TAU / 24.0;
                put(&mut out, (t[0] + 4.0 * a.cos()).round() as isize, (t[1] + 4.0 * a.sin()).round() as isize, [1.0, 0.9, 0.0]);
            }
        }
    }
    out
}
