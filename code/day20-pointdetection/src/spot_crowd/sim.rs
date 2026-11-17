//! Test images, a spot detector, grouping for the matching, and drawing. Not part of the exercise.

use imgcourse::{Border, Image, Rng};

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

/// A spot detector as in the first two units: the zero-mean matched filter (Gaussian of width
/// `sigma` minus its mean, 11 × 11 for σ = 1.5), local maxima in 5 × 5 windows, kept if above
/// `k` noise sd (noise from the MAD of neighbour differences). Returns `(x, y, score)` with the
/// score in noise sd, strongest first.
pub fn detect(img: &Image<f32>, sigma: f32, k: f32) -> Vec<(f32, f32, f32)> {
    let r = (3.0 * sigma).ceil() as isize;
    let mut w = Vec::new();
    for dy in -r..=r {
        for dx in -r..=r {
            w.push(((dx, dy), (-((dx * dx + dy * dy) as f32) / (2.0 * sigma * sigma)).exp()));
        }
    }
    let sum: f32 = w.iter().map(|p| p.1).sum();
    let mean = 1.0 / w.len() as f32;
    for p in w.iter_mut() {
        p.1 = p.1 / sum - mean;
    }
    let factor = w.iter().map(|p| p.1 * p.1).sum::<f32>().sqrt();
    let resp = img.map_xy(|x, y, _| {
        w.iter().map(|&((dx, dy), wt)| wt * img.get_border(x as isize + dx, y as isize + dy, Border::Mirror)).sum::<f32>()
    });
    let mut d = Vec::new();
    for y in 0..img.height() {
        for x in 0..img.width() - 1 {
            d.push(img[(x + 1, y)] - img[(x, y)]);
        }
    }
    let noise = mad_sigma(&d) / 2f32.sqrt() * factor;
    let mut out = Vec::new();
    for (x, y) in resp.coords() {
        let v = resp[(x, y)];
        if v <= k * noise {
            continue;
        }
        let mut is_max = true;
        for dy in -2..=2isize {
            for dx in -2..=2isize {
                let (nx, ny) = (x as isize + dx, y as isize + dy);
                if (dx, dy) != (0, 0) && resp.contains(nx, ny) {
                    let u = resp[(nx as usize, ny as usize)];
                    if u > v || (u == v && (dy < 0 || (dy == 0 && dx < 0))) {
                        is_max = false;
                    }
                }
            }
        }
        if is_max {
            out.push((x as f32, y as f32, v / noise));
        }
    }
    out.sort_by(|a, b| b.2.total_cmp(&a.2));
    out
}

/// The median absolute deviation / 0.6745 (the threshold unit's noise estimate).
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

/// Split the matching into small independent groups: two points are in the same group if they
/// are linked by a chain of truth–detection pairs within the gate. `matcher` is run on each
/// group, and its pairs are translated back to the full lists.
pub fn match_in_groups(
    truth: &[[f32; 2]],
    det: &[[f32; 2]],
    gate: f32,
    matcher: impl Fn(&[[f32; 2]], &[[f32; 2]], f32) -> Vec<(usize, usize)>,
) -> Vec<(usize, usize)> {
    // union–find over truths 0..nt and detections nt..nt+nd
    let (nt, nd) = (truth.len(), det.len());
    let mut parent: Vec<usize> = (0..nt + nd).collect();
    fn root(p: &mut [usize], mut i: usize) -> usize {
        while p[i] != i {
            p[i] = p[p[i]];
            i = p[i];
        }
        i
    }
    // a coarse grid of cells of the gate's size, so that only nearby pairs are compared
    let cell = |p: [f32; 2]| ((p[0] / gate).floor() as i64, (p[1] / gate).floor() as i64);
    let mut grid: std::collections::HashMap<(i64, i64), Vec<usize>> = std::collections::HashMap::new();
    for (j, d) in det.iter().enumerate() {
        grid.entry(cell(*d)).or_default().push(j);
    }
    for (i, t) in truth.iter().enumerate() {
        let (cx, cy) = cell(*t);
        for gx in cx - 1..=cx + 1 {
            for gy in cy - 1..=cy + 1 {
                for &j in grid.get(&(gx, gy)).map(|v| v.as_slice()).unwrap_or(&[]) {
                    if (t[0] - det[j][0]).hypot(t[1] - det[j][1]) <= gate {
                        let (a, b) = (root(&mut parent, i), root(&mut parent, nt + j));
                        parent[a] = b;
                    }
                }
            }
        }
    }
    let mut groups: std::collections::BTreeMap<usize, (Vec<usize>, Vec<usize>)> = Default::default();
    for i in 0..nt {
        let r = root(&mut parent, i);
        groups.entry(r).or_default().0.push(i);
    }
    for j in 0..nd {
        let r = root(&mut parent, nt + j);
        groups.entry(r).or_default().1.push(j);
    }
    let mut pairs = Vec::new();
    for (ti, dj) in groups.values() {
        if ti.is_empty() || dj.is_empty() {
            continue;
        }
        let t: Vec<[f32; 2]> = ti.iter().map(|&i| truth[i]).collect();
        let d: Vec<[f32; 2]> = dj.iter().map(|&j| det[j]).collect();
        for (a, b) in matcher(&t, &d, gate) {
            pairs.push((ti[a], dj[b]));
        }
    }
    pairs
}

/// The image in grey; true spots as rings (white = matched, yellow = missed), detections as
/// dots (green = matched, red = false).
pub fn draw(img: &Image<f32>, truth: &[[f32; 2]], det: &[[f32; 2]], pairs: &[(usize, usize)]) -> Image<[f32; 3]> {
    let (lo, hi) = (img.min(), img.max());
    let mut out = img.map(|v| {
        let g = 0.8 * (v - lo) / (hi - lo).max(1e-6);
        [g, g, g]
    });
    let put = |out: &mut Image<[f32; 3]>, x: f32, y: f32, c: [f32; 3]| {
        let (x, y) = (x.round() as isize, y.round() as isize);
        if out.contains(x, y) {
            out[(x as usize, y as usize)] = c;
        }
    };
    let mut t_ok = vec![false; truth.len()];
    let mut d_ok = vec![false; det.len()];
    for &(i, j) in pairs {
        t_ok[i] = true;
        d_ok[j] = true;
    }
    for (t, ok) in truth.iter().zip(&t_ok) {
        let c = if *ok { [1.0, 1.0, 1.0] } else { [1.0, 0.9, 0.0] };
        for k in 0..20 {
            let a = k as f32 * std::f32::consts::TAU / 20.0;
            put(&mut out, t[0] + 3.0 * a.cos(), t[1] + 3.0 * a.sin(), c);
        }
    }
    for (d, ok) in det.iter().zip(&d_ok) {
        let c = if *ok { [0.1, 0.9, 0.1] } else { [1.0, 0.1, 0.1] };
        for (dx, dy) in [(0.0, 0.0), (1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)] {
            put(&mut out, d[0] + dx, d[1] + dy, c);
        }
    }
    out
}
