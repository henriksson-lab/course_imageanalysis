// Model solution for the coding exercises of "Contrast and intensity transforms".
//
// Images here hold whole numbers (`u32`): grey levels 0 … L − 1, with L = 256 for 8 bits or
// 4096 for 12 bits. A curve is a lookup table (LUT): `lut[v]` is the new value of `v`.

use imgcourse::Image;

/// C1a. Apply a lookup table to every pixel: the new value of `v` is `lut[v]`.
pub fn apply_lut(img: &Image<u32>, lut: &[u32]) -> Image<u32> {
    img.map(|v| lut[v as usize])
}

/// C1b. The gamma curve as a table from `levels_in` to `levels_out` levels: scale v to [0, 1],
/// raise it to the power γ, scale to 0 … levels_out − 1 and round.
pub fn gamma_lut(levels_in: usize, levels_out: usize, gamma: f32) -> Vec<u32> {
    let (top_in, top_out) = ((levels_in - 1) as f32, (levels_out - 1) as f32);
    (0..levels_in).map(|v| (top_out * (v as f32 / top_in).powf(gamma)).round() as u32).collect()
}

/// C1c. The log curve as a table: s = c ln(1 + v), with c chosen so that the largest input
/// value maps to the largest output value: c = (levels_out − 1) / ln(levels_in).
pub fn log_lut(levels_in: usize, levels_out: usize) -> Vec<u32> {
    let c = (levels_out - 1) as f32 / (levels_in as f32).ln();
    (0..levels_in).map(|v| (c * (1.0 + v as f32).ln()).round() as u32).collect()
}

/// C2a. The histogram: how many pixels have each value 0 … levels − 1.
pub fn histogram(img: &Image<u32>, levels: usize) -> Vec<u32> {
    let mut h = vec![0; levels];
    for &v in img.pixels() {
        h[v as usize] += 1;
    }
    h
}

/// C2b. The equalisation curve: s_k = round((L − 1) c_k / N), where c_k is the cumulative
/// count (pixels with value ≤ k), N the number of pixels and L the number of levels.
pub fn equalise_lut(hist: &[u32]) -> Vec<u32> {
    let n: u32 = hist.iter().sum();
    let top = (hist.len() - 1) as f32;
    let mut cumulative = 0;
    let mut lut = Vec::with_capacity(hist.len());
    for &h in hist {
        cumulative += h;
        lut.push((top * cumulative as f32 / n as f32).round() as u32);
    }
    lut
}

/// C3. Histogram matching: for each input level k, the smallest target level j whose
/// cumulative fraction G_j reaches the input's cumulative fraction F_k (T = G⁻¹ ∘ F).
pub fn match_lut(hist: &[u32], target: &[u32]) -> Vec<u32> {
    let cumulative_fraction = |h: &[u32]| {
        let n: u32 = h.iter().sum();
        let mut c = 0;
        h.iter()
            .map(|&v| {
                c += v;
                c as f32 / n as f32
            })
            .collect::<Vec<f32>>()
    };
    let (f, g) = (cumulative_fraction(hist), cumulative_fraction(target));
    f.iter()
        .map(|&fk| {
            // A tiny tolerance, so that fractions that are equal on paper but rounded
            // differently still match.
            let j = g.iter().position(|&gj| gj >= fk - 1e-6).unwrap_or(g.len() - 1);
            j as u32
        })
        .collect()
}

/// C4. CLAHE's clipped histogram: cut every bin at β × the mean count, spread the excess
/// evenly over all bins, and repeat until no bin is above the limit (at most 100 times).
pub fn clip_histogram(hist: &[f32], beta: f32) -> Vec<f32> {
    let n: f32 = hist.iter().sum();
    let bins = hist.len() as f32;
    let limit = beta * n / bins;
    let mut h = hist.to_vec();
    for _ in 0..100 {
        let excess: f32 = h.iter().map(|&v| (v - limit).max(0.0)).sum();
        if excess <= 1e-6 * n {
            break;
        }
        for v in &mut h {
            *v = v.min(limit) + excess / bins;
        }
    }
    h
}

/// C5. CLAHE: cut the image into `tiles × tiles` tiles; for each tile, the equalisation curve
/// of its clipped histogram, T(v) = (L − 1)/N_tile · (sum of the clipped counts up to v). Each
/// pixel blends the curves of the four nearest tile centres with bilinear weights.
pub fn clahe(img: &Image<u32>, levels: usize, tiles: usize, beta: f32) -> Image<u32> {
    let (tw, th) = (img.width() / tiles, img.height() / tiles);
    let top = (levels - 1) as f32;
    // One curve per tile (the last row and column of tiles take any leftover pixels).
    let mut curves = Vec::new();
    for ty in 0..tiles {
        for tx in 0..tiles {
            let x_end = if tx + 1 == tiles { img.width() } else { (tx + 1) * tw };
            let y_end = if ty + 1 == tiles { img.height() } else { (ty + 1) * th };
            let tile = img.crop(tx * tw, ty * th, x_end - tx * tw, y_end - ty * th);
            let h: Vec<f32> = histogram(&tile, levels).iter().map(|&c| c as f32).collect();
            let h = clip_histogram(&h, beta);
            let n = tile.pixels().len() as f32;
            let mut c = 0.0;
            let curve: Vec<f32> = h
                .iter()
                .map(|&v| {
                    c += v;
                    top * c / n
                })
                .collect();
            curves.push(curve);
        }
    }
    // Blend: the position of the pixel among the tile centres, which sit at (i + ½) × tile size.
    let nearest = |p: usize, size: usize| {
        let f = (p as f32 + 0.5) / size as f32 - 0.5;
        let i0 = (f.floor().max(0.0) as usize).min(tiles - 1);
        let i1 = (i0 + 1).min(tiles - 1);
        let a = (f - i0 as f32).clamp(0.0, 1.0);
        (i0, i1, a)
    };
    img.map_xy(|x, y, v| {
        let (x0, x1, a) = nearest(x, tw);
        let (y0, y1, b) = nearest(y, th);
        let t = |tx: usize, ty: usize| curves[ty * tiles + tx][v as usize];
        let s = (1.0 - a) * (1.0 - b) * t(x0, y0) + a * (1.0 - b) * t(x1, y0) + (1.0 - a) * b * t(x0, y1) + a * b * t(x1, y1);
        s.round() as u32
    })
}
