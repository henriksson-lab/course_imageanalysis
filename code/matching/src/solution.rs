// Model solution for the coding exercises of "Finding a shift by sliding and scoring".

use imgcourse::Image;

/// The three scores of one shift, over the overlap of `n` pixel pairs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Scores {
    pub ssd: f32,
    pub cc: f32,
    pub ncc: f32,
    pub n: usize,
}

/// C1. SSD, CC and NCC of B placed at shift `t` on A (B's sample `x` lies on A's sample
/// `x + t`), over the overlap only (slide "Three scores"). NCC is 0 when one side of the
/// overlap is flat (no spread, exercise B2).
pub fn scores_1d(a: &[f32], b: &[f32], t: isize) -> Scores {
    let pairs: Vec<(f32, f32)> = (0..b.len())
        .filter(|&x| x as isize + t >= 0 && ((x as isize + t) as usize) < a.len())
        .map(|x| (a[(x as isize + t) as usize], b[x]))
        .collect();
    let n = pairs.len();
    let (mut ssd, mut cc) = (0.0, 0.0);
    for &(f, g) in &pairs {
        ssd += (f - g) * (f - g);
        cc += f * g;
    }
    let mf = pairs.iter().map(|p| p.0).sum::<f32>() / n.max(1) as f32;
    let mg = pairs.iter().map(|p| p.1).sum::<f32>() / n.max(1) as f32;
    let (mut sfg, mut sff, mut sgg) = (0.0, 0.0, 0.0);
    for &(f, g) in &pairs {
        sfg += (f - mf) * (g - mg);
        sff += (f - mf) * (f - mf);
        sgg += (g - mg) * (g - mg);
    }
    let ncc = if sff > 0.0 && sgg > 0.0 { sfg / (sff * sgg).sqrt() } else { 0.0 };
    Scores { ssd, cc, ncc, n }
}

/// C2. NCC of image B placed at shift `(tx, ty)` on image A (B's pixel `(x, y)` lies on A's
/// pixel `(x + tx, y + ty)`), over the overlap. Returns `None` if the overlap has fewer than
/// `min_overlap` pixels; 0 if one side is flat.
pub fn ncc_2d(a: &Image<f32>, b: &Image<f32>, tx: isize, ty: isize, min_overlap: usize) -> Option<f32> {
    let (mut n, mut sf, mut sg) = (0usize, 0.0f64, 0.0f64);
    let (mut sff, mut sgg, mut sfg) = (0.0f64, 0.0f64, 0.0f64);
    for (x, y) in b.coords() {
        let (ax, ay) = (x as isize + tx, y as isize + ty);
        if !a.contains(ax, ay) {
            continue;
        }
        let f = a[(ax as usize, ay as usize)] as f64;
        let g = b[(x, y)] as f64;
        n += 1;
        sf += f;
        sg += g;
        sff += f * f;
        sgg += g * g;
        sfg += f * g;
    }
    if n < min_overlap {
        return None;
    }
    // sums of products of the deviations from the means, from the plain sums
    let nn = n as f64;
    let (vf, vg) = (sff - sf * sf / nn, sgg - sg * sg / nn);
    let cov = sfg - sf * sg / nn;
    Some(if vf > 1e-12 && vg > 1e-12 { (cov / (vf * vg).sqrt()) as f32 } else { 0.0 })
}

/// C3. Brute-force registration: try every shift with `|tx|, |ty| <= w` and keep the one with
/// the largest NCC (at least `min_overlap` pixels). Returns `(tx, ty, ncc)`.
pub fn register(a: &Image<f32>, b: &Image<f32>, w: isize, min_overlap: usize) -> (isize, isize, f32) {
    let mut best = (0, 0, f32::NEG_INFINITY);
    for ty in -w..=w {
        for tx in -w..=w {
            if let Some(c) = ncc_2d(a, b, tx, ty, min_overlap) {
                if c > best.2 {
                    best = (tx, ty, c);
                }
            }
        }
    }
    best
}

/// C4. Template matching: the NCC of the template at every position where it fits completely
/// in the image. `out[(x, y)]` belongs to the template's top-left corner at `(x, y)`; the
/// output is `(W − w + 1) × (H − h + 1)` (slide "Template matching").
pub fn template_match(img: &Image<f32>, tpl: &Image<f32>) -> Image<f32> {
    let (w, h) = (img.width() + 1 - tpl.width(), img.height() + 1 - tpl.height());
    Image::from_fn(w, h, |x, y| ncc_2d(img, tpl, x as isize, y as isize, 1).unwrap_or(0.0))
}
