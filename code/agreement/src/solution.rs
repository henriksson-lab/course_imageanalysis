// Model solution for the coding exercises of "Boundaries, counts and the uncertain ground truth".

use imgcourse::Image;

/// C1. The boundary pixels of a mask: object pixels with at least one of their 4 neighbours
/// outside the object (pixels outside the image count as outside). Returned as points
/// `(x, y)`.
pub fn boundary(mask: &Image<bool>) -> Vec<(f32, f32)> {
    let mut points = Vec::new();
    for (x, y) in mask.coords() {
        if !mask[(x, y)] {
            continue;
        }
        let mut edge = false;
        for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
            let (nx, ny) = (x as isize + dx, y as isize + dy);
            if !mask.contains(nx, ny) || !mask[(nx as usize, ny as usize)] {
                edge = true;
            }
        }
        if edge {
            points.push((x as f32, y as f32));
        }
    }
    points
}

/// C2. For every point of `a`, the distance to the nearest point of `b` (by trying them all).
pub fn nearest_distances(a: &[(f32, f32)], b: &[(f32, f32)]) -> Vec<f32> {
    let mut d = Vec::with_capacity(a.len());
    for &(ax, ay) in a {
        let mut best = f32::INFINITY;
        for &(bx, by) in b {
            best = best.min(((ax - bx).powi(2) + (ay - by).powi(2)).sqrt());
        }
        d.push(best);
    }
    d
}

/// Boundary distances between two outlines.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoundaryScores {
    pub hausdorff: f32,
    pub hd95: f32,
    pub asd: f32,
}

/// C3. Hausdorff distance (the largest of all nearest distances, both ways), HD95 (the
/// larger of the two 95th percentiles) and ASD (the mean of all nearest distances of both
/// outlines together). The 95th percentile of `n` sorted values is the value number
/// `ceil(0.95 n)` (counting from 1).
pub fn boundary_scores(a: &[(f32, f32)], b: &[(f32, f32)]) -> BoundaryScores {
    let mut ab = nearest_distances(a, b);
    let mut ba = nearest_distances(b, a);
    let percentile95 = |d: &mut Vec<f32>| {
        d.sort_by(|p, q| p.total_cmp(q));
        let rank = (0.95 * d.len() as f32).ceil() as usize;
        d[rank.max(1) - 1]
    };
    let sum: f32 = ab.iter().sum::<f32>() + ba.iter().sum::<f32>();
    let asd = sum / (ab.len() + ba.len()) as f32;
    let hausdorff = ab.iter().chain(ba.iter()).fold(0.0f32, |m, &v| m.max(v));
    let hd95 = percentile95(&mut ab).max(percentile95(&mut ba));
    BoundaryScores { hausdorff, hd95, asd }
}

/// The Bland–Altman summary of a method against a reference.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BlandAltman {
    pub bias: f32,
    pub sd: f32,
    pub lower: f32,
    pub upper: f32,
}

/// C4. Bland–Altman: the differences `method - reference`, their mean (the bias), their
/// standard deviation (divide by n - 1) and the limits of agreement `bias ± 1.96 sd`.
pub fn bland_altman(reference: &[f32], method: &[f32]) -> BlandAltman {
    let n = reference.len() as f32;
    let d: Vec<f32> = reference.iter().zip(method).map(|(r, m)| m - r).collect();
    let bias = d.iter().sum::<f32>() / n;
    let sd = (d.iter().map(|v| (v - bias).powi(2)).sum::<f32>() / (n - 1.0)).sqrt();
    BlandAltman { bias, sd, lower: bias - 1.96 * sd, upper: bias + 1.96 * sd }
}

/// C5. The correlation coefficient r of two lists:
/// `sum((x - mean x)(y - mean y)) / sqrt(sum((x - mean x)^2) sum((y - mean y)^2))`.
pub fn correlation(x: &[f32], y: &[f32]) -> f32 {
    let n = x.len() as f32;
    let (mx, my) = (x.iter().sum::<f32>() / n, y.iter().sum::<f32>() / n);
    let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
    for i in 0..x.len() {
        sxy += (x[i] - mx) * (y[i] - my);
        sxx += (x[i] - mx).powi(2);
        syy += (y[i] - my).powi(2);
    }
    sxy / (sxx * syy).sqrt()
}
