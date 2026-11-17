// Model solution for the coding exercises of "Hough transform: circles".
// Positions are pixel coordinates (x, y), y down: the accumulator slice for one radius is an
// image of the same size as the input, and its value at (a, b) counts the votes for the centre (a, b).

use crate::sim::EdgePoint;
use imgcourse::Image;

/// Which way the gradient points at the edge of the objects we look for.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Polarity {
    /// bright object on a dark background: the gradient points inward, to the centre
    BrightOnDark,
    /// dark object on a bright background: the gradient points outward
    DarkOnBright,
    /// unknown: vote on both sides
    Either,
}

/// C1. Full voting for one radius `r`: every edge point adds one vote to every pixel whose
/// distance from it is within half a pixel of `r` (a ring of about 2πr pixels around it).
pub fn ring_votes(edges: &[EdgePoint], width: usize, height: usize, r: f32) -> Image<f32> {
    let mut acc = Image::new(width, height, 0.0);
    let reach = r.ceil() as isize + 1;
    for e in edges {
        let (px, py) = (e.x.round() as isize, e.y.round() as isize);
        for dy in -reach..=reach {
            for dx in -reach..=reach {
                let d = ((dx * dx + dy * dy) as f32).sqrt();
                if (d - r).abs() < 0.5 && acc.contains(px + dx, py + dy) {
                    acc[((px + dx) as usize, (py + dy) as usize)] += 1.0;
                }
            }
        }
    }
    acc
}

/// C2. Gradient voting for one radius `r`: with the unit gradient `u = (cos θ, sin θ)`, a bright
/// object has its centre at `p + r u`, a dark object at `p − r u`; `Either` votes at both.
/// Each vote goes to the nearest pixel (if inside the image).
pub fn gradient_votes(edges: &[EdgePoint], width: usize, height: usize, r: f32, polarity: Polarity) -> Image<f32> {
    let mut acc = Image::new(width, height, 0.0);
    let signs: &[f32] = match polarity {
        Polarity::BrightOnDark => &[1.0],
        Polarity::DarkOnBright => &[-1.0],
        Polarity::Either => &[1.0, -1.0],
    };
    for e in edges {
        for &s in signs {
            let cx = (e.x + s * r * e.theta.cos()).round() as isize;
            let cy = (e.y + s * r * e.theta.sin()).round() as isize;
            if acc.contains(cx, cy) {
                acc[(cx as usize, cy as usize)] += 1.0;
            }
        }
    }
    acc
}

/// A detected circle.
#[derive(Clone, Copy, Debug)]
pub struct Circle {
    pub x: usize,
    pub y: usize,
    pub r: f32,
    /// votes / (2π r): the fraction of a full circumference that voted
    pub score: f32,
}

/// C3. The circles in a 3D accumulator (`slices[i]` belongs to `radii[i]`): divide every slice
/// by 2πr, then keep the bins that are at least `threshold` and the largest in a window of
/// ±2 pixels and ±1 radius around them (of equal values, the first one: smaller radius, then
/// raster order). Strongest first.
pub fn find_circles(slices: &[Image<f32>], radii: &[f32], threshold: f32) -> Vec<Circle> {
    let tau = 2.0 * std::f32::consts::PI;
    let norm: Vec<Image<f32>> = slices.iter().zip(radii).map(|(s, &r)| s.map(|v| v / (tau * r))).collect();
    let mut out = Vec::new();
    for (i, slice) in norm.iter().enumerate() {
        for (x, y) in slice.coords() {
            let v = slice[(x, y)];
            if v < threshold {
                continue;
            }
            let mut is_max = true;
            for di in -1..=1isize {
                let ii = i as isize + di;
                if ii < 0 || ii as usize >= norm.len() {
                    continue;
                }
                for dy in -2..=2isize {
                    for dx in -2..=2isize {
                        let (nx, ny) = (x as isize + dx, y as isize + dy);
                        if (di, dx, dy) == (0, 0, 0) || !slice.contains(nx, ny) {
                            continue;
                        }
                        let u = norm[ii as usize][(nx as usize, ny as usize)];
                        let earlier = (di, dy, dx) < (0, 0, 0);
                        if u > v || (u == v && earlier) {
                            is_max = false;
                        }
                    }
                }
            }
            if is_max {
                out.push(Circle { x, y, r: radii[i], score: v });
            }
        }
    }
    out.sort_by(|a, b| b.score.total_cmp(&a.score));
    out
}
