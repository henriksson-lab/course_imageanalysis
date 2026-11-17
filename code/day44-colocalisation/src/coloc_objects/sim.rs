//! Simulated spot positions in two cells, and a picture of them.
//! Not part of the exercise: read it if you like.

use imgcourse::{Image, Rng};

/// A spot position (x, y), in nanometres (or pixels in the small examples).
pub type Point = (f32, f32);

/// The pixel size of the simulated images, in nm.
pub const PIXEL: f32 = 65.0;

/// The two cells: two ellipses that fill most of a `size × size` image (in `src/common.rs`).
pub use crate::common::cells;

/// The area of the mask in nm².
pub fn area(mask: &Image<bool>) -> f32 {
    mask.pixels().iter().filter(|&&m| m).count() as f32 * PIXEL * PIXEL
}

/// Is the point (in nm) inside the mask?
pub fn in_mask(mask: &Image<bool>, p: Point) -> bool {
    let (x, y) = ((p.0 / PIXEL).floor() as isize, (p.1 / PIXEL).floor() as isize);
    mask.contains(x, y) && mask[(x as usize, y as usize)]
}

/// Two channels of spots: `n_a` A spots at random in the mask; a share `paired` of them has a
/// B partner at `distance` nm in a random direction; plus `extra_b` independent B spots. Every
/// measured position has a Gaussian localisation error of `precision` nm per axis.
pub fn spots(mask: &Image<bool>, n_a: usize, paired: f32, distance: f32, extra_b: usize, precision: f32, seed: u64) -> (Vec<Point>, Vec<Point>) {
    let mut rng = Rng::new(seed);
    let random_in_mask = |rng: &mut Rng| loop {
        let p = (rng.uniform() * mask.width() as f32 * PIXEL, rng.uniform() * mask.height() as f32 * PIXEL);
        if in_mask(mask, p) {
            return p;
        }
    };
    let truth_a: Vec<Point> = (0..n_a).map(|_| random_in_mask(&mut rng)).collect();
    let mut truth_b = Vec::new();
    for &(x, y) in &truth_a {
        if rng.uniform() < paired {
            let t = std::f32::consts::TAU * rng.uniform();
            truth_b.push((x + distance * t.cos(), y + distance * t.sin()));
        }
    }
    for _ in 0..extra_b {
        truth_b.push(random_in_mask(&mut rng));
    }
    let mut jitter = |p: &Point| (p.0 + precision * rng.normal(), p.1 + precision * rng.normal());
    let a = truth_a.iter().map(&mut jitter).collect();
    let b = truth_b.iter().map(&mut jitter).collect();
    (a, b)
}

/// Picture of the spots, `scale` image pixels per nm-pixel: cells dark grey, A spots green dots,
/// B spots magenta rings, and a yellow line from each A spot to its nearest B spot if it is within
/// `d_max` nm.
pub fn picture(mask: &Image<bool>, a: &[Point], b: &[Point], d_max: f32, scale: usize) -> Image<[f32; 3]> {
    let (w, h) = (mask.width() * scale, mask.height() * scale);
    let mut img = Image::from_fn(w, h, |x, y| if mask[(x / scale, y / scale)] { [0.18; 3] } else { [0.0; 3] });
    let to_px = |p: Point| (p.0 / PIXEL * scale as f32, p.1 / PIXEL * scale as f32);
    // a disc (inner radius 0) or a ring
    let dot = |img: &mut Image<[f32; 3]>, p: Point, c: [f32; 3], r_in: isize, r_out: isize| {
        let (cx, cy) = to_px(p);
        for dy in -r_out..=r_out {
            for dx in -r_out..=r_out {
                let (x, y) = (cx.round() as isize + dx, cy.round() as isize + dy);
                let d2 = dx * dx + dy * dy;
                if d2 >= r_in * r_in && d2 <= r_out * r_out && img.contains(x, y) {
                    img[(x as usize, y as usize)] = c;
                }
            }
        }
    };
    for &p in b {
        dot(&mut img, p, [0.95, 0.2, 0.95], 4, 5);
    }
    for &p in a {
        dot(&mut img, p, [0.2, 0.95, 0.2], 0, 2);
    }
    for &pa in a {
        let nearest = b.iter().copied().min_by(|p, q| dist(pa, *p).total_cmp(&dist(pa, *q)));
        if let Some(pb) = nearest {
            if dist(pa, pb) <= d_max {
                let ((x0, y0), (x1, y1)) = (to_px(pa), to_px(pb));
                for k in 0..=20 {
                    let t = k as f32 / 20.0;
                    let (x, y) = ((x0 + t * (x1 - x0)).round() as isize, (y0 + t * (y1 - y0)).round() as isize);
                    if img.contains(x, y) {
                        img[(x as usize, y as usize)] = [1.0, 0.9, 0.1];
                    }
                }
            }
        }
    }
    img
}

/// The distance between two points.
pub fn dist(p: Point, q: Point) -> f32 {
    ((p.0 - q.0).powi(2) + (p.1 - q.1).powi(2)).sqrt()
}
