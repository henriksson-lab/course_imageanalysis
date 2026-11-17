//! Synthetic two-channel images (two cells with small structures) and pictures of the results.
//! Not part of the exercise: read it if you like.

use imgcourse::{Image, Rng};

/// What the two channels contain. Grey values are in [0, 1].
#[derive(Clone, Debug)]
pub struct Scene {
    /// Image width and height in pixels.
    pub size: usize,
    /// Number of structures per channel.
    pub spots: usize,
    /// Share of B's structures that sit exactly on A's (0 = independent, 1 = all shared).
    pub shared: f32,
    /// B has a few large structures, each containing some of A's small spots.
    pub a_inside_b: bool,
    /// Bleed-through: this share of A's light is added to channel B.
    pub leak: f32,
    /// A diffuse signal everywhere in the cells, in both channels.
    pub diffuse: f32,
    /// Width (sigma, px) of the blur by the optics.
    pub blur: f32,
    /// Shift of channel B in pixels (x, y), may be fractional.
    pub shift: (f32, f32),
    /// Background offset in both channels.
    pub offset: f32,
    /// Standard deviation of the Gaussian noise.
    pub noise: f32,
    pub seed: u64,
}

impl Default for Scene {
    fn default() -> Self {
        Scene { size: 128, spots: 40, shared: 1.0, a_inside_b: false, leak: 0.0, diffuse: 0.0, blur: 1.0, shift: (0.0, 0.0), offset: 0.04, noise: 0.015, seed: 1 }
    }
}

/// The two cells: two ellipses that fill most of the image.
pub fn cells(size: usize) -> Image<bool> {
    let s = size as f32;
    Image::from_fn(size, size, |x, y| {
        let (u, v) = (x as f32 / s, y as f32 / s);
        let e1 = ((u - 0.30) / 0.24).powi(2) + ((v - 0.50) / 0.36).powi(2) <= 1.0;
        let e2 = ((u - 0.73) / 0.22).powi(2) + ((v - 0.47) / 0.32).powi(2) <= 1.0;
        e1 || e2
    })
}

/// A random position inside the cells.
fn inside(mask: &Image<bool>, rng: &mut Rng) -> (f32, f32) {
    loop {
        let (x, y) = (rng.below(mask.width()), rng.below(mask.height()));
        if mask[(x, y)] {
            return (x as f32 + rng.uniform() - 0.5, y as f32 + rng.uniform() - 0.5);
        }
    }
}

/// Blurred structures: `(x, y, radius, brightness)`, each drawn as a Gaussian whose width
/// combines its radius and the blur of the optics.
fn render(size: usize, spots: &[(f32, f32, f32, f32)], blur: f32) -> Image<f32> {
    let mut img = Image::new(size, size, 0.0f32);
    for &(cx, cy, radius, bright) in spots {
        let s2 = blur * blur + 0.36 * radius * radius;
        let reach = (4.0 * s2.sqrt()).ceil() as isize;
        for dy in -reach..=reach {
            for dx in -reach..=reach {
                let (x, y) = (cx.round() as isize + dx, cy.round() as isize + dy);
                if img.contains(x, y) {
                    let d2 = (x as f32 - cx).powi(2) + (y as f32 - cy).powi(2);
                    img[(x as usize, y as usize)] += bright * (-d2 / (2.0 * s2)).exp();
                }
            }
        }
    }
    img
}

/// Channel A, channel B and the cell mask of a scene. Same seed, same images.
pub fn scene(p: &Scene) -> (Image<f32>, Image<f32>, Image<bool>) {
    let mut rng = Rng::new(p.seed);
    let mask = cells(p.size);
    let mut a_spots = Vec::new();
    let mut b_spots = Vec::new();
    if p.a_inside_b {
        // a few large B structures; every A spot sits inside one of them
        for _ in 0..p.spots / 4 {
            let (x, y) = inside(&mask, &mut rng);
            b_spots.push((x, y, 8.0, 0.2));
        }
        for _ in 0..p.spots {
            let (x, y, _, _) = b_spots[rng.below(b_spots.len())];
            a_spots.push((x + 2.0 * rng.normal(), y + 2.0 * rng.normal(), 1.3 + 1.2 * rng.uniform(), 0.3 + 0.3 * rng.uniform()));
        }
    } else {
        for _ in 0..p.spots {
            let (x, y) = inside(&mask, &mut rng);
            a_spots.push((x, y, 1.3 + 1.2 * rng.uniform(), 0.3 + 0.3 * rng.uniform()));
        }
        for &(x, y, r, _) in a_spots.iter() {
            let (bx, by) = if rng.uniform() < p.shared { (x, y) } else { inside(&mask, &mut rng) };
            b_spots.push((bx, by, r, 0.3 + 0.3 * rng.uniform()));
        }
    }
    let b_spots: Vec<_> = b_spots.iter().map(|&(x, y, r, v)| (x + p.shift.0, y + p.shift.1, r, v)).collect();
    let a_clean = render(p.size, &a_spots, p.blur);
    let b_clean = render(p.size, &b_spots, p.blur);
    let cell = |x: usize, y: usize| if mask[(x, y)] { p.diffuse } else { 0.0 };
    let a = Image::from_fn(p.size, p.size, |x, y| a_clean[(x, y)] + cell(x, y) + p.offset + p.noise * rng.normal());
    let b = Image::from_fn(p.size, p.size, |x, y| {
        b_clean[(x, y)] + p.leak * a_clean[(x, y)] + cell(x, y) + p.offset + p.noise * rng.normal()
    });
    (a, b, mask)
}

/// A mask that is true everywhere (the whole image as the region).
pub fn everywhere(width: usize, height: usize) -> Image<bool> {
    Image::new(width, height, true)
}

/// The `q`-quantile (0..1) of the pixel values inside the mask.
pub fn percentile(img: &Image<f32>, mask: &Image<bool>, q: f32) -> f32 {
    let mut v: Vec<f32> = img.coords().filter(|&p| mask[p]).map(|p| img[p]).collect();
    v.sort_by(f32::total_cmp);
    v[((v.len() - 1) as f32 * q).round() as usize]
}

/// The merge: A green, B magenta, white where both are bright. Each channel is scaled so that
/// its 99.5th percentile is full brightness.
pub fn merge(a: &Image<f32>, b: &Image<f32>) -> Image<[f32; 3]> {
    let all = everywhere(a.width(), a.height());
    let (ma, mb) = (percentile(a, &all, 0.995), percentile(b, &all, 0.995));
    a.zip_map(b, |va, vb| [vb / mb, va / ma, vb / mb])
}

/// The scatter plot (2D histogram) of the pixels in the mask: A to the right, B upwards,
/// `bins × bins` pixels, logarithmic grey. Thresholds, if given, are drawn as a green
/// (t_A, vertical) and a red (t_B, horizontal) line.
pub fn scatter(a: &Image<f32>, b: &Image<f32>, mask: &Image<bool>, bins: usize, thresholds: Option<(f32, f32)>) -> Image<[f32; 3]> {
    let all = everywhere(a.width(), a.height());
    let top = percentile(a, &all, 0.999).max(percentile(b, &all, 0.999)) * 1.05;
    let to_bin = |v: f32| ((v / top * bins as f32).floor().max(0.0) as usize).min(bins - 1);
    let mut h = Image::new(bins, bins, 0.0f32);
    for p in a.coords().filter(|&p| mask[p]) {
        h[(to_bin(a[p]), bins - 1 - to_bin(b[p]))] += 1.0;
    }
    let m = h.max().ln_1p();
    let mut pic = h.map(|c| {
        let g = 1.0 - c.ln_1p() / m;
        [g, g, g]
    });
    if let Some((ta, tb)) = thresholds {
        let (bx, by) = (to_bin(ta), bins - 1 - to_bin(tb));
        for i in 0..bins {
            pic[(bx, i)] = [0.1, 0.7, 0.1];
            pic[(i, by)] = [0.9, 0.1, 0.1];
        }
    }
    pic
}
