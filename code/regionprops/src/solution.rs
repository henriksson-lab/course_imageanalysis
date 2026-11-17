// Model solution for the coding exercises of "Region properties and moments".

use imgcourse::Image;

/// One row of the measurement table: the sums over the pixels of one object.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Region {
    /// number of pixels, m00
    pub area: usize,
    /// centroid (x̄, ȳ) = (m10 / m00, m01 / m00)
    pub cx: f32,
    pub cy: f32,
    /// central moments μ20 = Σ (x − x̄)², μ11 = Σ (x − x̄)(y − ȳ), μ02 = Σ (y − ȳ)²
    pub mu20: f32,
    pub mu11: f32,
    pub mu02: f32,
    /// bounding box, inclusive
    pub x_min: usize,
    pub x_max: usize,
    pub y_min: usize,
    pub y_max: usize,
}

impl Region {
    /// Area / area of the bounding box.
    pub fn extent(&self) -> f32 {
        self.area as f32 / ((self.x_max - self.x_min + 1) * (self.y_max - self.y_min + 1)) as f32
    }
}

/// The equivalent ellipse: the eigenvalues λ1 ≥ λ2 of S = (1/A) [[μ20, μ11], [μ11, μ02]], and
/// the angle `theta` of the eigenvector v1 = (cos θ, sin θ), in radians.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ellipse {
    pub lambda1: f32,
    pub lambda2: f32,
    pub theta: f32,
}

impl Ellipse {
    /// √(1 − λ2/λ1): 0 for a disc, near 1 for a rod.
    pub fn eccentricity(&self) -> f32 {
        (1.0 - self.lambda2 / self.lambda1).sqrt()
    }

    /// The semi-axes 2√λ1 and 2√λ2.
    pub fn semi_axes(&self) -> (f32, f32) {
        (2.0 * self.lambda1.sqrt(), 2.0 * self.lambda2.sqrt())
    }
}

/// C1. One scan over the label image: area, centroid, central moments and bounding box of
/// every object. Element `k - 1` of the result is label `k`, for the labels `1..=max`.
pub fn measure(labels: &Image<u32>) -> Vec<Region> {
    let n = labels.pixels().iter().copied().max().unwrap_or(0) as usize;
    // raw moments m00, m10, m01, m20, m11, m02 (in f64: sums of many squares)
    let mut m = vec![[0.0f64; 6]; n];
    let mut regions = vec![Region { x_min: usize::MAX, y_min: usize::MAX, ..Default::default() }; n];
    for (x, y) in labels.coords() {
        let l = labels[(x, y)] as usize;
        if l == 0 {
            continue;
        }
        let (xf, yf) = (x as f64, y as f64);
        let s = &mut m[l - 1];
        s[0] += 1.0;
        s[1] += xf;
        s[2] += yf;
        s[3] += xf * xf;
        s[4] += xf * yf;
        s[5] += yf * yf;
        let r = &mut regions[l - 1];
        r.x_min = r.x_min.min(x);
        r.x_max = r.x_max.max(x);
        r.y_min = r.y_min.min(y);
        r.y_max = r.y_max.max(y);
    }
    for (r, s) in regions.iter_mut().zip(&m) {
        let [m00, m10, m01, m20, m11, m02] = *s;
        r.area = m00 as usize;
        r.cx = (m10 / m00) as f32;
        r.cy = (m01 / m00) as f32;
        // central moments from the raw moments (slide "The math: moments")
        r.mu20 = (m20 - m10 * m10 / m00) as f32;
        r.mu11 = (m11 - m10 * m01 / m00) as f32;
        r.mu02 = (m02 - m01 * m01 / m00) as f32;
    }
    regions
}

/// C2. The equivalent ellipse of a region: the eigenvalues of S, largest first, and the angle
/// θ = ½ atan2(2 μ11, μ20 − μ02) of the major axis, measured from the x axis towards the
/// y axis (y points down, so positive θ turns clockwise on screen).
pub fn ellipse(r: &Region) -> Ellipse {
    let a = r.mu20 / r.area as f32;
    let b = r.mu11 / r.area as f32;
    let c = r.mu02 / r.area as f32;
    let mean = (a + c) / 2.0;
    let root = (((a - c) / 2.0).powi(2) + b * b).sqrt();
    Ellipse { lambda1: mean + root, lambda2: mean - root, theta: 0.5 * (2.0 * b).atan2(a - c) }
}

/// Is `(x, y)` an object pixel? Positions outside the image are background. (A helper for C3–C5.)
pub fn at(mask: &Image<bool>, x: isize, y: isize) -> bool {
    mask.contains(x, y) && mask[(x as usize, y as usize)]
}

/// C3. Perimeter by edge count: the number of pixel edges between an object pixel and a
/// background pixel (outside the image counts as background).
pub fn perimeter_edges(mask: &Image<bool>) -> usize {
    let mut edges = 0;
    for (x, y) in mask.coords() {
        if !mask[(x, y)] {
            continue;
        }
        for (dx, dy) in [(0, -1), (-1, 0), (1, 0), (0, 1)] {
            if !at(mask, x as isize + dx, y as isize + dy) {
                edges += 1;
            }
        }
    }
    edges
}

/// C4. Perimeter by the Crofton formula: count the crossings of the boundary along rows (n0),
/// columns (n90) and both diagonals (n45, n135), i.e. the pairs of neighbouring pixels in that
/// direction where one is object and the other background; then
/// P ≈ π/8 · (n0 + n90 + (n45 + n135)/√2).
pub fn perimeter_crofton(mask: &Image<bool>) -> f32 {
    let (w, h) = (mask.width() as isize, mask.height() as isize);
    let mut n = [0usize; 4];
    let steps = [(1, 0), (0, 1), (1, -1), (1, 1)];
    // every pair (p, p + step) with at least one pixel inside the image, padded with background
    for y in -1..=h {
        for x in -1..=w {
            for (k, (dx, dy)) in steps.iter().enumerate() {
                if at(mask, x, y) != at(mask, x + dx, y + dy) {
                    n[k] += 1;
                }
            }
        }
    }
    let pi = std::f32::consts::PI;
    pi / 8.0 * ((n[0] + n[1]) as f32 + (n[2] + n[3]) as f32 / 2f32.sqrt())
}

/// C5a. The bit-quad counts (Q1, Q3, QD): the 2 × 2 windows with exactly one object pixel,
/// with exactly three, and with two object pixels on a diagonal. All windows that overlap the
/// image are counted, with background outside.
pub fn bit_quads(mask: &Image<bool>) -> (i32, i32, i32) {
    let (w, h) = (mask.width() as isize, mask.height() as isize);
    let (mut q1, mut q3, mut qd) = (0, 0, 0);
    // window with top-left pixel (x, y)
    for y in -1..h {
        for x in -1..w {
            let a = at(mask, x, y);
            let b = at(mask, x + 1, y);
            let c = at(mask, x, y + 1);
            let d = at(mask, x + 1, y + 1);
            match a as i32 + b as i32 + c as i32 + d as i32 {
                1 => q1 += 1,
                3 => q3 += 1,
                2 if a == d => qd += 1, // a and d both set, or b and c both set
                _ => {}
            }
        }
    }
    (q1, q3, qd)
}

/// C5b. The Euler number (objects − holes) from the bit quads: (Q1 − Q3 − 2 QD)/4 for
/// 8-connected objects, (Q1 − Q3 + 2 QD)/4 for 4-connected objects.
pub fn euler_number(mask: &Image<bool>, eight: bool) -> i32 {
    let (q1, q3, qd) = bit_quads(mask);
    if eight {
        (q1 - q3 - 2 * qd) / 4
    } else {
        (q1 - q3 + 2 * qd) / 4
    }
}
