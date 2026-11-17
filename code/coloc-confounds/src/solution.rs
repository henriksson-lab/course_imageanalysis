// Model solution for the coding exercises of "What the microscope does to colocalisation".

use imgcourse::{Border, Image};

/// C1a. The bleed-through factor from a single-stained control (slide "Bleed-through"): the
/// slope through the origin of B against A over the mask, Σ A·B / Σ A².
pub fn leak_slope(a: &Image<f32>, b: &Image<f32>, mask: &Image<bool>) -> f32 {
    let (mut sab, mut saa) = (0.0f64, 0.0f64);
    for p in a.coords().filter(|&p| mask[p]) {
        sab += (a[p] * b[p]) as f64;
        saa += (a[p] * a[p]) as f64;
    }
    (sab / saa) as f32
}

/// C1b. Compensation: B − s·A at every pixel.
pub fn compensate(b: &Image<f32>, a: &Image<f32>, s: f32) -> Image<f32> {
    b.zip_map(a, |vb, va| vb - s * va)
}

/// C2. The normalised overlap of two Gaussian spots of widths `s1`, `s2` whose centres are
/// `delta` apart (slide "The math: Gaussian spot pairs"):
/// 2 s1 s2 / (s1² + s2²) · exp(−delta² / (2 (s1² + s2²))).
pub fn gaussian_overlap(s1: f32, s2: f32, delta: f32) -> f32 {
    let v = s1 * s1 + s2 * s2;
    2.0 * s1 * s2 / v * (-delta * delta / (2.0 * v)).exp()
}

/// C3. Background subtraction with a white top-hat (slide "Background: offset and shared
/// background"): the opening of the image (the minimum over the (2r+1) × (2r+1) square
/// around every pixel, then the maximum over the same square of that result) is the
/// background; return image − opening. Outside the image, read the nearest edge pixel.
pub fn subtract_background(img: &Image<f32>, radius: usize) -> Image<f32> {
    let r = radius as isize;
    let mut eroded = Image::new(img.width(), img.height(), 0.0f32);
    for (x, y) in img.coords() {
        let mut m = f32::INFINITY;
        for dy in -r..=r {
            for dx in -r..=r {
                m = m.min(img.get_border(x as isize + dx, y as isize + dy, Border::Clamp));
            }
        }
        eroded[(x, y)] = m;
    }
    let mut out = Image::new(img.width(), img.height(), 0.0f32);
    for (x, y) in img.coords() {
        let mut m = f32::NEG_INFINITY;
        for dy in -r..=r {
            for dx in -r..=r {
                m = m.max(eroded.get_border(x as isize + dx, y as isize + dy, Border::Clamp));
            }
        }
        out[(x, y)] = img[(x, y)] - m; // image minus opening
    }
    out
}

/// C4a. The shift of B against A, measured on bead images (slide "Chromatic shift"): the shift
/// (tx, ty), each from −max_shift to max_shift, that maximises the cross-correlation
/// c(tx, ty) = Σ (A(x, y) − mean A) (B(x + tx, y + ty) − mean B), with zero outside the image;
/// then refined in x and in y by a parabola through the peak and its two neighbours (as in
/// the shift curve of the last unit). If B is A moved right by 1.5 pixels, the answer is
/// close to (1.5, 0).
pub fn estimate_shift(a: &Image<f32>, b: &Image<f32>, max_shift: usize) -> (f32, f32) {
    let (ma, mb) = (a.mean(), b.mean());
    let (a0, b0) = (a.map(|v| v - ma), b.map(|v| v - mb));
    let corr = |tx: isize, ty: isize| {
        let mut c = 0.0;
        for (x, y) in a0.coords() {
            c += a0[(x, y)] * b0.get_border(x as isize + tx, y as isize + ty, Border::Zero);
        }
        c
    };
    let m = max_shift as isize;
    let (mut best, mut bx, mut by) = (f32::NEG_INFINITY, 0, 0);
    for ty in -m..=m {
        for tx in -m..=m {
            let c = corr(tx, ty);
            if c > best {
                (best, bx, by) = (c, tx, ty);
            }
        }
    }
    let vertex = |cm: f32, c0: f32, cp: f32| (cm - cp) / (2.0 * (cm - 2.0 * c0 + cp));
    let fx = vertex(corr(bx - 1, by), best, corr(bx + 1, by));
    let fy = vertex(corr(bx, by - 1), best, corr(bx, by + 1));
    (bx as f32 + fx, by as f32 + fy)
}

/// C4b. Shift an image back: the result at (x, y) is `img` at (x + dx, y + dy), read between
/// pixels by bilinear interpolation (the nearest edge pixel outside). With the shift from
/// `estimate_shift`, `shift_image(b, tx, ty)` lines B up with A.
pub fn shift_image(img: &Image<f32>, dx: f32, dy: f32) -> Image<f32> {
    img.map_xy(|x, y, _| {
        let (sx, sy) = (x as f32 + dx, y as f32 + dy);
        let (x0, y0) = (sx.floor(), sy.floor());
        let (fx, fy) = (sx - x0, sy - y0);
        let (x0, y0) = (x0 as isize, y0 as isize);
        let v = |xx: isize, yy: isize| img.get_border(xx, yy, Border::Clamp);
        (1.0 - fx) * (1.0 - fy) * v(x0, y0) + fx * (1.0 - fy) * v(x0 + 1, y0) + (1.0 - fx) * fy * v(x0, y0 + 1) + fx * fy * v(x0 + 1, y0 + 1)
    })
}
