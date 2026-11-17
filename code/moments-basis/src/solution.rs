// Model solution for the coding exercises of "Moments as projections onto monomials".

use imgcourse::Image;

/// A table of moments: `m[p][q]` is the moment of order `p + q` with the weight `x^p y^q`.
/// Only the entries with `p + q <= order` are filled; the others stay 0.
pub type Moments = Vec<Vec<f64>>;

/// C1. The raw moments `m[p][q] = sum over all pixels of x^p y^q I(x, y)` for `p + q <= order`,
/// with `x` the column and `y` the row in pixel units (slide "One weight per moment").
pub fn raw_moments(img: &Image<f32>, order: usize) -> Moments {
    let mut m = vec![vec![0.0; order + 1]; order + 1];
    for (x, y) in img.coords() {
        let v = img[(x, y)] as f64;
        for p in 0..=order {
            for q in 0..=order - p {
                m[p][q] += (x as f64).powi(p as i32) * (y as f64).powi(q as i32) * v;
            }
        }
    }
    m
}

/// C2. The central moments `mu[p][q] = sum of (x - xbar)^p (y - ybar)^q I(x, y)`, with the
/// centroid `xbar = m10 / m00`, `ybar = m01 / m00`: a first pass for the centroid, a second
/// pass for the sums.
pub fn central_moments(img: &Image<f32>, order: usize) -> Moments {
    let m = raw_moments(img, 1);
    let (xbar, ybar) = (m[1][0] / m[0][0], m[0][1] / m[0][0]);
    let mut mu = vec![vec![0.0; order + 1]; order + 1];
    for (x, y) in img.coords() {
        let v = img[(x, y)] as f64;
        let (dx, dy) = (x as f64 - xbar, y as f64 - ybar);
        for p in 0..=order {
            for q in 0..=order - p {
                mu[p][q] += dx.powi(p as i32) * dy.powi(q as i32) * v;
            }
        }
    }
    mu
}

/// C3. The image as a distribution: skewness and kurtosis of the two marginals, from a table
/// of central moments up to order 4. Returns `(skew_x, kurt_x, skew_y, kurt_y)` with
/// `skew_x = mu30 / (m00 sd_x^3)`, `kurt_x = mu40 / (m00 sd_x^4)`, `sd_x^2 = mu20 / m00`.
pub fn skewness_kurtosis(mu: &Moments) -> (f64, f64, f64, f64) {
    let m00 = mu[0][0];
    let sd_x = (mu[2][0] / m00).sqrt();
    let sd_y = (mu[0][2] / m00).sqrt();
    (
        mu[3][0] / (m00 * sd_x.powi(3)),
        mu[4][0] / (m00 * sd_x.powi(4)),
        mu[0][3] / (m00 * sd_y.powi(3)),
        mu[0][4] / (m00 * sd_y.powi(4)),
    )
}

/// C4. All raw moments at once, separably (slide "Computing moments"): first the column sums
/// weighted by `y^q`, `c[x][q] = sum over y of y^q I(x, y)`, then `m[p][q] = sum over x of
/// x^p c[x][q]`. With `scaled`, the pixel centres are mapped to [-1, 1] first:
/// `x -> (2x + 1) / width - 1`, `y -> (2y + 1) / height - 1`.
pub fn moments_separable(img: &Image<f32>, order: usize, scaled: bool) -> Moments {
    let (w, h) = (img.width(), img.height());
    let cx = |x: usize| if scaled { (2 * x + 1) as f64 / w as f64 - 1.0 } else { x as f64 };
    let cy = |y: usize| if scaled { (2 * y + 1) as f64 / h as f64 - 1.0 } else { y as f64 };
    // c[x][q]: one pass over the image
    let mut c = vec![vec![0.0; order + 1]; w];
    for (x, y) in img.coords() {
        let v = img[(x, y)] as f64;
        for q in 0..=order {
            c[x][q] += cy(y).powi(q as i32) * v;
        }
    }
    // then only the columns: width × (order + 1)² operations
    let mut m = vec![vec![0.0; order + 1]; order + 1];
    for x in 0..w {
        for p in 0..=order {
            for q in 0..=order - p {
                m[p][q] += cx(x).powi(p as i32) * c[x][q];
            }
        }
    }
    m
}
