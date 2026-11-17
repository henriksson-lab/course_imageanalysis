// Model solution for the coding exercises of "Prediction, dictionaries and lossless formats".

use imgcourse::{Border, Image};

/// The predictors of the lecture (slide "Predictors"), from the neighbours a = left, b = up,
/// c = up-left.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Predictor {
    /// No prediction: the residual is the pixel itself.
    None,
    Left,
    Up,
    /// floor((a + b) / 2)
    Average,
    /// a + b − c
    Planar,
    /// Of a, b, c the one closest to a + b − c (ties: a, then b). PNG.
    Paeth,
    /// The median edge detector of JPEG-LS.
    Med,
}

pub const ALL: [Predictor; 7] =
    [Predictor::None, Predictor::Left, Predictor::Up, Predictor::Average, Predictor::Planar, Predictor::Paeth, Predictor::Med];

/// C1. The prediction of a pixel from its neighbours a (left), b (up) and c (up-left).
pub fn predict(a: i32, b: i32, c: i32, predictor: Predictor) -> i32 {
    match predictor {
        Predictor::None => 0,
        Predictor::Left => a,
        Predictor::Up => b,
        Predictor::Average => (a + b).div_euclid(2), // floor, also for negative sums
        Predictor::Planar => a + b - c,
        Predictor::Paeth => {
            let p = a + b - c;
            let (da, db, dc) = ((p - a).abs(), (p - b).abs(), (p - c).abs());
            if da <= db && da <= dc {
                a
            } else if db <= dc {
                b
            } else {
                c
            }
        }
        Predictor::Med => {
            if c >= a.max(b) {
                a.min(b)
            } else if c <= a.min(b) {
                a.max(b)
            } else {
                a + b - c
            }
        }
    }
}

/// C2. The residual image e = x − x̂, scanning row by row. Neighbours outside the image count
/// as 0, so the first pixel is stored as it is (its prediction is 0).
pub fn residuals(img: &Image<i32>, predictor: Predictor) -> Image<i32> {
    let mut out = Image::new(img.width(), img.height(), 0);
    for (x, y) in img.coords() {
        let (x, y) = (x as isize, y as isize);
        let a = img.get_border(x - 1, y, Border::Zero);
        let b = img.get_border(x, y - 1, Border::Zero);
        let c = img.get_border(x - 1, y - 1, Border::Zero);
        out[(x as usize, y as usize)] = img[(x as usize, y as usize)] - predict(a, b, c, predictor);
    }
    out
}

/// C3. Decoding: rebuild the image from its residuals. Scan in the same order as the encoder,
/// predict from the pixels already decoded, and add the residual back.
pub fn decode(res: &Image<i32>, predictor: Predictor) -> Image<i32> {
    let mut out = Image::new(res.width(), res.height(), 0);
    for (x, y) in res.coords() {
        let (xi, yi) = (x as isize, y as isize);
        // `out` holds the decoded pixels so far: exactly what the encoder saw
        let a = out.get_border(xi - 1, yi, Border::Zero);
        let b = out.get_border(xi, yi - 1, Border::Zero);
        let c = out.get_border(xi - 1, yi - 1, Border::Zero);
        out[(x, y)] = predict(a, b, c, predictor) + res[(x, y)];
    }
    out
}

/// C4. The entropy in bits per value of a list of whole numbers (any range, also negative):
/// count how often each value occurs, then H = −Σ p log2 p.
pub fn entropy(values: &[i32]) -> f32 {
    let lo = *values.iter().min().unwrap();
    let hi = *values.iter().max().unwrap();
    let mut counts = vec![0usize; (hi - lo + 1) as usize];
    for &v in values {
        counts[(v - lo) as usize] += 1;
    }
    let n = values.len() as f32;
    let mut h = 0.0;
    for &c in &counts {
        if c > 0 {
            let p = c as f32 / n;
            h -= p * p.log2();
        }
    }
    h
}
