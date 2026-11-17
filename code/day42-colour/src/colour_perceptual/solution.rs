// Model solution for the coding exercises of "HSV, CIELAB and colour segmentation".

use imgcourse::Image;

/// A colour image: `[red, green, blue]` per pixel, values in [0, 1].
pub type Rgb = Image<[f32; 3]>;

/// sRGB decoding (from the last unit): stored value in [0, 1] → linear light.
pub fn srgb_decode(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// C1. Hue (degrees in [0, 360)), saturation and value of a colour with R, G, B in [0, 1]
/// (slide "The math: computing H, S, V"). For a grey (max = min) return hue 0 and
/// saturation 0; for black also saturation 0.
pub fn rgb_to_hsv(c: [f32; 3]) -> [f32; 3] {
    let [r, g, b] = c;
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let d = max - min;
    let s = if max > 0.0 { d / max } else { 0.0 };
    let h = if d == 0.0 {
        0.0 // undefined for greys
    } else if max == r {
        60.0 * ((g - b) / d).rem_euclid(6.0)
    } else if max == g {
        60.0 * ((b - r) / d + 2.0)
    } else {
        60.0 * ((r - g) / d + 4.0)
    };
    [h, s, max]
}

/// C2. The circular mean of hues in degrees: the angle of the mean of the unit vectors
/// (cos H, sin H), in [0, 360), and the length R of that mean vector (1: all hues agree,
/// near 0: spread out). Returns `(mean, r)`.
pub fn circular_mean(hues: &[f32]) -> (f32, f32) {
    let (mut sx, mut sy) = (0.0f32, 0.0f32);
    for h in hues {
        sx += h.to_radians().cos();
        sy += h.to_radians().sin();
    }
    let n = hues.len() as f32;
    let mean = sy.atan2(sx).to_degrees().rem_euclid(360.0);
    (mean, (sx * sx + sy * sy).sqrt() / n)
}

/// C3a. CIELAB (L*, a*, b*) of an encoded sRGB colour (values in [0, 1]), with the D65 white
/// (0.9505, 1, 1.089) (slide "The math: from XYZ to Lab"): decode, multiply by the sRGB
/// matrix to get XYZ, then
/// `L* = 116 f(Y/Yn) − 16`, `a* = 500 (f(X/Xn) − f(Y/Yn))`, `b* = 200 (f(Y/Yn) − f(Z/Zn))`.
pub fn srgb_to_lab(c: [f32; 3]) -> [f32; 3] {
    let lin = c.map(srgb_decode);
    let m = [[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]];
    let white = [0.9505, 1.0, 1.089];
    let f = |t: f32| {
        let d = 6.0f32 / 29.0;
        if t > d * d * d { t.cbrt() } else { t / (3.0 * d * d) + 4.0 / 29.0 }
    };
    let mut fxyz = [0.0f32; 3];
    for i in 0..3 {
        let v = m[i][0] * lin[0] + m[i][1] * lin[1] + m[i][2] * lin[2];
        fxyz[i] = f(v / white[i]);
    }
    [116.0 * fxyz[1] - 16.0, 500.0 * (fxyz[0] - fxyz[1]), 200.0 * (fxyz[1] - fxyz[2])]
}

/// C3b. ΔE*ab: the Euclidean distance between two Lab colours.
pub fn delta_e(lab1: [f32; 3], lab2: [f32; 3]) -> f32 {
    let mut s = 0.0;
    for i in 0..3 {
        s += (lab1[i] - lab2[i]).powi(2);
    }
    s.sqrt()
}

/// C4. A hue wedge: true where the hue lies in the interval from `h_lo` to `h_hi` degrees
/// (going up from `h_lo`; if `h_lo > h_hi` the interval wraps through 0°, as 340°–20°), and
/// the saturation and value are at least `s_min` and `v_min`.
pub fn hue_mask(img: &Rgb, h_lo: f32, h_hi: f32, s_min: f32, v_min: f32) -> Image<bool> {
    img.map(|c| {
        let [h, s, v] = rgb_to_hsv(c);
        let in_hue = if h_lo <= h_hi { h >= h_lo && h <= h_hi } else { h >= h_lo || h <= h_hi };
        in_hue && s >= s_min && v >= v_min
    })
}

/// C5. k-means of 3D points: starting from the given centres, repeat `iterations` times:
/// give every point to its nearest centre (squared Euclidean distance), then move each centre
/// to the mean of its points (a centre without points stays where it is).
/// Returns `(centres, labels)`, with `labels[i]` the centre of point i.
pub fn kmeans(points: &[[f32; 3]], centres: &[[f32; 3]], iterations: usize) -> (Vec<[f32; 3]>, Vec<usize>) {
    let k = centres.len();
    let mut centres = centres.to_vec();
    let mut labels = vec![0; points.len()];
    for _ in 0..iterations {
        // assignment step
        for (i, p) in points.iter().enumerate() {
            let mut best = (f32::INFINITY, 0);
            for (j, c) in centres.iter().enumerate() {
                let d = (p[0] - c[0]).powi(2) + (p[1] - c[1]).powi(2) + (p[2] - c[2]).powi(2);
                if d < best.0 {
                    best = (d, j);
                }
            }
            labels[i] = best.1;
        }
        // update step
        let mut sum = vec![[0.0f32; 3]; k];
        let mut count = vec![0usize; k];
        for (p, &l) in points.iter().zip(&labels) {
            for d in 0..3 {
                sum[l][d] += p[d];
            }
            count[l] += 1;
        }
        for j in 0..k {
            if count[j] > 0 {
                centres[j] = sum[j].map(|s| s / count[j] as f32);
            }
        }
    }
    (centres, labels)
}
