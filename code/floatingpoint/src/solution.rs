// Model solution for the coding exercises of "Floating point".
//
// Rust has 32-bit floats (`f32`) but no 16-bit ones, so here we build float16 ourselves:
// decode its bits, and round any number to the nearest float16.

/// A rounding function: the identity for float32 (whose operations round by themselves),
/// or `round_to_f16`.
pub type Round = fn(f32) -> f32;

/// A summation method: `sum_naive`, `sum_pairwise` or `sum_kahan`.
pub type Sum = fn(&[f32], Round) -> f32;

/// C1. The value of a float16 bit pattern: 1 sign bit, 5 exponent bits (bias 15), 10 fraction
/// bits. E = 0: zero or subnormal, 2^(−14) · F/1024; E = 31: infinity (F = 0) or NaN;
/// otherwise 2^(E − 15) · (1 + F/1024).
pub fn decode_f16(bits: u16) -> f32 {
    let sign = if bits >> 15 == 1 { -1.0 } else { 1.0 };
    let e = ((bits >> 10) & 0x1F) as i32;
    let f = (bits & 0x3FF) as f32;
    if e == 0 {
        sign * 2f32.powi(-14) * (f / 1024.0)
    } else if e == 31 {
        if f == 0.0 {
            sign * f32::INFINITY
        } else {
            f32::NAN
        }
    } else {
        sign * 2f32.powi(e - 15) * (1.0 + f / 1024.0)
    }
}

/// C2. Round `x` to the nearest float16 value (ties to the even neighbour), returned as an f32.
/// In [2^k, 2^(k+1)) the float16 numbers are 2^(k − 10) apart; below 2^(−14) (subnormals) the
/// gap stays 2^(−24). Anything that rounds above the largest float16, 65 504, becomes ±infinity.
pub fn round_to_f16(x: f32) -> f32 {
    if x == 0.0 || x.is_nan() || x.is_infinite() {
        return x;
    }
    let a = x.abs();
    // k = ⌊log2 a⌋, corrected in case log2 is off by a hair near a power of two.
    let mut k = a.log2().floor() as i32;
    if 2f32.powi(k) > a {
        k -= 1;
    }
    if 2f32.powi(k + 1) <= a {
        k += 1;
    }
    let gap = 2f32.powi(k.max(-14) - 10);
    let r = (a / gap).round_ties_even() * gap;
    let r = if r > 65504.0 { f32::INFINITY } else { r };
    r.copysign(x)
}

/// C3a. The naive running sum, rounding to the number format with `fl` after every addition
/// (`fl` = the identity for float32, whose additions round by themselves; `round_to_f16` for
/// float16).
pub fn sum_naive(xs: &[f32], fl: Round) -> f32 {
    let mut s = 0.0;
    for &x in xs {
        s = fl(s + x);
    }
    s
}

/// C3b. Pairwise summation: split the list in two halves, sum each half the same way, add the
/// two results (rounded with `fl`). A single number is its own sum.
pub fn sum_pairwise(xs: &[f32], fl: Round) -> f32 {
    match xs.len() {
        0 => 0.0,
        1 => xs[0],
        n => {
            let (left, right) = xs.split_at(n / 2);
            fl(sum_pairwise(left, fl) + sum_pairwise(right, fl))
        }
    }
}

/// C3c. Kahan (compensated) summation: for each term, y = x − c, t = s + y,
/// c = (t − s) − y, s = t, with every operation rounded by `fl`.
pub fn sum_kahan(xs: &[f32], fl: Round) -> f32 {
    let (mut s, mut c) = (0.0, 0.0);
    for &x in xs {
        let y = fl(x - c);
        let t = fl(s + y);
        c = fl(fl(t - s) - y);
        s = t;
    }
    s
}

/// C4a. The variance (divide by n) in one pass, in f32: the mean of the squares minus the
/// square of the mean. Prone to cancellation.
pub fn variance_one_pass(xs: &[f32]) -> f32 {
    let n = xs.len() as f32;
    let (mut s, mut s2) = (0.0f32, 0.0f32);
    for &x in xs {
        s += x;
        s2 += x * x;
    }
    s2 / n - (s / n) * (s / n)
}

/// C4b. The variance in two passes, in f32: first the mean, then the mean of the squared
/// deviations from it. Each pass uses the given sum (`sum_naive` or `sum_pairwise`, in f32).
pub fn variance_two_pass(xs: &[f32], sum: Sum) -> f32 {
    let n = xs.len() as f32;
    let mean = sum(xs, |v| v) / n;
    let squares: Vec<f32> = xs.iter().map(|x| (x - mean) * (x - mean)).collect();
    sum(&squares, |v| v) / n
}
