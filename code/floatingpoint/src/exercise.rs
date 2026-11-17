// Coding exercises for "Floating point". Replace each `todo!()` with your
// code, then check it with `cargo test -p floatingpoint` and look at `cargo run -p floatingpoint`.
#![allow(unused_variables, unused_imports)]

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
///
/// Hint: `bits >> 15` is the sign bit, `(bits >> 10) & 0x1F` the exponent field, `bits & 0x3FF`
/// the fraction; `2f32.powi(n)` is 2^n.
pub fn decode_f16(bits: u16) -> f32 {
    todo!("C1: decode_f16")
}

/// C2. Round `x` to the nearest float16 value (ties to the even neighbour), returned as an f32.
/// In [2^k, 2^(k+1)) the float16 numbers are 2^(k − 10) apart; below 2^(−14) (subnormals) the
/// gap stays 2^(−24). Anything that rounds above the largest float16, 65 504, becomes ±infinity.
///
/// Hint: find k with 2^k ≤ |x| < 2^(k+1) (`a.log2().floor()`, then check it, since `log2`
/// can be off by a hair); the gap is 2^(max(k, −14) − 10); `(a / gap).round_ties_even() * gap`
/// rounds; `r.copysign(x)` puts the sign back.
pub fn round_to_f16(x: f32) -> f32 {
    todo!("C2: round_to_f16")
}

/// C3a. The naive running sum, rounding to the number format with `fl` after every addition
/// (`fl` = the identity for float32, whose additions round by themselves; `round_to_f16` for
/// float16).
pub fn sum_naive(xs: &[f32], fl: Round) -> f32 {
    todo!("C3a: sum_naive")
}

/// C3b. Pairwise summation: split the list in two halves, sum each half the same way, add the
/// two results (rounded with `fl`). A single number is its own sum.
///
/// Hint: `xs.split_at(xs.len() / 2)` gives the two halves; the function calls itself on each.
pub fn sum_pairwise(xs: &[f32], fl: Round) -> f32 {
    todo!("C3b: sum_pairwise")
}

/// C3c. Kahan (compensated) summation: for each term, y = x − c, t = s + y,
/// c = (t − s) − y, s = t, with every operation rounded by `fl`.
///
/// Hint: follow the four lines of B9 literally, with `fl(...)` around every operation.
pub fn sum_kahan(xs: &[f32], fl: Round) -> f32 {
    todo!("C3c: sum_kahan")
}

/// C4a. The variance (divide by n) in one pass, in f32: the mean of the squares minus the
/// square of the mean. Prone to cancellation.
pub fn variance_one_pass(xs: &[f32]) -> f32 {
    todo!("C4a: variance_one_pass")
}

/// C4b. The variance in two passes, in f32: first the mean, then the mean of the squared
/// deviations from it. Each pass uses the given sum (`sum_naive` or `sum_pairwise`, in f32).
pub fn variance_two_pass(xs: &[f32], sum: Sum) -> f32 {
    todo!("C4b: variance_two_pass")
}
