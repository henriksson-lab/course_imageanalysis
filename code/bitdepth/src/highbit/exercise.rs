// Coding exercises for "12- and 16-bit images in practice". Replace each `todo!()` with your
// code, then check it with `cargo test -p bitdepth highbit::` and look at `cargo run -p bitdepth -- highbit`.
#![allow(unused_variables, unused_imports)]

/// C1a. Subtraction in uint16 with wrap-around: the result modulo 2^16 (what plain integer
/// arithmetic does). Computed in a wider type, without the built-in `wrapping_sub`.
///
/// Hint: `a as i32 - b as i32` cannot overflow; `x.rem_euclid(65536)` is x modulo 2^16, also
/// for negative x.
pub fn sub_wrap(a: u16, b: u16) -> u16 {
    todo!("C1a: sub_wrap")
}

/// C1b. Addition in uint16 with wrap-around.
pub fn add_wrap(a: u16, b: u16) -> u16 {
    todo!("C1b: add_wrap")
}

/// C1c. Subtraction in uint16 with saturation: results below 0 become 0 (and above 65 535
/// would become 65 535).
///
/// Hint: `x.clamp(0, 65535)`.
pub fn sub_saturate(a: u16, b: u16) -> u16 {
    todo!("C1c: sub_saturate")
}

/// C2a. Pack two 12-bit pixels into 3 bytes: the top 8 bits of p1; the low 4 bits of p1
/// followed by the top 4 bits of p2; the low 8 bits of p2.
///
/// Hint: `p >> 4` drops the low 4 bits, `p & 0xF` keeps only them, `<<` shifts left and `|`
/// combines bits.
pub fn pack12(p1: u16, p2: u16) -> [u8; 3] {
    todo!("C2a: pack12")
}

/// C2b. Unpack 3 bytes into the two 12-bit pixels again.
pub fn unpack12(b: [u8; 3]) -> (u16, u16) {
    todo!("C2b: unpack12")
}

/// C3. The display level (0 … 255) of the raw value `v` with the window [lo, hi]:
/// round(255 · clip((v − lo)/(hi − lo), 0, 1)).
///
/// Hint: `x.clamp(0.0, 1.0)`, then `.round() as u8`.
pub fn window(v: f32, lo: f32, hi: f32) -> u8 {
    todo!("C3: window")
}

/// C4. The `p`-th percentile (p between 0 and 1) of some values: sort them, and take the one
/// at position round(p · (n − 1)). For auto-contrast windows such as 0.5 % … 99.5 %.
pub fn percentile(values: &[u16], p: f32) -> u16 {
    todo!("C4: percentile")
}
