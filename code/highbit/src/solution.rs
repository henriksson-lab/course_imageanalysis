// Model solution for the coding exercises of "12- and 16-bit images in practice".

/// C1a. Subtraction in uint16 with wrap-around: the result modulo 2^16 (what plain integer
/// arithmetic does). Computed in a wider type, without the built-in `wrapping_sub`.
pub fn sub_wrap(a: u16, b: u16) -> u16 {
    (a as i32 - b as i32).rem_euclid(65536) as u16
}

/// C1b. Addition in uint16 with wrap-around.
pub fn add_wrap(a: u16, b: u16) -> u16 {
    ((a as i32 + b as i32) % 65536) as u16
}

/// C1c. Subtraction in uint16 with saturation: results below 0 become 0 (and above 65 535
/// would become 65 535).
pub fn sub_saturate(a: u16, b: u16) -> u16 {
    (a as i32 - b as i32).clamp(0, 65535) as u16
}

/// C2a. Pack two 12-bit pixels into 3 bytes: the top 8 bits of p1; the low 4 bits of p1
/// followed by the top 4 bits of p2; the low 8 bits of p2.
pub fn pack12(p1: u16, p2: u16) -> [u8; 3] {
    [(p1 >> 4) as u8, (((p1 & 0xF) << 4) | (p2 >> 8)) as u8, (p2 & 0xFF) as u8]
}

/// C2b. Unpack 3 bytes into the two 12-bit pixels again.
pub fn unpack12(b: [u8; 3]) -> (u16, u16) {
    let p1 = ((b[0] as u16) << 4) | (b[1] as u16 >> 4);
    let p2 = ((b[1] as u16 & 0xF) << 8) | b[2] as u16;
    (p1, p2)
}

/// C3. The display level (0 … 255) of the raw value `v` with the window [lo, hi]:
/// round(255 · clip((v − lo)/(hi − lo), 0, 1)).
pub fn window(v: f32, lo: f32, hi: f32) -> u8 {
    (255.0 * ((v - lo) / (hi - lo)).clamp(0.0, 1.0)).round() as u8
}

/// C4. The `p`-th percentile (p between 0 and 1) of some values: sort them, and take the one
/// at position round(p · (n − 1)). For auto-contrast windows such as 0.5 % … 99.5 %.
pub fn percentile(values: &[u16], p: f32) -> u16 {
    let mut sorted = values.to_vec();
    sorted.sort();
    let i = (p * (sorted.len() - 1) as f32).round() as usize;
    sorted[i]
}
