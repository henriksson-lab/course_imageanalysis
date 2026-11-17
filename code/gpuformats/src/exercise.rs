// Coding exercises for "How graphics cards store colour".
// Replace each `todo!()` with your code, then check it with `cargo test -p gpuformats`
// and look at `cargo run -p gpuformats`.
// Bytes are `u8`, colours `[f32; 3]` or `[r, g, b, alpha]` with values in [0, 1].
#![allow(unused_variables, unused_imports)]

use imgcourse::Image;

/// C1a. The byte offset of channel `c` of pixel `(x, y)` in an interleaved image with
/// `channels` channels of `bytes` bytes each and a row stride of `stride` bytes
/// (slide "The math: where is a pixel?").
pub fn byte_offset(x: usize, y: usize, c: usize, stride: usize, channels: usize, bytes: usize) -> usize {
    todo!("C1a: byte_offset")
}

/// C1b. Read an interleaved RGB image, 8 bits per channel, with the given row stride.
/// Every byte `k` becomes the value `k / 255`.
///
/// Hint: loop over `img.coords()` and the three channels, and read
/// `data[byte_offset(x, y, c, stride, 3, 1)]`.
pub fn read_rgb8(data: &[u8], width: usize, height: usize, stride: usize) -> Image<[f32; 3]> {
    todo!("C1b: read_rgb8")
}

/// C2. The 16-bit value number `i` in `data` (bytes `2i` and `2i + 1`), in little-endian (low
/// byte first) or big-endian (high byte first) order.
///
/// Hint: turn the two bytes into `u16` first; `hi << 8 | lo` puts `hi` in the top byte.
pub fn read_u16(data: &[u8], i: usize, little_endian: bool) -> u16 {
    todo!("C2: read_u16")
}

/// The code of a value in [0, 1] with `bits` bits: `round(v · (2^bits − 1))`.
pub fn code(v: f32, bits: u32) -> u32 {
    let max = ((1u32 << bits) - 1) as f32;
    (v.clamp(0.0, 1.0) * max).round() as u32
}

/// C3a. Pack a colour into 5-6-5: red in the top 5 bits, green in the middle 6, blue in the
/// low 5 bits.
///
/// Hint: `code(v, bits)` above gives the integer code of a value; shift red by 11 and green by 5,
/// and combine with `|`.
pub fn pack565(rgb: [f32; 3]) -> u16 {
    todo!("C3a: pack565")
}

/// C3b. Unpack a 5-6-5 word into a colour in [0, 1].
///
/// Hint: `(word >> 5) & 63` extracts the 6 green bits.
pub fn unpack565(word: u16) -> [f32; 3] {
    todo!("C3b: unpack565")
}

/// C3c. Pack a colour and a 2-bit alpha code (0–3) into 10-10-10-2: red in the lowest 10
/// bits, then green, then blue, alpha in the top 2 bits.
///
/// Hint: work in `u32`: green is shifted by 10, blue by 20, alpha by 30.
pub fn pack1010102(rgb: [f32; 3], alpha: u32) -> u32 {
    todo!("C3c: pack1010102")
}

/// C4a. Straight alpha to premultiplied: `(αR, αG, αB, α)`.
pub fn premultiply(c: [f32; 4]) -> [f32; 4] {
    todo!("C4a: premultiply")
}

/// C4b. The "over" operator on premultiplied colours: `src + (1 − α_src) · dst`, in all four
/// channels (slide "The math: compositing and filtering with alpha").
pub fn over(src: [f32; 4], dst: [f32; 4]) -> [f32; 4] {
    todo!("C4b: over")
}

/// C5a. The next mipmap level: every output pixel is the mean of a 2 × 2 block (the width
/// and height are halved, rounded down, but at least 1).
///
/// Hint: `Image::from_fn` with the mean of the pixels (2x + dx, 2y + dy), dx, dy in 0..2, that lie
/// inside the image.
pub fn half_size(img: &Image<f32>) -> Image<f32> {
    todo!("C5a: half_size")
}

/// C5b. The whole mipmap chain: the image itself, then `half_size` again and again down to
/// 1 × 1.
///
/// Hint: keep calling `half_size` on the last level until it is 1 × 1.
pub fn mip_chain(img: &Image<f32>) -> Vec<Image<f32>> {
    todo!("C5b: mip_chain")
}
