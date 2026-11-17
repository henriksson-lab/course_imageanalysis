// Model solution for the coding exercises of "How graphics cards store colour".
// Bytes are `u8`, colours `[f32; 3]` or `[r, g, b, alpha]` with values in [0, 1].

use imgcourse::Image;

/// C1a. The byte offset of channel `c` of pixel `(x, y)` in an interleaved image with
/// `channels` channels of `bytes` bytes each and a row stride of `stride` bytes
/// (slide "The math: where is a pixel?").
pub fn byte_offset(x: usize, y: usize, c: usize, stride: usize, channels: usize, bytes: usize) -> usize {
    y * stride + x * channels * bytes + c * bytes
}

/// C1b. Read an interleaved RGB image, 8 bits per channel, with the given row stride.
/// Every byte `k` becomes the value `k / 255`.
pub fn read_rgb8(data: &[u8], width: usize, height: usize, stride: usize) -> Image<[f32; 3]> {
    let mut img = Image::new(width, height, [0.0f32; 3]);
    for (x, y) in img.coords() {
        for c in 0..3 {
            img[(x, y)][c] = data[byte_offset(x, y, c, stride, 3, 1)] as f32 / 255.0;
        }
    }
    img
}

/// C2. The 16-bit value number `i` in `data` (bytes `2i` and `2i + 1`), in little-endian (low
/// byte first) or big-endian (high byte first) order.
pub fn read_u16(data: &[u8], i: usize, little_endian: bool) -> u16 {
    let (first, second) = (data[2 * i] as u16, data[2 * i + 1] as u16);
    if little_endian {
        first | (second << 8)
    } else {
        (first << 8) | second
    }
}

/// The code of a value in [0, 1] with `bits` bits: `round(v · (2^bits − 1))`.
pub fn code(v: f32, bits: u32) -> u32 {
    let max = ((1u32 << bits) - 1) as f32;
    (v.clamp(0.0, 1.0) * max).round() as u32
}

/// C3a. Pack a colour into 5-6-5: red in the top 5 bits, green in the middle 6, blue in the
/// low 5 bits.
pub fn pack565(rgb: [f32; 3]) -> u16 {
    let (r, g, b) = (code(rgb[0], 5), code(rgb[1], 6), code(rgb[2], 5));
    ((r << 11) | (g << 5) | b) as u16
}

/// C3b. Unpack a 5-6-5 word into a colour in [0, 1].
pub fn unpack565(word: u16) -> [f32; 3] {
    let w = word as u32;
    let (r, g, b) = ((w >> 11) & 31, (w >> 5) & 63, w & 31);
    [r as f32 / 31.0, g as f32 / 63.0, b as f32 / 31.0]
}

/// C3c. Pack a colour and a 2-bit alpha code (0–3) into 10-10-10-2: red in the lowest 10
/// bits, then green, then blue, alpha in the top 2 bits.
pub fn pack1010102(rgb: [f32; 3], alpha: u32) -> u32 {
    code(rgb[0], 10) | (code(rgb[1], 10) << 10) | (code(rgb[2], 10) << 20) | (alpha << 30)
}

/// C4a. Straight alpha to premultiplied: `(αR, αG, αB, α)`.
pub fn premultiply(c: [f32; 4]) -> [f32; 4] {
    [c[3] * c[0], c[3] * c[1], c[3] * c[2], c[3]]
}

/// C4b. The "over" operator on premultiplied colours: `src + (1 − α_src) · dst`, in all four
/// channels (slide "The math: compositing and filtering with alpha").
pub fn over(src: [f32; 4], dst: [f32; 4]) -> [f32; 4] {
    let mut out = [0.0; 4];
    for k in 0..4 {
        out[k] = src[k] + (1.0 - src[3]) * dst[k];
    }
    out
}

/// C5a. The next mipmap level: every output pixel is the mean of a 2 × 2 block (the width
/// and height are halved, rounded down, but at least 1).
pub fn half_size(img: &Image<f32>) -> Image<f32> {
    let (w, h) = ((img.width() / 2).max(1), (img.height() / 2).max(1));
    Image::from_fn(w, h, |x, y| {
        let mut sum = 0.0;
        let mut n = 0.0;
        for dy in 0..2 {
            for dx in 0..2 {
                let (sx, sy) = (2 * x + dx, 2 * y + dy);
                if sx < img.width() && sy < img.height() {
                    sum += img[(sx, sy)];
                    n += 1.0;
                }
            }
        }
        sum / n
    })
}

/// C5b. The whole mipmap chain: the image itself, then `half_size` again and again down to
/// 1 × 1.
pub fn mip_chain(img: &Image<f32>) -> Vec<Image<f32>> {
    let mut levels = vec![img.clone()];
    while levels.last().unwrap().width() > 1 || levels.last().unwrap().height() > 1 {
        let next = half_size(levels.last().unwrap());
        levels.push(next);
    }
    levels
}
