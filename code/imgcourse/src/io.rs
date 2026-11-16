//! Reading and writing image files (PNG, JPEG, TIFF). Grey values are `f32` in `[0, 1]`:
//! an 8-bit file is divided by 255, a 16-bit file by 65535. These functions stop the
//! program with a message if the file cannot be read or written.

use crate::Image;
use std::path::Path;

/// Read an image file as grey values in `[0, 1]` (colour files are converted to grey).
pub fn load_gray(path: impl AsRef<Path>) -> Image<f32> {
    let path = path.as_ref();
    let img = image::open(path).unwrap_or_else(|e| panic!("could not read {}: {e}", path.display()));
    let g = img.into_luma16();
    let (w, h) = g.dimensions();
    Image::from_vec(w as usize, h as usize, g.into_raw().into_iter().map(|v| v as f32 / 65535.0).collect())
}

/// Read an image file as colour, `[red, green, blue]` in `[0, 1]`.
pub fn load_rgb(path: impl AsRef<Path>) -> Image<[f32; 3]> {
    let path = path.as_ref();
    let img = image::open(path).unwrap_or_else(|e| panic!("could not read {}: {e}", path.display()));
    let c = img.into_rgb32f();
    let (w, h) = c.dimensions();
    let raw = c.into_raw();
    let data = (0..raw.len() / 3).map(|i| [raw[3 * i], raw[3 * i + 1], raw[3 * i + 2]]).collect();
    Image::from_vec(w as usize, h as usize, data)
}

/// Grey values in `[0, 1]` from bytes in memory (used for the built-in sample images).
pub(crate) fn decode_gray(bytes: &[u8]) -> Image<f32> {
    let g = image::load_from_memory(bytes).expect("built-in image").into_luma16();
    let (w, h) = g.dimensions();
    Image::from_vec(w as usize, h as usize, g.into_raw().into_iter().map(|v| v as f32 / 65535.0).collect())
}

fn to_byte(v: f32) -> u8 {
    if v.is_nan() {
        0
    } else {
        (v.clamp(0.0, 1.0) * 255.0).round() as u8
    }
}

fn write_gray8(img: &Image<f32>, path: &Path, f: impl Fn(f32) -> f32) {
    if let Some(dir) = path.parent() {
        if !dir.as_os_str().is_empty() {
            std::fs::create_dir_all(dir).unwrap_or_else(|e| panic!("could not create {}: {e}", dir.display()));
        }
    }
    let bytes: Vec<u8> = img.pixels().iter().map(|&v| to_byte(f(v))).collect();
    image::GrayImage::from_raw(img.width() as u32, img.height() as u32, bytes)
        .expect("size")
        .save(path)
        .unwrap_or_else(|e| panic!("could not write {}: {e}", path.display()));
}

/// Save grey values as an 8-bit file: 0 is black, 1 is white, values outside are clipped.
/// The file type follows the extension (`.png` is a good choice).
pub fn save_gray(img: &Image<f32>, path: impl AsRef<Path>) {
    write_gray8(img, path.as_ref(), |v| v);
}

/// Save with the smallest value as black and the largest as white (for results whose
/// range is not `[0, 1]`, such as a gradient magnitude or a log spectrum).
pub fn save_gray_stretched(img: &Image<f32>, path: impl AsRef<Path>) {
    let (lo, hi) = (img.min(), img.max());
    let range = if hi > lo { hi - lo } else { 1.0 };
    write_gray8(img, path.as_ref(), |v| (v - lo) / range);
}

/// Save values that can be negative (edge filters): 0 is mid-grey, the largest size
/// `|v|` is white (positive) or black (negative).
pub fn save_gray_signed(img: &Image<f32>, path: impl AsRef<Path>) {
    let m = img.pixels().iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let m = if m > 0.0 { m } else { 1.0 };
    write_gray8(img, path.as_ref(), |v| 0.5 + 0.5 * v / m);
}

/// Save a colour image, `[red, green, blue]` in `[0, 1]`.
pub fn save_rgb(img: &Image<[f32; 3]>, path: impl AsRef<Path>) {
    let path = path.as_ref();
    if let Some(dir) = path.parent() {
        if !dir.as_os_str().is_empty() {
            std::fs::create_dir_all(dir).unwrap_or_else(|e| panic!("could not create {}: {e}", dir.display()));
        }
    }
    let bytes: Vec<u8> = img.pixels().iter().flat_map(|p| p.map(to_byte)).collect();
    image::RgbImage::from_raw(img.width() as u32, img.height() as u32, bytes)
        .expect("size")
        .save(path)
        .unwrap_or_else(|e| panic!("could not write {}: {e}", path.display()));
}
