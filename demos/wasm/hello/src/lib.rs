//! Smallest useful example of the demo Wasm convention: plain numeric functions over
//! flat arrays, no DOM. The TypeScript side (demos/test/wasm-test.ts) checks each one
//! against a reference implementation.
use wasm_bindgen::prelude::*;

/// a + b (the "is anything working at all" check).
#[wasm_bindgen]
pub fn add(a: i32, b: i32) -> i32 {
    a.wrapping_add(b)
}

/// Sum of a float array, accumulated in f64.
#[wasm_bindgen]
pub fn sum_f32(values: &[f32]) -> f64 {
    values.iter().map(|&v| v as f64).sum()
}

/// Invert 8-bit pixels in place (255 - v): shows a `&mut [u8]` round trip.
#[wasm_bindgen]
pub fn invert_u8(pixels: &mut [u8]) {
    for p in pixels.iter_mut() {
        *p = 255 - *p;
    }
}

/// 3x3 box blur of a w x h grey image with replicated borders; returns a new array.
#[wasm_bindgen]
pub fn box3(img: &[f32], w: usize, h: usize) -> Vec<f32> {
    assert_eq!(img.len(), w * h, "box3: image length must be w*h");
    let mut out = vec![0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut s = 0f32;
            for dy in -1i64..=1 {
                for dx in -1i64..=1 {
                    let yy = (y as i64 + dy).clamp(0, h as i64 - 1) as usize;
                    let xx = (x as i64 + dx).clamp(0, w as i64 - 1) as usize;
                    s += img[yy * w + xx];
                }
            }
            out[y * w + x] = s / 9.0;
        }
    }
    out
}

/// A string from Rust, to check the glue's text decoding.
#[wasm_bindgen]
pub fn greet(name: &str) -> String {
    format!("Hello, {name}, from Rust")
}
