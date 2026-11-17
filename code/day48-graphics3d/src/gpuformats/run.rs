// Stores the course micrograph in different byte layouts and formats, reads it back with your
// code (rightly and wrongly), and writes the results to out/gpuformats/.
//   cargo run -p day48 -- gpuformats                      (your code)
//   cargo run -p day48 --features solution -- gpuformats  (the model solution)

use crate::gpuformats::*;
use imgcourse::*;

/// A simple colour map, so that the grey micrograph has three different channels.
fn colourise(v: f32) -> [f32; 3] {
    [v.sqrt(), v, v * v]
}

/// Bilinear interpolation of a 4-channel image at (x, y) in texel units (texel centres at
/// k + 0.5), with the edge copied outside. As in the images unit.
fn bilinear(img: &Image<[f32; 4]>, x: f32, y: f32) -> [f32; 4] {
    let (fx, fy) = (x - 0.5, y - 0.5);
    let (x0, y0) = (fx.floor(), fy.floor());
    let (tx, ty) = (fx - x0, fy - y0);
    let at = |i: f32, j: f32| {
        let i = (i as isize).clamp(0, img.width() as isize - 1) as usize;
        let j = (j as isize).clamp(0, img.height() as isize - 1) as usize;
        img[(i, j)]
    };
    let (a, b, c, d) = (at(x0, y0), at(x0 + 1.0, y0), at(x0, y0 + 1.0), at(x0 + 1.0, y0 + 1.0));
    [0, 1, 2, 3].map(|k| (1.0 - ty) * ((1.0 - tx) * a[k] + tx * b[k]) + ty * ((1.0 - tx) * c[k] + tx * d[k]))
}

pub fn main() {
    let dir = "out/gpuformats";
    let cell = samples::cell();

    // 1. The stride: 253 pixels wide, RGB, 759 bytes per row padded to a stride of 760.
    let (w, h, stride) = (253, 256, 760);
    let mut bytes = vec![0u8; stride * h];
    for y in 0..h {
        for x in 0..w {
            let rgb = colourise(cell[(x, y)]);
            for c in 0..3 {
                bytes[y * stride + 3 * x + c] = (rgb[c] * 255.0).round() as u8;
            }
        }
    }
    save_rgb(&read_rgb8(&bytes, w, h, 760), format!("{dir}/stride-760.png"));
    save_rgb(&read_rgb8(&bytes, w, h, 759), format!("{dir}/stride-759.png"));
    println!("stride: {} bytes of RGB data per row, stored with a stride of {stride}", 3 * w);

    // 2. Endianness: the micrograph as 16-bit values 0–4000, stored little-endian.
    let mut bytes16 = Vec::new();
    for &v in cell.pixels() {
        let k = (v * 4000.0).round() as u16;
        bytes16.extend_from_slice(&k.to_le_bytes());
    }
    for (name, le) in [("little", true), ("big", false)] {
        let img = Image::from_fn(256, 256, |x, y| read_u16(&bytes16, y * 256 + x, le) as f32);
        println!("read as {name}-endian: values {} to {}", img.min(), img.max());
        save_gray_stretched(&img, format!("{dir}/endian-{name}.png"));
    }

    // 3. A grey ramp from 0.30 to 0.45, stored in 8 bits per channel (top) and in 5-6-5
    //    (bottom), decoded and contrast-stretched so that each stored step shows.
    let (rw, rh) = (480, 60);
    let stretch = |v: f32| ((v - 0.30) / 0.15).clamp(0.0, 1.0);
    let ramp = Image::from_fn(rw, rh, |x, y| {
        let v = 0.30 + 0.15 * x as f32 / (rw - 1) as f32;
        let stored = if y < rh / 2 {
            let k = code(v, 8) as f32 / 255.0;
            [k, k, k]
        } else {
            unpack565(pack565([v, v, v]))
        };
        stored.map(stretch)
    });
    save_rgb(&ramp, format!("{dir}/ramp-8bit-565.png"));

    // 4. Premultiplied alpha: a red disc on a 12 × 12 texture whose transparent texels hold
    //    green, magnified 20 times with bilinear filtering and drawn over white.
    let tex = Image::from_fn(12, 12, |x, y| {
        let (dx, dy) = (x as f32 + 0.5 - 6.0, y as f32 + 0.5 - 6.0);
        if dx * dx + dy * dy < 16.0 {
            [1.0, 0.0, 0.0, 1.0]
        } else {
            [0.0, 1.0, 0.0, 0.0]
        }
    });
    let tex_pre = tex.map(premultiply);
    let white = [1.0, 1.0, 1.0, 1.0];
    for (name, premultiplied) in [("straight", false), ("premultiplied", true)] {
        let out = Image::from_fn(240, 240, |x, y| {
            let (u, v) = ((x as f32 + 0.5) / 20.0, (y as f32 + 0.5) / 20.0);
            // straight: filter the straight values, premultiply only for compositing
            let c = if premultiplied { bilinear(&tex_pre, u, v) } else { premultiply(bilinear(&tex, u, v)) };
            let o = over(c, white);
            [o[0], o[1], o[2]]
        });
        save_rgb(&out, format!("{dir}/fringe-{name}.png"));
    }

    // 5. Mipmaps of the micrograph, side by side, and their memory.
    let levels = mip_chain(&cell);
    let mut sheet = Image::new(256 + 128 + 64 + 32 + 16 + 8 + 4 + 2 + 1 + 8, 256, 0.3f32);
    let mut x0 = 0;
    for l in &levels {
        for (x, y) in l.coords() {
            sheet[(x0 + x, y)] = l[(x, y)];
        }
        x0 += l.width() + 1;
    }
    save_gray(&sheet, format!("{dir}/mipmaps.png"));
    let total: usize = levels.iter().map(|l| l.width() * l.height()).sum();
    println!("mipmaps: {} levels, {total} texels in all, {:.4} times the full image", levels.len(), total as f64 / (256.0 * 256.0));
    println!("wrote the pictures to {dir}/");
}
