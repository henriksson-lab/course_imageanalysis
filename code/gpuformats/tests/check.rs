// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p gpuformats

use gpuformats::*;
use imgcourse::Image;

fn close(a: f32, b: f32) -> bool {
    (a - b).abs() < 1e-5
}

#[test]
fn c1_offset_b1() {
    // 1001 × 600 RGB, rows padded to 3004 bytes; green of (10, 2)
    let o = byte_offset(10, 2, 1, 3004, 3, 1);
    assert_eq!(o, 6039, "green of (10, 2): got offset {o}, expected 6039");
    // 16-bit, 2 channels: (3, 1), channel 1 with stride 100
    let o = byte_offset(3, 1, 1, 100, 2, 2);
    assert_eq!(o, 114, "2 channels of 2 bytes, (3, 1) channel 1, stride 100: got {o}, expected 114");
}

#[test]
fn c1_read_with_padding() {
    // a 2 × 2 RGB image, rows padded from 6 to 8 bytes (the padding bytes hold 99)
    let data = [10, 20, 30, 40, 50, 60, 99, 99, 70, 80, 90, 100, 110, 120, 99, 99];
    let img = read_rgb8(&data, 2, 2, 8);
    let want = [[10u8, 20, 30], [40, 50, 60], [70, 80, 90], [100, 110, 120]];
    for (k, (x, y)) in [(0, 0), (1, 0), (0, 1), (1, 1)].into_iter().enumerate() {
        for c in 0..3 {
            let e = want[k][c] as f32 / 255.0;
            assert!(close(img[(x, y)][c], e), "pixel ({x}, {y}) channel {c}: got {}, expected {}/255", img[(x, y)][c], want[k][c]);
        }
    }
}

#[test]
fn c2_endianness_b2() {
    let data = [0xE8u8, 0x03]; // 1000 stored little-endian
    let le = read_u16(&data, 0, true);
    let be = read_u16(&data, 0, false);
    assert_eq!(le, 1000, "little-endian: got {le}, expected 1000");
    assert_eq!(be, 59395, "big-endian: got {be}, expected 59395 (0xE803)");
    let data = [0u8, 0, 0x12, 0x34];
    assert_eq!(read_u16(&data, 1, true), 0x3412, "value number 1 is in bytes 2 and 3");
}

#[test]
fn c3_packed_b3() {
    let orange = [1.0, 0.5, 0.0];
    let w = pack565(orange);
    assert_eq!(w, 0xFC00, "5-6-5 of (1, 0.5, 0): got {w:#06X}, expected 0xFC00");
    let back = unpack565(w);
    assert!(close(back[1], 32.0 / 63.0), "green decodes to {}, expected 32/63 = 0.508", back[1]);
    assert!(close(back[0], 1.0) && close(back[2], 0.0), "red and blue decode to {} and {}, expected 1 and 0", back[0], back[2]);
    let w = pack1010102(orange, 3);
    assert_eq!(w, 0xC00803FF, "10-10-10-2 with alpha 3: got {w:#010X}, expected 0xC00803FF");
}

#[test]
fn c4_premultiplied_b4() {
    let red = [1.0, 0.0, 0.0, 1.0];
    let clear_green = [0.0, 1.0, 0.0, 0.0];
    let white = [1.0, 1.0, 1.0, 1.0];
    let avg = |a: [f32; 4], b: [f32; 4]| [0, 1, 2, 3].map(|k| 0.5 * (a[k] + b[k]));
    // straight alpha: average first, then composite
    let straight = over(premultiply(avg(red, clear_green)), white);
    let want = [0.75, 0.75, 0.5];
    assert!((0..3).all(|k| close(straight[k], want[k])), "straight alpha over white: got {straight:?}, expected (0.75, 0.75, 0.5)");
    // premultiplied: premultiply first, then average and composite
    let pre = over(avg(premultiply(red), premultiply(clear_green)), white);
    let want = [1.0, 0.5, 0.5];
    assert!((0..3).all(|k| close(pre[k], want[k])), "premultiplied over white: got {pre:?}, expected (1, 0.5, 0.5)");
    assert!(close(pre[3], 1.0), "alpha over an opaque background: got {}, expected 1", pre[3]);
}

#[test]
fn c5_mipmaps_b7() {
    let img = Image::from_fn(4, 4, |x, y| (x + 4 * y) as f32);
    let levels = mip_chain(&img);
    let sizes: Vec<(usize, usize)> = levels.iter().map(|l| (l.width(), l.height())).collect();
    assert_eq!(sizes, [(4, 4), (2, 2), (1, 1)], "level sizes");
    let total: usize = levels.iter().map(|l| l.width() * l.height()).sum();
    assert_eq!(total, 21, "texels in all levels: got {total}, expected 21 = (64 − 1)/3");
    let l1 = &levels[1];
    assert!(close(l1[(0, 0)], 2.5) && close(l1[(1, 1)], 12.5), "level 1: got {:?}, expected 2.5, 4.5, 10.5, 12.5", l1.pixels());
    assert!(close(levels[2][(0, 0)], img.mean()), "the 1 × 1 level is the mean: got {}, expected {}", levels[2][(0, 0)], img.mean());
}
