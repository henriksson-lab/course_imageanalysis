// Averages in codes and in linear light, builds the sRGB matrix, splits a synthetic scene into
// Y′CbCr and white-balances it. Pictures in out/rgb-linear/.
//   cargo run -p rgb-linear                      (your code)
//   cargo run -p rgb-linear --features solution  (the model solution)

use imgcourse::*;
use rgb_linear::*;

/// A synthetic scene in **linear** light, `size × size`: a brown table with a soft shadow, a
/// red apple, a yellow fruit, a green leaf, a blue cup and a grey card (18 % reflectance).
/// Returns the image and the mask of the grey card.
fn scene(size: usize) -> (Rgb, Image<bool>) {
    let s = size as f32;
    let mut card = Image::new(size, size, false);
    let img = Image::from_fn(size, size, |x, y| {
        let (u, v) = (x as f32 / s, y as f32 / s);
        let disc = |cx: f32, cy: f32, r: f32| ((u - cx).powi(2) + (v - cy).powi(2)).sqrt() < r;
        // light from the upper left; a soft shadow on the lower right of the table
        let shade = 1.0 - 0.5 * (((u - 0.55) + (v - 0.55)) * 4.0).clamp(0.0, 1.0);
        let (colour, round) = if disc(0.3, 0.3, 0.16) {
            ([0.50, 0.035, 0.03], Some((0.3, 0.3, 0.16)))
        } else if disc(0.72, 0.28, 0.12) {
            ([0.62, 0.42, 0.03], Some((0.72, 0.28, 0.12)))
        } else if ((u - 0.28) / 0.18).powi(2) + ((v - 0.75) / 0.08).powi(2) < 1.0 {
            ([0.06, 0.25, 0.04], None)
        } else if (0.58..0.85).contains(&u) && (0.55..0.88).contains(&v) {
            ([0.03, 0.07, 0.40], None)
        } else if (0.08..0.22).contains(&u) && (0.08..0.16).contains(&v) {
            ([0.18, 0.18, 0.18], None)
        } else {
            ([0.25, 0.14, 0.07], None)
        };
        if (0.08..0.22).contains(&u) && (0.08..0.16).contains(&v) {
            card[(x, y)] = true;
        }
        // round objects get brighter towards the upper left
        let light = match round {
            Some((cx, cy, r)) => 0.45 + 0.75 * (1.0 - ((u - cx + 0.4 * r).powi(2) + (v - cy + 0.4 * r).powi(2)).sqrt() / (1.6 * r)).max(0.0),
            None => 1.0,
        };
        colour.map(|c| (c * light * if round.is_some() { 1.0 } else { shade }).min(1.0))
    });
    (img, card)
}

/// Enlarge by `k` (nearest neighbour), to see the pixels of small images.
fn enlarge(img: &Rgb, k: usize) -> Rgb {
    Image::from_fn(img.width() * k, img.height() * k, |x, y| img[(x / k, y / k)])
}

fn encode_img(img: &Rgb) -> Rgb {
    img.map(|p| p.map(|v| srgb_encode(v.clamp(0.0, 1.0))))
}

fn main() {
    let dir = "out/rgb-linear";

    // 1. Black and white stripes, one pixel wide, averaged 4 × 4.
    let stripes: Rgb = Image::from_fn(64, 32, |x, _| [(x % 2) as f32; 3]);
    let (a, b) = (downsample(&stripes, 4, false), downsample(&stripes, 4, true));
    println!("stripes averaged on codes: code {:.1}; in linear light: code {:.1}", 255.0 * a[(0, 0)][0], 255.0 * b[(0, 0)][0]);

    // 2. Red and green squares of 6 pixels, averaged in 4 × 4 blocks: some blocks mix the two.
    let checker: Rgb = Image::from_fn(96, 96, |x, y| if (x / 6 + y / 6) % 2 == 0 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] });
    save_rgb(&checker, format!("{dir}/checker.png"));
    let codes = downsample(&checker, 4, false);
    let light = downsample(&checker, 4, true);
    save_rgb(&enlarge(&codes, 4), format!("{dir}/checker-codes.png"));
    save_rgb(&enlarge(&light, 4), format!("{dir}/checker-linear.png"));
    let p = (1, 0); // block x = 4..7: columns 4, 5 red, 6, 7 green
    println!("a half red, half green block: codes {:.0?} on codes, {:.0?} in linear light", codes[p].map(|v| 255.0 * v), light[p].map(|v| 255.0 * v));

    // 3. The sRGB matrix from its primaries and the D65 white.
    let m = rgb_to_xyz_matrix([[0.64, 0.33], [0.30, 0.60], [0.15, 0.06]], [0.3127, 0.3290]);
    println!("sRGB → XYZ:");
    for row in m {
        println!("  {:.4?}", row);
    }

    // 4. Y′CbCr of the scene (codes 0..255), saved as three grey images.
    let (lin, card) = scene(256);
    let enc = encode_img(&lin);
    save_rgb(&enc, format!("{dir}/scene.png"));
    let ycc = enc.map(|p| ycbcr(p.map(|v| 255.0 * v)));
    for (i, name) in ["y", "cb", "cr"].iter().enumerate() {
        save_gray(&ycc.map(|p| p[i] / 255.0), format!("{dir}/scene-{name}.png"));
    }

    // 5. White balance: the scene under a warm lamp, corrected by grey world and by the card.
    let lamp = [1.0, 0.70, 0.40];
    let captured = lin.map(|p| [p[0] * lamp[0], p[1] * lamp[1], p[2] * lamp[2]]);
    save_rgb(&encode_img(&captured), format!("{dir}/lamp.png"));
    let (gains, corrected) = grey_world(&captured);
    save_rgb(&encode_img(&corrected), format!("{dir}/greyworld.png"));
    let mut card_mean = [0.0f32; 3];
    let mut n = 0.0;
    for (x, y) in card.coords() {
        if card[(x, y)] {
            for c in 0..3 {
                card_mean[c] += captured[(x, y)][c];
            }
            n += 1.0;
        }
    }
    let card_gains = [card_mean[1] / card_mean[0], 1.0, card_mean[1] / card_mean[2]];
    let by_card = captured.map(|p| [p[0] * card_gains[0], p[1] * card_gains[1], p[2] * card_gains[2]]);
    save_rgb(&encode_img(&by_card), format!("{dir}/greycard.png"));
    println!("white balance gains (R, G, B): ideal {:.3?}", [lamp[1] / lamp[0], 1.0, lamp[1] / lamp[2]]);
    println!("  grey world {:.3?}   grey card {:.3?} ({} card pixels)", gains, card_gains, n);
    println!("wrote the pictures to {dir}/");
}
