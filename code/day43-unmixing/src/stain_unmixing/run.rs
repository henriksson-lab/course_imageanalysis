// A synthetic H&E slide made by Beer–Lambert with its own stain vectors: separate it with the
// published vectors, with the true ones and with Macenko's estimate, and with a wrong white
// level. Pictures in out/stain-unmixing/.
//   cargo run -p day43 -- stain-unmixing                      (your code)
//   cargo run -p day43 --features solution -- stain-unmixing  (the model solution)

use imgcourse::*;
use crate::stain_unmixing::*;

const SIZE: usize = 256;

/// True hematoxylin and eosin amounts of the slide, and the mask of its two empty holes.
fn slide_amounts() -> (Image<f32>, Image<f32>, Image<bool>) {
    let mut rng = Rng::new(5);
    let nuclei: Vec<(f32, f32, f32, f32)> = (0..45).map(|_| (SIZE as f32 * rng.uniform(), SIZE as f32 * rng.uniform(), 4.0 + 5.0 * rng.uniform(), 0.5 + 0.5 * rng.uniform())).collect();
    let s = SIZE as f32;
    let hole = Image::from_fn(SIZE, SIZE, |x, y| {
        let (xf, yf) = (x as f32, y as f32);
        ((xf - 0.25 * s).powi(2) + (yf - 0.7 * s).powi(2)).sqrt() < 0.1 * s || ((xf - 0.75 * s).powi(2) + (yf - 0.3 * s).powi(2)).sqrt() < 0.07 * s
    });
    let h = Image::from_fn(SIZE, SIZE, |x, y| {
        let (xf, yf) = (x as f32, y as f32);
        if hole[(x, y)] {
            return 0.0;
        }
        nuclei.iter().map(|&(cx, cy, r, a)| if (xf - cx).powi(2) + (yf - cy).powi(2) < r * r { a } else { 0.0 }).fold(0.0, f32::max)
    });
    let e = Image::from_fn(SIZE, SIZE, |x, y| {
        if hole[(x, y)] {
            0.0
        } else {
            let base = 0.35 + 0.2 * (x as f32 / 23.0).sin() * (y as f32 / 31.0).cos();
            if h[(x, y)] > 0.0 { 0.3 * base } else { base }
        }
    });
    (h, e, hole)
}

/// Stack grey images side by side with a white gap.
fn side_by_side(images: &[Image<f32>]) -> Image<f32> {
    let w = images[0].width();
    Image::from_fn(images.len() * (w + 4) - 4, images[0].height(), |x, y| {
        let (i, xx) = (x / (w + 4), x % (w + 4));
        if xx < w { images[i][(xx, y)] } else { 1.0 }
    })
}

fn angle(u: [f32; 3], v: [f32; 3]) -> f32 {
    let d = u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
    let n = (u.iter().map(|a| a * a).sum::<f32>() * v.iter().map(|a| a * a).sum::<f32>()).sqrt();
    (d / n).clamp(-1.0, 1.0).acos().to_degrees()
}

pub fn main() {
    let dir = "out/stain-unmixing";
    // This slide's own stains, a little different from the published ones.
    let true_s = stain_matrix([0.61, 0.73, 0.31], [0.13, 0.94, 0.34]);
    let col = |s: &Mat3, k: usize| [s[0][k], s[1][k], s[2][k]];
    println!("true stains: H {:.1}° and E {:.1}° from the published vectors", angle(col(&true_s, 0), HEMATOXYLIN), angle(col(&true_s, 1), EOSIN));

    // Render by Beer–Lambert, white level 242 of 255, detector noise, 8-bit rounding.
    let (h, e, hole) = slide_amounts();
    let i0 = [242.0 / 255.0; 3];
    let mut rng = Rng::new(9);
    let slide: Rgb = Image::from_fn(SIZE, SIZE, |x, y| {
        transmit([h[(x, y)], e[(x, y)], 0.0], &true_s, i0).map(|v| ((v + 0.004 * rng.normal()).clamp(0.0, 1.0) * 255.0).round() / 255.0)
    });
    save_rgb(&slide, format!("{dir}/slide.png"));
    let od = optical_density(&slide, i0);

    // Crosstalk: what a matrix does to each true stain (deconvolve the true stain vectors).
    let crosstalk = |s: &Mat3| {
        let probe = Image::from_rows(&[[col(&true_s, 0), col(&true_s, 1)]]);
        let c = deconvolve(&probe, s);
        [[c[0][(0, 0)], c[0][(1, 0)]], [c[1][(0, 0)], c[1][(1, 0)]]]
    };
    let show = |c: &Image<f32>| c.map(|v| 1.0 - v / 1.2); // stain dark on white
    let published = stain_matrix(HEMATOXYLIN, EOSIN);
    let od_pixels: Vec<[f32; 3]> = od.pixels().to_vec();
    let [mh, me] = macenko(&od_pixels, 0.15, 1.0);
    println!("Macenko: H {:.1}° and E {:.1}° from the true stains", angle(mh, col(&true_s, 0)), angle(me, col(&true_s, 1)));
    let estimated = stain_matrix(mh, me);
    for (name, s) in [("published", &published), ("true", &true_s), ("macenko", &estimated)] {
        let t = crosstalk(s);
        let c = deconvolve(&od, s);
        save_gray(&side_by_side(&[show(&c[0]), show(&c[1])]), format!("{dir}/he-{name}.png"));
        println!(
            "{name:9}: H map = {:.2} H {:+.2} E, E map = {:+.2} H {:+.2} E; residual map mean {:.3}",
            t[0][0], t[0][1], t[1][0], t[1][1], c[2].mean()
        );
    }

    // A white level set too low (205 instead of 242).
    let od_low = optical_density(&slide, [205.0 / 255.0; 3]);
    let c = deconvolve(&od_low, &true_s);
    let in_holes: Vec<f32> = hole.coords().filter(|&p| hole[p]).map(|p| c[0][p]).collect();
    println!("I0 = 205 instead of 242: background OD {:.3}, H map in the empty holes {:.3}", od_low[(64, 179)][0], in_holes.iter().sum::<f32>() / in_holes.len() as f32);
    println!("wrote the pictures to {dir}/");
}
