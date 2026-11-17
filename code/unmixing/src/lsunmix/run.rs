// A synthetic lambda stack (16 channels, 4 components, photon noise): unmix it one channel per
// dye and by least squares, look at the residual with and without the autofluorescence, and
// measure how similar spectra amplify noise. Pictures in out/lsunmix/.
//   cargo run -p unmixing -- lsunmix                      (your code)
//   cargo run -p unmixing --features solution -- lsunmix  (the model solution)

use imgcourse::*;
use crate::lsunmix::*;

const NAMES: [&str; 4] = ["nucleus", "mitochondria", "membrane", "autofluorescence"];

/// An emission spectrum with a red tail (a skewed bump), sampled at the channel centres and
/// scaled to sum 1 over the channels.
fn spectrum(peak: f32, left: f32, right: f32, centres: &[f32]) -> Vec<f32> {
    let s: Vec<f32> = centres.iter().map(|&l| (-0.5 * ((l - peak) / if l < peak { left } else { right }).powi(2)).exp()).collect();
    let total: f32 = s.iter().sum();
    s.iter().map(|v| v / total).collect()
}

/// The mixing matrix (channels × 4) for `n` channels between 480 and 700 nm; the mitochondria
/// dye peaks `gap` nm after the nucleus dye.
fn mixing(n: usize, gap: f32) -> Matrix {
    let centres: Vec<f32> = (0..n).map(|c| 480.0 + 220.0 * (c as f32 + 0.5) / n as f32).collect();
    let cols = [spectrum(515.0, 12.0, 25.0, &centres), spectrum(515.0 + gap, 12.0, 25.0, &centres), spectrum(625.0, 14.0, 28.0, &centres), spectrum(565.0, 60.0, 80.0, &centres)];
    (0..n).map(|c| (0..4).map(|k| cols[k][c]).collect()).collect()
}

/// The true abundance maps of the cell (128 × 128).
fn cell() -> Vec<Image<f32>> {
    let size = 128;
    let c = 64.0;
    let r = |x: usize, y: usize, cx: f32, cy: f32| ((x as f32 - cx).powi(2) + (y as f32 - cy).powi(2)).sqrt();
    let nucleus = Image::from_fn(size, size, |x, y| if r(x, y, c + 4.0, c - 3.0) < 20.0 { 1.0 } else { 0.0 });
    let membrane = Image::from_fn(size, size, |x, y| if (52.0..56.0).contains(&r(x, y, c, c)) { 1.0 } else { 0.0 });
    let mut rng = Rng::new(4);
    let mito_centres: Vec<(f32, f32, f32)> = (0..14)
        .map(|_| {
            let (a, d) = (std::f32::consts::TAU * rng.uniform(), 28.0 + 18.0 * rng.uniform());
            (c + d * a.cos(), c + d * a.sin(), std::f32::consts::PI * rng.uniform())
        })
        .collect();
    let mito = Image::from_fn(size, size, |x, y| {
        let hit = mito_centres.iter().any(|&(mx, my, t)| {
            let (dx, dy) = (x as f32 - mx, y as f32 - my);
            let (u, v) = (dx * t.cos() + dy * t.sin(), -dx * t.sin() + dy * t.cos());
            (u / 6.0).powi(2) + (v / 2.0).powi(2) < 1.0
        });
        if hit && nucleus[(x, y)] == 0.0 { 0.8 } else { 0.0 }
    });
    let auto = Image::from_fn(size, size, |x, y| if r(x, y, c, c) < 52.0 && nucleus[(x, y)] == 0.0 { 0.35 } else { 0.0 });
    vec![nucleus, mito, membrane, auto]
}

/// Record the lambda stack: photons × M × abundances, with photon noise.
fn record(m: &Matrix, truth: &[Image<f32>], photons: f32, seed: u64) -> Vec<Image<f32>> {
    let mut rng = Rng::new(seed);
    let (w, h) = (truth[0].width(), truth[0].height());
    (0..m.len())
        .map(|c| {
            Image::from_fn(w, h, |x, y| {
                let mean: f32 = (0..4).map(|k| photons * m[c][k] * truth[k][(x, y)]).sum();
                rng.poisson(mean)
            })
        })
        .collect()
}

fn rmse(a: &Image<f32>, b: &Image<f32>) -> f32 {
    let n = (a.width() * a.height()) as f32;
    (a.zip_map(b, |u, v| (u - v).powi(2)).sum() / n).sqrt()
}

/// Four maps as one colour picture: nucleus blue, mitochondria green, membrane magenta,
/// autofluorescence orange.
fn merge(maps: &[Image<f32>]) -> Image<[f32; 3]> {
    let colours = [[0.2, 0.4, 1.0], [0.1, 1.0, 0.2], [1.0, 0.1, 1.0], [1.0, 0.6, 0.0]];
    maps[0].map_xy(|x, y, _| {
        let mut c = [0.0f32; 3];
        for (k, map) in maps.iter().enumerate() {
            for i in 0..3 {
                c[i] += colours[k][i] * map[(x, y)].max(0.0);
            }
        }
        c.map(|v| v.min(1.0))
    })
}

pub fn main() {
    let dir = "out/lsunmix";
    let photons = 400.0;
    let truth = cell();
    let m = mixing(16, 55.0);
    let stack = record(&m, &truth, photons, 1);
    // scale so that abundance 1 gives 1: unmix with photons × M
    let mp: Matrix = m.iter().map(|row| row.iter().map(|v| v * photons).collect()).collect();
    save_rgb(&merge(&truth), format!("{dir}/truth.png"));

    // One channel per dye: each component read in its brightest channel, as if alone.
    let one: Vec<Image<f32>> = (0..4)
        .map(|k| {
            let c = (0..16).max_by(|&a, &b| mp[a][k].total_cmp(&mp[b][k])).unwrap();
            stack[c].map(|v| v / mp[c][k])
        })
        .collect();
    let ls = unmix_image(&stack, &mp);
    save_rgb(&merge(&one), format!("{dir}/one-channel.png"));
    save_rgb(&merge(&ls), format!("{dir}/least-squares.png"));
    for k in 0..4 {
        println!("{:16} RMSE: one channel {:.3}, least squares {:.3}", NAMES[k], rmse(&one[k], &truth[k]), rmse(&ls[k], &truth[k]));
    }
    let negative = ls.iter().flat_map(|a| a.pixels().iter()).filter(|&&v| v < 0.0).count();
    println!("negative abundances: {:.1} % of all values", 100.0 * negative as f32 / (4 * 128 * 128) as f32);

    // The residual, with all four spectra and without the autofluorescence.
    let cyto = truth[3].map(|v| v > 0.0);
    let nuc = truth[0].map(|v| v > 0.0);
    let mean_over = |img: &Image<f32>, mask: &Image<bool>| {
        let v: Vec<f32> = img.coords().filter(|&p| mask[p]).map(|p| img[p]).collect();
        v.iter().sum::<f32>() / v.len() as f32
    };
    let res4 = residual_map(&stack, &mp, &ls);
    let m3: Matrix = mp.iter().map(|row| row[..3].to_vec()).collect();
    let ls3 = unmix_image(&stack, &m3);
    let res3 = residual_map(&stack, &m3, &ls3);
    let both = Image::from_fn(2 * 128 + 4, 128, |x, y| if x < 128 { res3[(x, y)] } else if x < 132 { 0.0 } else { res4[(x - 132, y)] });
    save_gray(&both.map(|v| v / 2.0), format!("{dir}/residual-3-vs-4.png"));
    println!("residual (× noise), 3 spectra: cytoplasm {:.2}, nucleus {:.2}", mean_over(&res3, &cyto), mean_over(&res3, &nuc));
    println!("residual (× noise), 4 spectra: cytoplasm {:.2}, nucleus {:.2}", mean_over(&res4, &cyto), mean_over(&res4, &nuc));
    println!("  mitochondria RMSE with 3 spectra {:.3}", rmse(&ls3[1], &truth[1]));

    // Similar spectra: the mitochondria dye 55 nm and 6 nm from the nucleus dye.
    for gap in [55.0, 6.0] {
        let m2 = mixing(16, gap);
        let s2 = record(&m2, &truth, photons, 2);
        let mp2: Matrix = m2.iter().map(|row| row.iter().map(|v| v * photons).collect()).collect();
        let a = unmix_image(&s2, &mp2);
        let pair: Matrix = m2.iter().map(|row| row[..2].to_vec()).collect();
        println!(
            "gap {gap:2} nm: κ of all four spectra {:5.1}, of nucleus + mitochondria alone {:5.1}; RMSE nucleus {:.3}, mitochondria {:.3}",
            condition_number(&m2),
            condition_number(&pair),
            rmse(&a[0], &truth[0]),
            rmse(&a[1], &truth[1])
        );
        if gap == 6.0 {
            save_rgb(&merge(&a), format!("{dir}/similar.png"));
        }
    }

    // Few photons: ordinary against weighted least squares (weights 1 / expected count, the
    // expected counts from a first ordinary fit).
    let few = 20.0;
    let s20 = record(&m, &truth, few, 3);
    let m20: Matrix = m.iter().map(|row| row.iter().map(|v| v * few).collect()).collect();
    let ols = unmix_image(&s20, &m20);
    let mut wls: Vec<Image<f32>> = (0..4).map(|_| Image::new(128, 128, 0.0)).collect();
    for (x, y) in truth[0].coords() {
        let pixel: Vec<f32> = s20.iter().map(|ch| ch[(x, y)]).collect();
        let a0: Vec<f32> = ols.iter().map(|o| o[(x, y)]).collect();
        let r = residual(&m20, &pixel, &a0);
        let w: Vec<f32> = (0..16).map(|c| 1.0 / (pixel[c] - r[c]).max(0.5)).collect();
        let a = unmix_weighted(&m20, &pixel, &w);
        for k in 0..4 {
            wls[k][(x, y)] = a[k];
        }
    }
    for k in 0..4 {
        println!("{:16} at {few} photons: RMSE ordinary {:.3}, weighted {:.3}", NAMES[k], rmse(&ols[k], &truth[k]), rmse(&wls[k], &truth[k]));
    }
    println!("wrote the pictures to {dir}/");
}
