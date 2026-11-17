// Stitches a 3 × 3 mosaic of tiles cut from the (enlarged) micrograph at jittered stage
// positions, and estimates a sub-pixel shift with a parabola. Results in out/stitching/.
//   cargo run -p fftreg -- stitching                      (your code)
//   cargo run -p fftreg --features solution -- stitching  (the model solution)

use imgcourse::fft::fft;
use imgcourse::*;
use crate::stitching::*;

const TILE: usize = 128;
const STEP: usize = 104; // nominal distance between tiles: 24 px overlap

/// The micrograph enlarged 1.5 times (bilinear), 384 × 384: a field large enough for the
/// mosaic. (Only 256² pixels of it are real data.)
fn field() -> Image<f32> {
    let cell = samples::cell();
    Image::from_fn(384, 384, |x, y| {
        let (u, v) = ((x as f32 / 1.5).min(254.999), (y as f32 / 1.5).min(254.999));
        let (x0, y0) = (u as usize, v as usize);
        let (fx, fy) = (u - x0 as f32, v - y0 as f32);
        let top = cell[(x0, y0)] * (1.0 - fx) + cell[(x0 + 1, y0)] * fx;
        let bottom = cell[(x0, y0 + 1)] * (1.0 - fx) + cell[(x0 + 1, y0 + 1)] * fx;
        top * (1.0 - fy) + bottom * fy
    })
}

/// The value of `img` at a position between pixels (bilinear interpolation).
fn sample(img: &Image<f32>, u: f32, v: f32) -> f32 {
    let (x0, y0) = (u.floor() as usize, v.floor() as usize);
    let (fx, fy) = (u - x0 as f32, v - y0 as f32);
    let top = img[(x0, y0)] * (1.0 - fx) + img[(x0 + 1, y0)] * fx;
    let bottom = img[(x0, y0 + 1)] * (1.0 - fx) + img[(x0 + 1, y0 + 1)] * fx;
    top * (1.0 - fy) + bottom * fy
}

/// The largest distance between estimated positions and the true ones (both relative to tile 0).
fn max_error(pos: &[(f64, f64)], truth: &[(f32, f32)]) -> f64 {
    pos.iter()
        .zip(truth)
        .map(|(p, t)| (p.0 - (t.0 - truth[0].0) as f64).hypot(p.1 - (t.1 - truth[0].1) as f64))
        .fold(0.0, f64::max)
}

/// Cuts nine tiles from `field` at jittered stage positions (not whole pixels), registers every
/// pair of neighbours, trusts the pairs with NCC >= `trust`, and places the tiles by stage
/// position, by a chain of pairs and by global least squares.
fn mosaic(name: &str, field: &Image<f32>, trust: f32) {
    let dir = "out/stitching";
    let mut rng = Rng::new(5);
    let mut truth = Vec::new();
    let mut tiles = Vec::new();
    for row in 0..3 {
        for col in 0..3 {
            let (x, y) = ((10 + col * STEP) as f32 + 12.0 * rng.uniform() - 6.0, (10 + row * STEP) as f32 + 12.0 * rng.uniform() - 6.0);
            truth.push((x, y));
            let gain = 0.85 + 0.3 * rng.uniform();
            let tile = Image::from_fn(TILE, TILE, |u, v| gain * sample(field, x + u as f32, y + v as f32));
            tiles.push(samples::add_noise(&tile, 0.03, 10 + tiles.len() as u64));
        }
    }

    println!("\n== {name}: trust pairs with NCC >= {trust}\npair   shift (x, y)   NCC  overlap   true shift");
    let mut pairs = Vec::new();
    for i in 0..9 {
        for j in [i + 1, i + 3] {
            if j >= 9 || (j == i + 1 && i % 3 == 2) {
                continue;
            }
            if let Some((tx, ty, c)) = register_pair(&tiles[i], &tiles[j], 1000) {
                let n = ncc_overlap(&tiles[i], &tiles[j], tx, ty).1;
                let t = (truth[j].0 - truth[i].0, truth[j].1 - truth[i].1);
                println!("{i}-{j}   ({tx:4}, {ty:4})   {c:5.2} {n:7}   ({:6.1}, {:6.1})", t.0, t.1);
                if c >= trust {
                    pairs.push(Pair { i, j, dx: tx as f64, dy: ty as f64, w: c as f64 });
                }
            }
        }
    }

    let stage: Vec<(f64, f64)> = (0..9).map(|k| (((k % 3) * STEP) as f64, ((k / 3) * STEP) as f64)).collect();
    let mut chain = vec![(0.0, 0.0); 9];
    let order = [0, 1, 2, 5, 4, 3, 6, 7, 8];
    for w in order.windows(2) {
        let (i, j) = (w[0], w[1]);
        let sign = if i < j { 1.0 } else { -1.0 };
        let d = pairs.iter().find(|p| (p.i, p.j) == (i.min(j), i.max(j))).map(|p| (p.dx, p.dy));
        let (a, b) = (i.min(j), i.max(j));
        let d = d.unwrap_or((stage[b].0 - stage[a].0, stage[b].1 - stage[a].1)); // untrusted: the stage
        chain[j] = (chain[i].0 + sign * d.0, chain[i].1 + sign * d.1);
    }
    let global = global_positions(9, &pairs).expect("every tile has a trusted pair");
    println!("largest position error: stage {:.1} px, chain 0-1-2-5-4-3-6-7-8 {:.1} px, global {:.1} px", max_error(&stage, &truth), max_error(&chain, &truth), max_error(&global, &truth));
    print!("residuals after global alignment (length, px):");
    for p in &pairs {
        let r = (global[p.j].0 - global[p.i].0 - p.dx).hypot(global[p.j].1 - global[p.i].1 - p.dy);
        print!("  {}-{} {r:.2}", p.i, p.j);
    }
    println!();

    // The mosaic, each tile pasted at its position (later tiles on top).
    for (method, pos) in [("stage", &stage), ("global", &global)] {
        let mut out = Image::new(STEP * 2 + TILE + 20, STEP * 2 + TILE + 20, 0.0f32);
        for (k, tile) in tiles.iter().enumerate() {
            let (ox, oy) = (pos[k].0.round() as isize + 10, pos[k].1.round() as isize + 10);
            for (x, y) in tile.coords() {
                let (mx, my) = (ox + x as isize, oy + y as isize);
                if out.contains(mx, my) {
                    out[(mx as usize, my as usize)] = tile[(x, y)];
                }
            }
        }
        save_gray(&out, format!("{dir}/{name}-{method}.png"));
    }
}

pub fn main() {
    let dir = "out/stitching";
    let field = field();
    mosaic("mosaic", &field, 0.7);

    // The same, but the region shared by tiles 4 and 5 is empty (flat grey with noise).
    let mut rng = Rng::new(9);
    let blank = Image::from_fn(384, 384, |x, y| if (212..248).contains(&x) && (108..248).contains(&y) { 0.6 + 0.03 * rng.normal() } else { field[(x, y)] });
    save_gray(&blank, format!("{dir}/field-empty-overlap.png"));
    mosaic("empty-all-pairs", &blank, 0.0);
    mosaic("empty-trusted", &blank, 0.7);

    // Sub-pixel: a row of the micrograph (64 samples) and a copy shifted by t0 with the Fourier
    // shift theorem; the phase-correlation peak and the parabola through its top three samples.
    println!("\n true shift   highest sample   parabola   error");
    let row: Vec<Complex> = samples::cell().row(100)[96..160].iter().map(|&v| Complex::from(v)).collect();
    let fa = fft(&row, false);
    for t0 in [3.0f32, 3.1, 3.3, 3.5, 3.7] {
        // B(x) = A(x + t0): multiply the spectrum by e^{2πi k t0 / N}
        let fb: Vec<Complex> = fa.iter().enumerate().map(|(k, &v)| v * Complex::from_angle(2.0 * std::f32::consts::PI * fft::frequency(k, 64) * t0 / 64.0)).collect();
        let r: Vec<f32> = fft(&fa.iter().zip(&fb).map(|(&p, &q)| { let c = p * q.conj(); c / (c.abs() + 1e-6) }).collect::<Vec<_>>(), true).iter().map(|z| z.re).collect();
        let k = (0..64).max_by(|&i, &j| r[i].total_cmp(&r[j])).unwrap();
        let est = k as f32 + parabola(r[(k + 63) % 64], r[k], r[(k + 1) % 64]);
        println!("{t0:10.2} {k:16} {est:10.3} {:7.3}", est - t0);
    }
    println!("wrote the results to {dir}/");
}
