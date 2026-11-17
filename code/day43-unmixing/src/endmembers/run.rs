// A synthetic 16-channel cell: count its components with singular values, find the spectra
// by vertex finding (with and without pure pixels) and by NMF from two seeds, and compare
// least squares with NNLS in dim pixels. Pictures in out/endmembers/.
//   cargo run -p day43 -- endmembers                      (your code)
//   cargo run -p day43 --features solution -- endmembers  (the model solution)

use crate::endmembers::*;
use imgcourse::*;

const N: usize = 16; // channels
const SIZE: usize = 96;

/// The spectra (columns, scaled to sum 1): nucleus, mitochondria, membrane, autofluorescence.
fn spectra() -> Matrix {
    let centres: Vec<f32> = (0..N).map(|c| 480.0 + 220.0 * (c as f32 + 0.5) / N as f32).collect();
    let bump = |peak: f32, left: f32, right: f32| {
        let s: Vec<f32> = centres.iter().map(|&l| (-0.5 * ((l - peak) / if l < peak { left } else { right }).powi(2)).exp()).collect();
        let t: f32 = s.iter().sum();
        s.into_iter().map(|v| v / t).collect::<Vec<f32>>()
    };
    let cols = [bump(515.0, 12.0, 25.0), bump(570.0, 12.0, 25.0), bump(625.0, 14.0, 28.0), bump(565.0, 60.0, 80.0)];
    (0..N).map(|c| (0..4).map(|k| cols[k][c]).collect()).collect()
}

/// True abundances (4 maps): nucleus, mitochondria, membrane, a dim diffuse autofluorescence,
/// slightly blurred so that boundaries are mixtures. With `pure` false every pixel gets 30 %
/// of the other two of the first three components (no pixel is pure).
fn abundances(pure: bool) -> Vec<Image<f32>> {
    let c = SIZE as f32 / 2.0;
    let r = |x: usize, y: usize, cx: f32, cy: f32| ((x as f32 - cx).powi(2) + (y as f32 - cy).powi(2)).sqrt();
    let mut rng = Rng::new(4);
    let mito: Vec<(f32, f32)> = (0..10)
        .map(|_| {
            let (a, d) = (std::f32::consts::TAU * rng.uniform(), 22.0 + 12.0 * rng.uniform());
            (c + d * a.cos(), c + d * a.sin())
        })
        .collect();
    let sharp = [
        Image::from_fn(SIZE, SIZE, |x, y| if r(x, y, c, c) < 15.0 { 1.0 } else { 0.0 }),
        Image::from_fn(SIZE, SIZE, |x, y| if mito.iter().any(|&(mx, my)| r(x, y, mx, my) < 3.5) { 1.0 } else { 0.0 }),
        Image::from_fn(SIZE, SIZE, |x, y| if (40.0..44.0).contains(&r(x, y, c, c)) { 1.0 } else { 0.0 }),
    ];
    let blur = |img: &Image<f32>| {
        Image::from_fn(SIZE, SIZE, |x, y| {
            let mut s = 0.0;
            for dy in -1..=1 {
                for dx in -1..=1 {
                    s += img.get_border(x as isize + dx, y as isize + dy, Border::Clamp);
                }
            }
            s / 9.0
        })
    };
    let mut maps: Vec<Image<f32>> = sharp.iter().map(blur).collect();
    if !pure {
        let old = maps.clone();
        for k in 0..3 {
            maps[k] = Image::from_fn(SIZE, SIZE, |x, y| 0.7 * old[k][(x, y)] + 0.3 * (old[(k + 1) % 3][(x, y)] + old[(k + 2) % 3][(x, y)]));
        }
    }
    maps.push(Image::from_fn(SIZE, SIZE, |x, y| if r(x, y, c, c) < 40.0 { 0.3 } else { 0.0 }));
    maps
}

/// Every pixel as a vector of N photon counts.
fn record(m: &Matrix, a: &[Image<f32>], photons: f32, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = Rng::new(seed);
    a[0].coords()
        .map(|(x, y)| (0..N).map(|ch| rng.poisson(photons * (0..a.len()).map(|k| m[ch][k] * a[k][(x, y)]).sum::<f32>())).collect())
        .collect()
}

fn angle_deg(u: &[f32], v: &[f32]) -> f32 {
    let dot: f32 = u.iter().zip(v).map(|(a, b)| a * b).sum();
    let nu = u.iter().map(|a| a * a).sum::<f32>().sqrt();
    let nv = v.iter().map(|a| a * a).sum::<f32>().sqrt();
    (dot / (nu * nv)).clamp(-1.0, 1.0).acos().to_degrees()
}

/// For each true spectrum, the angle to the nearest found one.
fn angles(truth: &Matrix, found: &[Vec<f32>], k: usize) -> Vec<f32> {
    (0..k).map(|j| {
        let t: Vec<f32> = truth.iter().map(|row| row[j]).collect();
        found.iter().map(|f| angle_deg(&t, f)).fold(f32::INFINITY, f32::min)
    }).collect()
}

/// Columns of the pixels (channels × pixels) for the pixels in `idx`.
fn as_matrix(pixels: &[Vec<f32>], idx: &[usize]) -> Matrix {
    (0..N).map(|c| idx.iter().map(|&i| pixels[i][c]).collect()).collect()
}

/// Every vector projected onto the `k` leading left singular vectors of the set (the best
/// rank-k approximation): removes the noise outside the signal space.
fn project_top(vectors: &[Vec<f32>], k: usize) -> Vec<Vec<f32>> {
    let mut yyt = vec![vec![0.0f64; N]; N];
    for v in vectors {
        for i in 0..N {
            for j in 0..N {
                yyt[i][j] += v[i] as f64 * v[j] as f64;
            }
        }
    }
    let (_, u) = linalg::sym_eigen(&yyt);
    vectors
        .iter()
        .map(|v| {
            let mut out = vec![0.0f32; N];
            for d in u.iter().take(k) {
                let dot: f64 = (0..N).map(|c| d[c] * v[c] as f64).sum();
                for c in 0..N {
                    out[c] += (dot * d[c]) as f32;
                }
            }
            out
        })
        .collect()
}

/// ‖Y − W H‖² relative to ‖Y‖².
fn rel_error(y: &Matrix, w: &Matrix, h: &Matrix) -> f32 {
    let (mut e, mut t) = (0.0f64, 0.0f64);
    for c in 0..y.len() {
        for q in 0..y[0].len() {
            let fit: f32 = (0..w[0].len()).map(|j| w[c][j] * h[j][q]).sum();
            e += ((y[c][q] - fit) as f64).powi(2);
            t += (y[c][q] as f64).powi(2);
        }
    }
    (e / t) as f32
}

pub fn main() {
    let dir = "out/endmembers";
    let m = spectra();
    let photons = 400.0;

    // 1. How many components? Singular values of all pixels, divided by the largest.
    let pixels = record(&m, &abundances(true), photons, 1);
    let all: Vec<usize> = (0..pixels.len()).collect();
    let sv = singular_values(&as_matrix(&pixels, &all));
    let rel: Vec<String> = sv.iter().take(7).map(|s| format!("{:.3}", s / sv[0])).collect();
    println!("singular values / largest: {}", rel.join(", "));

    // 2. Vertex finding on the bright pixels, normalised to sum 1, with three components
    //    (the autofluorescence is too dim and diffuse to have pure pixels).
    let m3: Matrix = m.iter().map(|row| row[..3].to_vec()).collect();
    let mut cloud_pics = vec![];
    for pure in [true, false] {
        let mut ab = abundances(pure);
        ab.pop(); // no autofluorescence here
        let px = record(&m3, &ab, photons, 2);
        let bright: Vec<usize> = (0..px.len()).filter(|&i| px[i].iter().sum::<f32>() > 0.3 * photons).collect();
        let norm: Vec<Vec<f32>> = bright.iter().map(|&i| {
            let t: f32 = px[i].iter().sum();
            px[i].iter().map(|v| v / t).collect()
        }).collect();
        // remove the noise outside the 3-dimensional signal space first (slide "The math:
        // volume and the best rank-K approximation"), then look for the corners
        let clean = project_top(&norm, 3);
        let found_idx = find_vertices(&clean, 3);
        let found: Vec<Vec<f32>> = found_idx.iter().map(|&i| clean[i].clone()).collect();
        println!("vertex finding, pure pixels {pure:5}: {} bright pixels, angles to the true spectra {:.1?} degrees", bright.len(), angles(&m3, &found, 3));
        cloud_pics.push(draw_cloud(&norm, &m3, &found));
    }
    let both = Image::from_fn(2 * 240 + 8, 240, |x, y| if x < 240 { cloud_pics[0][(x, y)] } else if x < 248 { [1.0; 3] } else { cloud_pics[1][(x - 248, y)] });
    save_rgb(&both, format!("{dir}/vertices.png"));

    // 3. NMF without pure pixels, from two seeds, compared with the true spectra.
    let mut ab = abundances(false);
    ab.pop();
    let px = record(&m3, &ab, photons, 3);
    let bright: Vec<usize> = (0..px.len()).filter(|&i| px[i].iter().sum::<f32>() > 0.3 * photons).collect();
    let y = as_matrix(&px, &bright);
    let p = bright.len();
    // the true spectra with NNLS abundances
    let mt: Matrix = m3.iter().map(|row| row.iter().map(|v| v * photons).collect()).collect();
    let mut ht = vec![vec![0.0f32; p]; 3];
    for q in 0..p {
        let a = nnls(&mt, &px[bright[q]]);
        for j in 0..3 {
            ht[j][q] = a[j];
        }
    }
    println!("true spectra (NNLS abundances): relative error {:.5}", rel_error(&y, &mt, &ht));
    for seed in [1, 2] {
        let mut rng = Rng::new(seed);
        let mut w: Matrix = (0..N).map(|_| (0..3).map(|_| 0.1 + rng.uniform()).collect()).collect();
        let mut h: Matrix = (0..3).map(|_| (0..p).map(|_| 0.1 + rng.uniform()).collect()).collect();
        for _ in 0..400 {
            update_h(&y, &w, &mut h);
            update_w(&y, &mut w, &h);
        }
        let found: Vec<Vec<f32>> = (0..3).map(|j| w.iter().map(|row| row[j]).collect()).collect();
        println!("NMF seed {seed}: relative error {:.5}, angles to the true spectra {:.1?} degrees", rel_error(&y, &w, &h), angles(&m3, &found, 3));
    }

    // 4. Dim pixels: least squares against NNLS (four components, 40 photons).
    let dim = record(&m, &abundances(true), 40.0, 4);
    let md: Matrix = m.iter().map(|row| row.iter().map(|v| v * 40.0).collect()).collect();
    let (mut neg, mut total) = (0, 0);
    let rows: Vec<Vec<f64>> = md.iter().map(|r| r.iter().map(|&v| v as f64).collect()).collect();
    let mut nnls_map = Image::new(SIZE, SIZE, 0.0f32);
    let mut ls_map = Image::new(SIZE, SIZE, 0.0f32);
    for (i, pxl) in dim.iter().enumerate() {
        let yd: Vec<f64> = pxl.iter().map(|&v| v as f64).collect();
        let ls = linalg::least_squares(&rows, &yd).unwrap();
        neg += ls.iter().filter(|&&v| v < 0.0).count();
        total += 4;
        let nn = nnls(&md, pxl);
        ls_map[(i % SIZE, i / SIZE)] = ls[1] as f32;
        nnls_map[(i % SIZE, i / SIZE)] = nn[1];
    }
    println!("40 photons: least squares gives {:.1} % negative abundances, NNLS none", 100.0 * neg as f32 / total as f32);
    let pair = Image::from_fn(2 * SIZE + 4, SIZE, |x, y| if x < SIZE { ls_map[(x, y)] } else if x < SIZE + 4 { 0.0 } else { nnls_map[(x - SIZE - 4, y)] });
    save_gray_signed(&pair, format!("{dir}/mito-ls-vs-nnls.png"));
    println!("wrote the pictures to {dir}/");
}

/// The normalised pixels projected on the plane of the true spectra (in their simplex):
/// grey dots, true spectra as open circles, found ones as filled squares.
fn draw_cloud(norm: &[Vec<f32>], m3: &Matrix, found: &[Vec<f32>]) -> Image<[f32; 3]> {
    // the true spectra, normalised to sum 1 (they already are), as triangle corners
    let corners = [[30.0f32, 210.0], [210.0, 210.0], [120.0, 30.0]];
    let truth: Vec<Vec<f32>> = (0..3).map(|j| m3.iter().map(|row| row[j]).collect()).collect();
    // position of a vector = its least-squares fractions in the true spectra, placed in the triangle
    let place = |v: &[f32]| {
        let rows: Vec<Vec<f64>> = (0..N).map(|c| (0..3).map(|j| truth[j][c] as f64).collect()).collect();
        let yd: Vec<f64> = v.iter().map(|&x| x as f64).collect();
        let a = linalg::least_squares(&rows, &yd).unwrap();
        let s: f64 = a.iter().sum();
        let mut p = [0.0f32; 2];
        for j in 0..3 {
            for d in 0..2 {
                p[d] += (a[j] / s) as f32 * corners[j][d];
            }
        }
        p
    };
    let mut img = Image::new(240, 240, [1.0f32; 3]);
    let mut dot = |p: [f32; 2], r: i32, colour: [f32; 3], filled: bool| {
        for dy in -r..=r {
            for dx in -r..=r {
                let d2 = dx * dx + dy * dy;
                let on = if filled { d2 <= r * r } else { d2 <= r * r && d2 >= (r - 1) * (r - 1) };
                let (x, y) = (p[0] as i32 + dx, p[1] as i32 + dy);
                if on && (0..240).contains(&x) && (0..240).contains(&y) {
                    img[(x as usize, y as usize)] = colour;
                }
            }
        }
    };
    for v in norm {
        dot(place(v), 0, [0.6; 3], true);
    }
    for c in corners {
        dot(c, 7, [0.0, 0.0, 0.0], false);
    }
    for f in found {
        dot(place(f), 4, [0.85, 0.3, 0.0], true);
    }
    img
}
