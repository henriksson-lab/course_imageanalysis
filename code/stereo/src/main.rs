// Matches a synthetic rectified stereo pair with your cost volumes, winner-take-all, scanline
// dynamic programming and sub-pixel refinement. Disparity maps in out/stereo/.
//   cargo run -p stereo                      (your code)
//   cargo run -p stereo --features solution  (the model solution)

use imgcourse::*;
use stereo::*;

const W: usize = 160;
const H: usize = 120;
const MAX_D: usize = 32;
/// The raised square, in left-image coordinates, and its disparity.
const SQ: (usize, usize, usize, usize) = (60, 125, 30, 90);
const D_SQ: f32 = 26.0;

/// The slanted ground: disparity 6 at the top row, 18 at the bottom.
fn ground_d(y: usize) -> f32 {
    6.0 + 12.0 * y as f32 / (H - 1) as f32
}

fn in_square(x: f32, y: usize) -> bool {
    x >= SQ.0 as f32 && x < SQ.1 as f32 && y >= SQ.2 && y < SQ.3
}

/// The pair, the true disparity of the left image, and the mask of the left pixels that can be
/// evaluated (not occluded in the right image, and not in the left strip x < MAX_D). With
/// `textureless`, the inside of the square is one flat grey, with only a textured frame.
fn make_pair(textureless: bool) -> (Image<f32>, Image<f32>, Image<f32>, Image<bool>) {
    let mut rng = Rng::new(11);
    let noise_tex: Vec<f32> = (0..(W + 64) * H).map(|_| 0.1 + 0.8 * rng.uniform()).collect();
    // the texture of the ground, between grid points by linear interpolation along x
    let ground = |x: f32, y: usize| {
        let x = x.clamp(0.0, (W + 62) as f32);
        let i = x.floor() as usize;
        let t = x - i as f32;
        (1.0 - t) * noise_tex[y * (W + 64) + i] + t * noise_tex[y * (W + 64) + i + 1]
    };
    // the square: a textured frame 5 pixels wide around a textureless inside
    let square = |x: usize, y: usize| {
        let inside = textureless && x >= SQ.0 + 5 && x < SQ.1 - 5 && y >= SQ.2 + 5 && y < SQ.3 - 5;
        if inside { 0.5 } else { ground(x as f32 + 40.0, (y + 50) % H) }
    };
    let left = Image::from_fn(W, H, |x, y| if in_square(x as f32, y) { square(x, y) } else { ground(x as f32, y) });
    let right = Image::from_fn(W, H, |x, y| {
        let xs = x as f32 + D_SQ;
        if in_square(xs, y) { square(xs as usize, y) } else { ground(x as f32 + ground_d(y), y) }
    });
    let truth = Image::from_fn(W, H, |x, y| if in_square(x as f32, y) { D_SQ } else { ground_d(y) });
    let valid = Image::from_fn(W, H, |x, y| {
        // a ground pixel is occluded if the square covers its position in the right image
        let occluded = !in_square(x as f32, y) && in_square(x as f32 - ground_d(y) + D_SQ, y);
        x >= MAX_D && !occluded
    });
    (samples::add_noise(&left, 0.01, 1), samples::add_noise(&right, 0.01, 2), truth, valid)
}

/// The percentage of evaluated pixels whose disparity is more than 1 off.
fn bad(disp: &Image<f32>, truth: &Image<f32>, valid: &Image<bool>) -> f32 {
    let (mut n, mut wrong) = (0, 0);
    for (x, y) in disp.coords() {
        if valid[(x, y)] {
            n += 1;
            if (disp[(x, y)] - truth[(x, y)]).abs() > 1.0 {
                wrong += 1;
            }
        }
    }
    100.0 * wrong as f32 / n as f32
}

/// The census cost volume: the Hamming distances of the census codes, summed over the window.
fn census_cost_volume(left: &Image<f32>, right: &Image<f32>, max_d: usize, radius: usize) -> Vec<Image<f32>> {
    let (cl, cr) = (census(left), census(right));
    let r = radius as isize;
    (0..=max_d as isize)
        .map(|d| {
            cl.map_xy(|x, y, _| {
                let mut sum = 0;
                for j in -r..=r {
                    for i in -r..=r {
                        let (xl, yy) = ((x as isize + i).clamp(0, W as isize - 1), (y as isize + j).clamp(0, H as isize - 1));
                        let xr = (xl - d).max(0);
                        sum += hamming(cl[(xl as usize, yy as usize)], cr[(xr as usize, yy as usize)]);
                    }
                }
                sum as f32
            })
        })
        .collect()
}

fn save_disp(d: &Image<f32>, path: String) {
    save_gray(&d.map(|v| v / MAX_D as f32), path);
}

fn main() {
    let dir = "out/stereo";
    let (left, right, truth, valid) = make_pair(false);
    save_gray(&left, format!("{dir}/left.png"));
    save_gray(&right, format!("{dir}/right.png"));
    save_disp(&truth, format!("{dir}/truth.png"));

    // 1. SAD and winner-take-all with three window sizes.
    let mut sad7 = Vec::new();
    for radius in [1, 3, 7] {
        let costs = sad_cost_volume(&left, &right, MAX_D, radius);
        let d = winner_take_all(&costs);
        let n = 2 * radius + 1;
        println!("SAD {n:2} × {n:2}, winner-take-all:  {:5.1} % bad pixels", bad(&d, &truth, &valid));
        save_disp(&d, format!("{dir}/wta-sad{n}.png"));
        if radius == 3 {
            sad7 = costs;
        }
    }

    // 2. A square with a textureless inside: winner-take-all and scanline dynamic programming
    //    on the 3 × 3 SAD costs.
    let (left_flat, right_flat, _, _) = make_pair(true);
    save_gray(&left_flat, format!("{dir}/left-textureless.png"));
    let costs = sad_cost_volume(&left_flat, &right_flat, MAX_D, 1);
    let wta = winner_take_all(&costs);
    println!("textureless inside, SAD 3 × 3, winner-take-all: {:5.1} % bad pixels", bad(&wta, &truth, &valid));
    save_disp(&wta, format!("{dir}/textureless-wta-sad3.png"));
    let mut dp = Image::new(W, H, 0.0f32);
    for y in 0..H {
        let row: Vec<Vec<f32>> = (0..W).map(|x| (0..=MAX_D).map(|d| costs[d][(x, y)]).collect()).collect();
        for (x, d) in scanline_dp(&row, 0.3, 3.0).into_iter().enumerate() {
            dp[(x, y)] = d as f32;
        }
    }
    println!("textureless inside, SAD 3 × 3, scanline DP:     {:5.1} % bad pixels", bad(&dp, &truth, &valid));
    save_disp(&dp, format!("{dir}/textureless-dp-sad3.png"));

    // 3. A gain change: the right camera is less sensitive and has an offset.
    let right_gain = right.map(|v| 0.5 * v + 0.3);
    let d = winner_take_all(&sad_cost_volume(&left, &right_gain, MAX_D, 3));
    println!("gain change, SAD 7 × 7:          {:5.1} % bad pixels", bad(&d, &truth, &valid));
    save_disp(&d, format!("{dir}/gain-sad7.png"));
    let d = winner_take_all(&census_cost_volume(&left, &right_gain, MAX_D, 3));
    println!("gain change, census 7 × 7:       {:5.1} % bad pixels", bad(&d, &truth, &valid));
    save_disp(&d, format!("{dir}/gain-census7.png"));

    // 4. Sub-pixel disparity on the slanted ground (SAD 7 × 7): the mean error of the good
    //    pixels, with integer disparities and with the parabola.
    let d = winner_take_all(&sad7);
    let (mut n, mut e_int, mut e_sub) = (0, 0.0, 0.0);
    for (x, y) in d.coords() {
        let k = d[(x, y)] as usize;
        if valid[(x, y)] && !in_square(x as f32, y) && (d[(x, y)] - truth[(x, y)]).abs() <= 1.0 && k > 0 && k < MAX_D {
            let off = subpixel_offset(sad7[k - 1][(x, y)], sad7[k][(x, y)], sad7[k + 1][(x, y)]);
            n += 1;
            e_int += (d[(x, y)] - truth[(x, y)]).abs();
            e_sub += (d[(x, y)] + off - truth[(x, y)]).abs();
        }
    }
    println!("sub-pixel, ground: mean error {:.3} px with whole disparities, {:.3} px with the parabola ({n} pixels)", e_int / n as f32, e_sub / n as f32);
    println!("wrote the pictures to {dir}/");
}
