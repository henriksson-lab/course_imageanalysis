// Runs your coordinate chain on a small scene of houses and draws wireframes in out/pipeline3d/.
//   cargo run -p pipeline3d                      (your code)
//   cargo run -p pipeline3d --features solution  (the model solution)

use imgcourse::*;
use pipeline3d::*;

const W: usize = 320;
const H: usize = 240;

/// A house in model space: a box 1.2 wide (x), 0.6 high and 1 deep (z), with a roof ridge
/// along z at height 1. Vertices 0–3 floor, 4–7 top of the walls, 8–9 the ridge.
fn house() -> (Vec<Vec3>, Vec<[usize; 2]>) {
    let v = vec![
        [-0.6, 0., 0.5], [0.6, 0., 0.5], [0.6, 0., -0.5], [-0.6, 0., -0.5],
        [-0.6, 0.6, 0.5], [0.6, 0.6, 0.5], [0.6, 0.6, -0.5], [-0.6, 0.6, -0.5],
        [0., 1., 0.5], [0., 1., -0.5],
    ];
    let e = vec![
        [0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7], [4, 8], [5, 8], [6, 9], [7, 9], [8, 9],
    ];
    (v, e)
}

/// A post: a thin vertical line from the ground to height `h` at (x, z).
fn post(x: f64, z: f64, h: f64) -> (Vec<Vec3>, Vec<[usize; 2]>) {
    (vec![[x, 0., z], [x, h, z]], vec![[0, 1]])
}

/// Draw a straight line between two screen points (y up) in steps of half a pixel.
fn draw_line(img: &mut Image<[f32; 3]>, a: Vec3, b: Vec3, colour: [f32; 3]) {
    let n = ((b[0] - a[0]).abs().max((b[1] - a[1]).abs()) * 2.0).ceil().max(1.0) as usize;
    for i in 0..=n {
        let t = i as f64 / n as f64;
        let x = a[0] + t * (b[0] - a[0]);
        let y = H as f64 - (a[1] + t * (b[1] - a[1])); // the screen has y up, the image y down
        if x >= 0.0 && y >= 0.0 && (x as usize) < W && (y as usize) < H {
            img[(x as usize, y as usize)] = colour;
        }
    }
}

/// Draw the edges of a mesh placed by `model` and seen through `view_proj` (= P V).
fn draw(img: &mut Image<[f32; 3]>, mesh: &(Vec<Vec3>, Vec<[usize; 2]>), model: &Mat4, view_proj: &Mat4, colour: [f32; 3]) {
    let pvm = mat_mul(view_proj, model);
    let screen: Vec<Option<Vec3>> = mesh.0.iter().map(|p| to_screen(apply(&pvm, [p[0], p[1], p[2], 1.0]), W as f64, H as f64)).collect();
    for e in &mesh.1 {
        if let (Some(a), Some(b)) = (screen[e[0]], screen[e[1]]) {
            draw_line(img, a, b, colour);
        }
    }
}

fn main() {
    let dir = "out/pipeline3d";
    let aspect = W as f64 / H as f64;
    let (white, orange, blue) = ([1.0, 1.0, 1.0], [1.0, 0.6, 0.1], [0.3, 0.6, 1.0]);

    // 1. A street: five equal houses, 2.5 apart along −z, seen from the side of the street.
    let view = look_at([3.0, 1.6, 3.0], [0.0, 0.8, -4.0], [0.0, 1.0, 0.0]);
    let pv = mat_mul(&perspective(50.0, aspect, 0.5, 50.0), &view);
    let mut img = Image::new(W, H, [0.0f32; 3]);
    for k in 0..5 {
        let model = mat_mul(&translation([0.0, 0.0, -2.5 * k as f64]), &rotation_y(90.0));
        draw(&mut img, &house(), &model, &pv, white);
    }
    save_rgb(&img, format!("{dir}/street.png"));

    // 2. The dolly zoom: the house at the origin, a row of posts 6 behind it. From distance 4
    //    with a 40° field of view, then from distance 9 with 18.4° (exercise B7).
    let scene_posts: Vec<_> = (-4..=4).map(|i| post(1.2 * i as f64, -6.0, 1.5)).collect();
    for (d, fov) in [(4.0, 40.0), (9.0, 18.4)] {
        let view = look_at([0.0, 0.8, d], [0.0, 0.5, 0.0], [0.0, 1.0, 0.0]);
        let pv = mat_mul(&perspective(fov, aspect, 0.5, 50.0), &view);
        let mut img = Image::new(W, H, [0.0f32; 3]);
        for p in &scene_posts {
            draw(&mut img, p, &IDENTITY, &pv, blue);
        }
        draw(&mut img, &house(), &rotation_y(30.0), &pv, orange);
        save_rgb(&img, format!("{dir}/dolly-{d}.png"));
        // the house's extent in NDC x
        let pvm = mat_mul(&pv, &rotation_y(30.0));
        let xs: Vec<f64> = house().0.iter().map(|p| {
            let c = apply(&pvm, [p[0], p[1], p[2], 1.0]);
            c[0] / c[3]
        }).collect();
        let (lo, hi) = (xs.iter().cloned().fold(f64::INFINITY, f64::min), xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max));
        // the middle post's height on the screen
        let bottom = to_screen(apply(&pv, [0.0, 0.0, -6.0, 1.0]), W as f64, H as f64).unwrap();
        let top = to_screen(apply(&pv, [0.0, 1.5, -6.0, 1.0]), W as f64, H as f64).unwrap();
        println!("dolly zoom, distance {d}, field of view {fov}°: house {:.0} pixels wide, a post {:.0} pixels high", (hi - lo) / 2.0 * W as f64, top[1] - bottom[1]);
    }

    // 3. Follow the front end of the ridge (vertex 8) through all the spaces.
    let model = rotation_y(30.0);
    let view = look_at([0.0, 1.5, 4.0], [0.0, 0.5, 0.0], [0.0, 1.0, 0.0]);
    let proj = perspective(40.0, aspect, 1.0, 10.0);
    let p = house().0[8];
    let world = apply(&model, [p[0], p[1], p[2], 1.0]);
    let eye = apply(&view, world);
    let clip = apply(&proj, eye);
    println!("\nvertex 8 (front end of the ridge):");
    println!("  model  {p:.3?}");
    println!("  world  {:.3?}", &world[..3]);
    println!("  view   {:.3?}", &eye[..3]);
    println!("  clip   {clip:.3?}");
    println!("  NDC    {:.3?}", [clip[0] / clip[3], clip[1] / clip[3], clip[2] / clip[3]]);
    println!("  screen {:.1?}", to_screen(clip, W as f64, H as f64));

    // 4. Depth precision: the range of NDC depths the house uses, for two near planes.
    println!();
    for near in [1.0, 0.05] {
        let pvm = mat_mul(&mat_mul(&perspective(40.0, aspect, near, 30.0), &view), &model);
        let zs: Vec<f64> = house().0.iter().map(|p| {
            let c = apply(&pvm, [p[0], p[1], p[2], 1.0]);
            c[2] / c[3]
        }).collect();
        let (lo, hi) = (zs.iter().cloned().fold(f64::INFINITY, f64::min), zs.iter().cloned().fold(f64::NEG_INFINITY, f64::max));
        println!("near = {near:4}, far = 30: the house has NDC depths {lo:.4} to {hi:.4}, {:.2} % of [−1, 1]", 100.0 * (hi - lo) / 2.0);
    }
    println!("\nwrote the pictures to {dir}/");
}
