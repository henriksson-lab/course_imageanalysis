// Marching cubes on test volumes with your code: counts, topology, and shaded pictures of the
// meshes (face normals and gradient normals) in out/marchingcubes/.
//   cargo run -p marchingcubes -- marchingcubes                      (your code)
//   cargo run -p marchingcubes --features solution -- marchingcubes  (the model solution)

use imgcourse::*;
use crate::common::blur3;
use crate::marchingcubes::*;
use std::collections::HashMap;

/// A volume of size n³ whose voxel (x, y, z) is f(x, y, z).
fn volume(n: usize, f: impl Fn(f32, f32, f32) -> f32) -> Volume<f32> {
    Volume::from_fn(n, n, n, |x, y, z| f(x as f32, y as f32, z as f32))
}

/// Vertices, edges, triangles, connected components, boundary edges (cracks or holes) and the
/// Euler characteristic χ = V − E + F of a mesh.
fn topology(mesh: &Mesh) -> (usize, usize, usize, usize, usize, i64) {
    let mut edges: HashMap<(usize, usize), usize> = HashMap::new();
    let mut parent: Vec<usize> = (0..mesh.vertices.len()).collect();
    fn root(p: &mut [usize], mut i: usize) -> usize {
        while p[i] != i {
            p[i] = p[p[i]];
            i = p[i];
        }
        i
    }
    for t in &mesh.triangles {
        for i in 0..3 {
            let (a, b) = (t[i].min(t[(i + 1) % 3]), t[i].max(t[(i + 1) % 3]));
            *edges.entry((a, b)).or_default() += 1;
            let (ra, rb) = (root(&mut parent, a), root(&mut parent, b));
            parent[ra] = rb;
        }
    }
    let components = (0..parent.len()).filter(|&i| root(&mut parent, i) == i).count();
    let boundary = edges.values().filter(|&&n| n == 1).count();
    let (v, e, f) = (mesh.vertices.len(), edges.len(), mesh.triangles.len());
    (v, e, f, components, boundary, v as i64 - e as i64 + f as i64)
}

/// A shaded picture of the mesh, seen from a direction turned by `yaw` about the vertical and
/// `pitch` about the horizontal (degrees), with a z-buffer. `smooth`: interpolate the vertex
/// normals (gradient normals); otherwise one normal per triangle (face normals).
fn render(mesh: &Mesh, yaw: f32, pitch: f32, smooth: bool, size: usize) -> Image<f32> {
    let (lo, hi) = mesh.vertices.iter().fold(([f32::MAX; 3], [f32::MIN; 3]), |(lo, hi), p| {
        (std::array::from_fn(|i| lo[i].min(p[i])), std::array::from_fn(|i| hi[i].max(p[i])))
    });
    let centre: [f32; 3] = std::array::from_fn(|i| 0.5 * (lo[i] + hi[i]));
    let radius = (0..3).map(|i| (hi[i] - lo[i]).powi(2)).sum::<f32>().sqrt() / 2.0;
    let (cy, sy, cp, sp) = (yaw.to_radians().cos(), yaw.to_radians().sin(), pitch.to_radians().cos(), pitch.to_radians().sin());
    let turn = |v: [f32; 3]| {
        let (x, z) = (cy * v[0] + sy * v[2], -sy * v[0] + cy * v[2]);
        [x, cp * v[1] - sp * z, sp * v[1] + cp * z]
    };
    let scale = 0.47 * size as f32 / radius;
    let turned: Vec<[f32; 3]> = mesh.vertices.iter().map(|p| turn([p[0] - centre[0], p[1] - centre[1], p[2] - centre[2]])).collect();
    let screen: Vec<[f32; 3]> = turned.iter().map(|q| [size as f32 / 2.0 + scale * q[0], size as f32 / 2.0 + scale * q[1], q[2]]).collect();
    let normals: Vec<[f32; 3]> = mesh.normals.iter().map(|&n| turn(n)).collect();
    let light = { let l = [-0.4f32, -0.5, -1.0]; let n = (l[0] * l[0] + l[1] * l[1] + l[2] * l[2]).sqrt(); l.map(|v| v / n) };
    let mut depth = Image::new(size, size, f32::MAX);
    let mut img = Image::new(size, size, 1.0f32);
    for t in &mesh.triangles {
        let [a, b, c] = t.map(|v| screen[v]);
        let face = {
            let [a, b, c] = t.map(|v| turned[v]);
            let (u, w) = ([b[0] - a[0], b[1] - a[1], b[2] - a[2]], [c[0] - a[0], c[1] - a[1], c[2] - a[2]]);
            let n = [u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0]];
            let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt().max(1e-12);
            n.map(|v| v / l)
        };
        let area = (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]);
        if area.abs() < 1e-9 {
            continue;
        }
        let (x0, x1) = (a[0].min(b[0]).min(c[0]).max(0.0) as usize, (a[0].max(b[0]).max(c[0]).ceil() as usize).min(size - 1));
        let (y0, y1) = (a[1].min(b[1]).min(c[1]).max(0.0) as usize, (a[1].max(b[1]).max(c[1]).ceil() as usize).min(size - 1));
        for y in y0..=y1 {
            for x in x0..=x1 {
                let (px, py) = (x as f32 + 0.5, y as f32 + 0.5);
                let wa = ((b[0] - px) * (c[1] - py) - (c[0] - px) * (b[1] - py)) / area;
                let wb = ((c[0] - px) * (a[1] - py) - (a[0] - px) * (c[1] - py)) / area;
                let wc = 1.0 - wa - wb;
                if wa < 0.0 || wb < 0.0 || wc < 0.0 {
                    continue;
                }
                let z = wa * a[2] + wb * b[2] + wc * c[2];
                if z >= depth[(x, y)] {
                    continue;
                }
                depth[(x, y)] = z;
                let n = if smooth {
                    let n: [f32; 3] = std::array::from_fn(|i| wa * normals[t[0]][i] + wb * normals[t[1]][i] + wc * normals[t[2]][i]);
                    let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt().max(1e-12);
                    n.map(|v| v / l)
                } else {
                    face
                };
                let lambert = (n[0] * light[0] + n[1] * light[1] + n[2] * light[2]).max(0.0);
                img[(x, y)] = 0.12 + 0.8 * lambert;
            }
        }
    }
    img
}

fn print_topology(name: &str, mesh: &Mesh) {
    let (v, e, f, comp, boundary, chi) = topology(mesh);
    println!("{name:34} V {v:6}  E {e:6}  F {f:6}  components {comp:5}  boundary edges {boundary:4}  χ = {chi}");
}

pub fn main() {
    let dir = "out/marchingcubes";
    std::fs::create_dir_all(dir).unwrap();
    let one = [1.0, 1.0, 1.0];

    // A small sphere of radius 4.3 voxels: face normals against gradient normals.
    let sphere = volume(12, |x, y, z| (1.0 - ((x - 5.5).powi(2) + (y - 5.5).powi(2) + (z - 5.5).powi(2)).sqrt() / 8.6).max(0.0));
    let mesh = marching_cubes(&sphere, one, 0.5, FaceRule::Decider);
    print_topology("sphere r = 4.3 in 12³", &mesh);
    save_gray(&render(&mesh, 30.0, 20.0, false, 256), format!("{dir}/sphere-face-normals.png"));
    save_gray(&render(&mesh, 30.0, 20.0, true, 256), format!("{dir}/sphere-gradient-normals.png"));

    // Two Gaussian blobs (value 1 at the centres, σ = 6, 20 voxels apart) as c drops.
    let blob = |x: f32, y: f32, z: f32, cx: f32| (-((x - cx).powi(2) + (y - 23.5).powi(2) + (z - 23.5).powi(2)) / 72.0).exp();
    let blobs = volume(48, |x, y, z| blob(x, y, z, 13.5) + blob(x, y, z, 33.5));
    for c in [0.6, 0.5, 0.49] {
        let mesh = marching_cubes(&blobs, one, c, FaceRule::Decider);
        print_topology(&format!("two blobs, c = {c}"), &mesh);
        if c == 0.49 {
            save_gray(&render(&mesh, 20.0, 25.0, true, 256), format!("{dir}/blobs-c049.png"));
        }
    }

    // A torus: one hole.
    let torus = volume(48, |x, y, z| {
        let ring = (((x - 23.5).powi(2) + (y - 23.5).powi(2)).sqrt() - 13.0).hypot(z - 23.5);
        (0.5 + (5.5 - ring) / 3.0).clamp(0.0, 1.0)
    });
    let mesh = marching_cubes(&torus, one, 0.5, FaceRule::Decider);
    print_topology("torus R = 13, r = 5.5", &mesh);
    save_gray(&render(&mesh, 0.0, 55.0, true, 256), format!("{dir}/torus.png"));

    // A noisy ball of radius 15: without and with blurring first, and the complement rule.
    let ball = volume(48, |x, y, z| (0.5 + (15.0 - ((x - 23.5).powi(2) + (y - 23.5).powi(2) + (z - 23.5).powi(2)).sqrt()) / 2.0).clamp(0.0, 1.0));
    let mut rng = Rng::new(5);
    let noisy = Volume::from_fn(48, 48, 48, |x, y, z| {
        let border = x == 0 || y == 0 || z == 0 || x == 47 || y == 47 || z == 47;
        if border { 0.0 } else { ball[(x, y, z)] + 0.25 * rng.normal() } // a dark frame closes every surface
    });
    let mesh = marching_cubes(&noisy, one, 0.5, FaceRule::Decider);
    print_topology("noisy ball, decider", &mesh);
    save_gray(&render(&mesh, 30.0, 20.0, true, 256), format!("{dir}/noisy-ball.png"));
    print_topology("noisy ball, complement rule", &marching_cubes(&noisy, one, 0.5, FaceRule::Complement));
    let blurred = blur3(&noisy, 1.5);
    let mesh = marching_cubes(&blurred, one, 0.5, FaceRule::Decider);
    print_topology("noisy ball blurred σ = 1.5", &mesh);
    save_gray(&render(&mesh, 30.0, 20.0, true, 256), format!("{dir}/noisy-ball-blurred.png"));
    println!("wrote the pictures to {dir}/");
}
