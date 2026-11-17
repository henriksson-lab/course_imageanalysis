// Runs your rasteriser on a coarse sphere (three ways of shading) and on a checkerboard floor
// seen in perspective (with and without the correction); pictures in out/rasterise/.
//   cargo run -p graphics3d -- rasterise                      (your code)
//   cargo run -p graphics3d --features solution -- rasterise  (the model solution)

use imgcourse::*;
use crate::rasterise::*;

/// A coarse sphere of radius 1: `stacks` × `slices` quads, each split into two triangles,
/// all wound counter-clockwise seen from outside. Returns the vertices and the triangles.
fn sphere(stacks: usize, slices: usize) -> (Vec<Vec3>, Vec<[usize; 3]>) {
    let mut v = Vec::new();
    for i in 0..=stacks {
        let theta = std::f64::consts::PI * i as f64 / stacks as f64;
        for j in 0..slices {
            let phi = 2.0 * std::f64::consts::PI * j as f64 / slices as f64;
            v.push([theta.sin() * phi.cos(), theta.cos(), theta.sin() * phi.sin()]);
        }
    }
    let id = |i: usize, j: usize| i * slices + j % slices;
    let mut t = Vec::new();
    for i in 0..stacks {
        for j in 0..slices {
            for tri in [[id(i, j), id(i + 1, j), id(i + 1, j + 1)], [id(i, j), id(i + 1, j + 1), id(i, j + 1)]] {
                let n = face(&v, tri);
                if n == [0.0; 3] {
                    continue; // a triangle squeezed to a line at a pole
                }
                // make it wind counter-clockwise seen from outside: its normal points away from 0
                let c = v[tri[0]];
                if dot(n, c) < 0.0 {
                    t.push([tri[0], tri[2], tri[1]]);
                } else {
                    t.push(tri);
                }
            }
        }
    }
    (v, t)
}

/// The face normal (b − a) × (c − a), not scaled (zero for a degenerate triangle).
fn face(v: &[Vec3], t: [usize; 3]) -> Vec3 {
    let (a, b, c) = (v[t[0]], v[t[1]], v[t[2]]);
    let (p, q) = ([b[0] - a[0], b[1] - a[1], b[2] - a[2]], [c[0] - a[0], c[1] - a[1], c[2] - a[2]]);
    let n = [p[1] * q[2] - p[2] * q[1], p[2] * q[0] - p[0] * q[2], p[0] * q[1] - p[1] * q[0]];
    if dot(n, n) < 1e-20 {
        [0.0; 3]
    } else {
        n
    }
}

/// The image with row 0 at the bottom (the screen has y up) turned the right way for saving.
fn flip(img: &Image<f32>) -> Image<f32> {
    Image::from_fn(img.width(), img.height(), |x, y| img[(x, img.height() - 1 - y)])
}

pub fn main() {
    let dir = "out/rasterise";

    // 1. A coarse sphere, seen orthographically along −z, 240 × 240, three ways of shading.
    let (verts, tris) = sphere(6, 10);
    let size = 240;
    let to_screen = |p: Vec3| [size as f64 / 2.0 + 100.0 * p[0], size as f64 / 2.0 + 100.0 * p[1], -p[2]];
    let light = normalize([-0.5, 0.6, 1.0]);
    let eye = [0.0, 0.0, 1.0];
    let shade = |n: Vec3| blinn_phong(normalize(n), light, eye, 0.1, 0.7, 0.6, 40.0) as f32;
    println!("sphere: {} vertices, {} triangles", verts.len(), tris.len());
    for mode in ["flat", "gouraud", "pixel"] {
        let mut colour = Image::new(size, size, 0.0f32);
        let mut depth = Image::new(size, size, f32::INFINITY);
        let mut drawn = 0;
        for &t in &tris {
            let corners = [to_screen(verts[t[0]]), to_screen(verts[t[1]]), to_screen(verts[t[2]])];
            let normals = [verts[t[0]], verts[t[1]], verts[t[2]]]; // on a unit sphere, normal = position
            let f = face(&verts, t);
            let at_vertex = [shade(normals[0]), shade(normals[1]), shade(normals[2])];
            drawn += match mode {
                // one value per triangle, from the face normal
                "flat" => draw_triangle(&mut colour, &mut depth, corners, &|_| shade(f)),
                // the lighting at the corners, interpolated
                "gouraud" => draw_triangle(&mut colour, &mut depth, corners, &|b| (b[0] as f32) * at_vertex[0] + (b[1] as f32) * at_vertex[1] + (b[2] as f32) * at_vertex[2]),
                // the normal interpolated, the lighting at every pixel
                _ => draw_triangle(&mut colour, &mut depth, corners, &|b| {
                    let n = [0, 1, 2].map(|k| b[0] * normals[0][k] + b[1] * normals[1][k] + b[2] * normals[2][k]);
                    shade(n)
                }),
            };
        }
        println!("  {mode:7}: {drawn} pixels drawn, brightest {:.3}", colour.max());
        save_gray(&flip(&colour), format!("{dir}/sphere-{mode}.png"));
    }

    // 2. A checkerboard floor seen in perspective: a square from 2 to 8 units in front of the
    //    camera, drawn as two triangles, with texture coordinates interpolated affinely on the
    //    screen and perspective-correctly.
    let (w, h) = (240usize, 160usize);
    let f = 200.0;
    let corners3d: [Vec3; 4] = [[-1.5, -1.0, -2.0], [1.5, -1.0, -2.0], [1.5, -1.0, -8.0], [-1.5, -1.0, -8.0]];
    let uv = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
    let project = |p: Vec3| [w as f64 / 2.0 + f * p[0] / -p[2], h as f64 * 0.8 + f * p[1] / -p[2], -p[2]];
    let checker = |u: f64, v: f64| if ((u * 8.0).floor() as i64 + (v * 8.0).floor() as i64) % 2 == 0 { 0.9f32 } else { 0.2 };
    for (name, correct) in [("affine", false), ("correct", true)] {
        let mut colour = Image::new(w, h, 0.0f32);
        let mut depth = Image::new(w, h, f32::INFINITY);
        for t in [[0, 1, 2], [0, 2, 3]] {
            let screen = [project(corners3d[t[0]]), project(corners3d[t[1]]), project(corners3d[t[2]])];
            let ws = [screen[0][2], screen[1][2], screen[2][2]];
            let us = [uv[t[0]][0], uv[t[1]][0], uv[t[2]][0]];
            let vs = [uv[t[0]][1], uv[t[1]][1], uv[t[2]][1]];
            draw_triangle(&mut colour, &mut depth, screen, &|b| {
                if correct {
                    checker(perspective_correct(b, ws, us), perspective_correct(b, ws, vs))
                } else {
                    checker(b[0] * us[0] + b[1] * us[1] + b[2] * us[2], b[0] * vs[0] + b[1] * vs[1] + b[2] * vs[2])
                }
            });
        }
        save_gray(&flip(&colour), format!("{dir}/floor-{name}.png"));
    }
    println!("wrote the pictures to {dir}/");
}
