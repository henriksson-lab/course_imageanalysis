// Ray casts a small scene (two spheres, a triangle, a checkered ground) and writes the colour
// image, the depth, the normals and the object ids to out/raycast/.
//   cargo run -p raytracing -- raycast                      (your code)
//   cargo run -p raytracing --features solution -- raycast  (the model solution)

use imgcourse::*;
use crate::raycast::*;
use std::time::Instant;

pub fn main() {
    let dir = "out/raycast";
    let scene = [
        Shape::Plane { point: v3(0.0, -1.0, 0.0), normal: v3(0.0, 1.0, 0.0) },
        Shape::Sphere { centre: v3(-0.9, -0.2, -5.0), radius: 0.8 },
        Shape::Sphere { centre: v3(1.3, 0.0, -8.0), radius: 1.0 },
        Shape::Triangle { a: v3(-0.2, -1.0, -6.5), b: v3(0.9, -1.0, -6.5), c: v3(0.3, 0.9, -6.8) },
    ];
    let albedo = [[0.0, 0.0, 0.0], [0.85, 0.2, 0.2], [0.2, 0.35, 0.9], [0.95, 0.8, 0.2]];
    let light = v3(-1.0, 2.0, 1.5).normalize(); // direction towards a far light

    let (w, h) = (320, 240);
    let start = Instant::now();
    let mut colour = Image::new(w, h, [0.0f32; 3]);
    let mut depth = Image::new(w, h, 0.0f32);
    let mut normal = Image::new(w, h, [0.0f32; 3]);
    let mut object = Image::new(w, h, 0u32);
    for (i, j) in colour.coords() {
        let ray = camera_ray(i, j, w, h, 50.0);
        let Some(hit) = nearest_hit(ray, &scene) else {
            colour[(i, j)] = [0.55, 0.7, 0.9]; // sky
            continue;
        };
        let p = ray.at(hit.t);
        let mut a = albedo[hit.object];
        if hit.object == 0 {
            // The ground: a checkerboard of unit squares.
            let even = ((p.x.floor() + p.z.floor()) as i64).rem_euclid(2) == 0;
            a = if even { [0.8, 0.8, 0.8] } else { [0.3, 0.3, 0.3] };
        }
        let lambert = hit.normal.dot(light).max(0.0);
        colour[(i, j)] = a.map(|c| c * (0.15 + 0.85 * lambert));
        depth[(i, j)] = 1.0 / hit.t; // near = bright
        let n = hit.normal;
        normal[(i, j)] = [0.5 + 0.5 * n.x, 0.5 + 0.5 * n.y, 0.5 + 0.5 * n.z];
        object[(i, j)] = hit.object as u32 + 1;
    }
    let elapsed = start.elapsed();
    save_rgb(&colour, format!("{dir}/colour.png"));
    save_gray_stretched(&depth, format!("{dir}/depth.png"));
    save_rgb(&normal, format!("{dir}/normal.png"));
    save_labels(&object, format!("{dir}/object.png"));

    let rays = w * h;
    println!("{w} × {h} pixels, {} objects: {rays} rays, {} ray–object tests, {:.0} ms", scene.len(), rays * scene.len(), elapsed.as_secs_f64() * 1e3);
    // The count for a full HD image of a mesh of a million triangles (slide "Why it is slow").
    let tests = 1920.0 * 1080.0 * 1e6;
    let per_test = elapsed.as_secs_f64() / (rays * scene.len()) as f64;
    println!("1920 × 1080 pixels, 10^6 triangles: {tests:.2e} tests, about {:.1} hours at this speed", tests * per_test / 3600.0);
    println!("wrote the pictures to {dir}/");
}
