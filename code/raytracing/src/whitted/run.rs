// Whitted ray tracing of a small scene: a checkered floor, a matte, a mirror and a glass
// ball. Pictures in out/whitted/; counts of rays and shadow pixels on the terminal.
//   cargo run -p raytracing -- whitted                      (your code)
//   cargo run -p raytracing --features solution -- whitted  (the model solution)

use imgcourse::*;
use crate::whitted::*;

const W: usize = 320;
const H: usize = 240;
const EYE: Vec3 = v3(0.0, 0.3, 0.0);

fn the_scene() -> Scene {
    let ball = |x: f32, z: f32, r: f32, material: Material, colour: [f32; 3]| {
        Object::new(Shape::Sphere { centre: v3(x, -1.0 + r, z), radius: r }, material, colour)
    };
    let mut floor = Object::new(Shape::Plane { point: v3(0.0, -1.0, 0.0), normal: v3(0.0, 1.0, 0.0) }, Material::Diffuse, [0.9, 0.9, 0.9]);
    floor.checker = true;
    Scene {
        objects: vec![
            floor,
            ball(-1.6, -6.0, 0.7, Material::Diffuse, [0.9, 0.25, 0.2]),
            ball(1.5, -7.5, 1.0, Material::Mirror(0.9), [0.0; 3]),
            ball(0.0, -4.5, 0.75, Material::Glass(1.5), [0.0; 3]),
            ball(0.6, -10.0, 0.8, Material::Diffuse, [0.2, 0.4, 0.9]),
        ],
        light: v3(-4.0, 6.0, -1.0),
        sky: [0.6, 0.75, 0.95],
        ambient: 0.15,
    }
}

/// Render the scene; returns the picture and the number of rays traced.
fn render(scene: &Scene, max_depth: usize, eps: f32) -> (Image<[f32; 3]>, usize) {
    let mut rays = 0;
    let mut img = Image::new(W, H, [0.0f32; 3]);
    for (i, j) in img.coords() {
        let c = trace(camera_ray(EYE, i, j, W, H, 50.0), scene, 0, max_depth, eps, &mut rays);
        img[(i, j)] = c.map(|v| v.min(1.0));
    }
    (img, rays)
}

/// For every pixel that sees a matte surface directly (only the floor, object 0, if
/// `floor_only`): how many are in shadow?
fn matte_shadows(scene: &Scene, eps: f32, floor_only: bool) -> (usize, usize) {
    let (mut matte, mut dark) = (0, 0);
    for j in 0..H {
        for i in 0..W {
            let ray = camera_ray(EYE, i, j, W, H, 50.0);
            if let Some(hit) = scene.nearest_hit(ray) {
                if matches!(scene.objects[hit.object].material, Material::Diffuse) && (hit.object == 0 || !floor_only) {
                    matte += 1;
                    if in_shadow(ray.at(hit.t), hit.normal, scene.light, scene, eps) {
                        dark += 1;
                    }
                }
            }
        }
    }
    (matte, dark)
}

pub fn main() {
    let dir = "out/whitted";
    let scene = the_scene();
    for (name, depth, eps) in [("whitted", 5, 1e-4), ("depth0", 0, 1e-4), ("depth1", 1, 1e-4), ("acne", 5, 0.0)] {
        let (img, rays) = render(&scene, depth, eps);
        save_rgb(&img, format!("{dir}/{name}.png"));
        println!("{name}: max depth {depth}, eps {eps}: {rays} rays, {:.2} per pixel", rays as f32 / (W * H) as f32);
    }
    for eps in [0.0, 1e-4] {
        let (matte, dark) = matte_shadows(&scene, eps, false);
        println!("eps = {eps}: {dark} of {matte} matte pixels in shadow");
    }

    // Light leaks: a small ball whose top is 0.12 above the floor, and a far-off light.
    let mut low = scene.clone();
    low.objects.truncate(1);
    low.objects.push(Object::new(Shape::Sphere { centre: v3(0.0, -0.93, -4.0), radius: 0.05 }, Material::Diffuse, [0.9, 0.8, 0.2]));
    low.light = v3(-2.0, 4.0, -2.0);
    for eps in [1e-4, 0.1, 0.2] {
        let (_, dark) = matte_shadows(&low, eps, true);
        let (img, _) = render(&low, 1, eps);
        save_rgb(&img, format!("{dir}/leak-eps{eps}.png"));
        println!("small ball 0.12 high, eps = {eps}: {dark} floor pixels in shadow");
    }
    println!("wrote the pictures to {dir}/");
}
