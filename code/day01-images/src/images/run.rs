// Runs your transforms on the course micrograph and the letter F; results in out/images/.
//   cargo run -p day01 -- images                      (your code)
//   cargo run -p day01 --features solution -- images  (the model solution)

use crate::images::*;
use imgcourse::*;

pub fn main() {
    let dir = "out/images";
    let img = samples::cell();
    let (cx, cy) = ((img.width() - 1) as f32 / 2.0, (img.height() - 1) as f32 / 2.0);
    save_gray(&img, format!("{dir}/input.png"));

    // Rotate by 30° about the centre of the image, with both kinds of lookup.
    let turn = rotate_about(cx, cy, 30f32.to_radians());
    save_gray(&warp(&img, &turn, nearest), format!("{dir}/rotate30-nearest.png"));
    save_gray(&warp(&img, &turn, bilinear), format!("{dir}/rotate30-bilinear.png"));

    // Enlarge a 32 × 32 piece 8 times: here the two kinds of lookup differ most.
    let piece = img.crop(112, 112, 32, 32);
    let zoom = Image::new(256, 256, 0.0f32).map_xy(|x, y, _| (x as f32 / 8.0, y as f32 / 8.0));
    save_gray(&zoom.map(|(x, y)| nearest(&piece, x, y)), format!("{dir}/zoom-nearest.png"));
    save_gray(&zoom.map(|(x, y)| bilinear(&piece, x, y)), format!("{dir}/zoom-bilinear.png"));

    // The letter F: a mirror image, a shear, and a stretch followed by a quarter turn.
    let f = samples::letter_f(128);
    let c = 63.5;
    let about_centre = |a: Mat3| mat3_mul(&translation(c, c), &mat3_mul(&a, &translation(-c, -c)));
    let mirror = [[-1., 0., 0.], [0., 1., 0.], [0., 0., 1.]];
    let shear = [[1., 0.5, 0.], [0., 1., 0.], [0., 0., 1.]];
    let stretch = [[1.5, 0., 0.], [0., 1., 0.], [0., 0., 1.]];
    save_gray(&f, format!("{dir}/f.png"));
    save_gray(&warp(&f, &about_centre(mirror), nearest), format!("{dir}/f-mirror.png"));
    save_gray(&warp(&f, &about_centre(shear), bilinear), format!("{dir}/f-shear.png"));
    let stretch_then_turn = mat3_mul(&rotation(90f32.to_radians()), &stretch);
    save_gray(&warp(&f, &about_centre(stretch_then_turn), bilinear), format!("{dir}/f-stretch-then-turn.png"));
    println!("wrote the results to {dir}/");
}
