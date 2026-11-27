//! Helpers shared by the programs (`run.rs`) of the units of day 32: counting the pieces of a
//! region, drawing a zero level and putting pictures side by side.

use imgcourse::Image;

/// The number of separate pieces of the region `φ < c` (4-neighbours, flood fill).
pub fn pieces(phi: &Image<f32>, c: f32) -> usize {
    let mut seen = Image::new(phi.width(), phi.height(), false);
    let mut count = 0;
    for (x, y) in phi.coords() {
        if phi[(x, y)] >= c || seen[(x, y)] {
            continue;
        }
        count += 1;
        let mut stack = vec![(x, y)];
        seen[(x, y)] = true;
        while let Some((x, y)) = stack.pop() {
            for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                let (nx, ny) = (x as isize + dx, y as isize + dy);
                if phi.contains(nx, ny) && phi[(nx as usize, ny as usize)] < c && !seen[(nx as usize, ny as usize)] {
                    seen[(nx as usize, ny as usize)] = true;
                    stack.push((nx as usize, ny as usize));
                }
            }
        }
    }
    count
}

/// Mark the pixels just inside the zero level (φ < 0 with a 4-neighbour φ ≥ 0) in `colour`.
pub fn draw_contour(pic: &mut Image<[f32; 3]>, phi: &Image<f32>, colour: [f32; 3]) {
    for (x, y) in phi.coords() {
        if phi[(x, y)] < 0.0 {
            let edge = [(1, 0), (-1, 0), (0, 1), (0, -1)].iter().any(|&(dx, dy)| {
                let (nx, ny) = (x as isize + dx, y as isize + dy);
                phi.contains(nx, ny) && phi[(nx as usize, ny as usize)] >= 0.0
            });
            if edge {
                pic[(x, y)] = colour;
            }
        }
    }
}

/// Pictures of the same size enlarged twice, side by side, with 4-pixel gaps of grey `gap`.
pub fn sheet(pics: &[Image<[f32; 3]>], gap: f32) -> Image<[f32; 3]> {
    let (size, height) = (pics[0].width(), pics[0].height());
    let w = size + 4;
    Image::from_fn(2 * (pics.len() * w - 4), 2 * height, |x, y| {
        let (x, y) = (x / 2, y / 2);
        if x % w < size { pics[x / w][(x % w, y)] } else { [gap; 3] }
    })
}
