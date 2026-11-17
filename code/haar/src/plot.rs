// Drawing curves and points into a picture, for main.rs. Plumbing, not part of the exercises.
#![allow(dead_code)]

use imgcourse::{save_rgb, Image};

pub type Colour = [f32; 3];
pub const BLACK: Colour = [0.0, 0.0, 0.0];
pub const GREY: Colour = [0.6, 0.6, 0.6];
pub const RED: Colour = [0.85, 0.1, 0.1];
pub const BLUE: Colour = [0.1, 0.3, 0.85];
pub const GREEN: Colour = [0.1, 0.6, 0.2];
pub const ORANGE: Colour = [0.95, 0.55, 0.0];
pub const PURPLE: Colour = [0.55, 0.2, 0.7];

/// A white picture showing the region `x0..x1` × `y0..y1` of the plane, with `y` pointing up.
pub struct Plot {
    img: Image<Colour>,
    x0: f32,
    x1: f32,
    y0: f32,
    y1: f32,
}

impl Plot {
    pub fn new(width: usize, height: usize, (x0, x1): (f32, f32), (y0, y1): (f32, f32)) -> Plot {
        let mut p = Plot { img: Image::new(width, height, [1.0; 3]), x0, x1, y0, y1 };
        // Light grey axes through 0, if 0 is in view.
        let light = [0.85; 3];
        p.line([x0, 0.0], [x1, 0.0], light);
        p.line([0.0, y0], [0.0, y1], light);
        p
    }

    /// Picture position (column, row) of a point of the plane.
    fn to_pixel(&self, p: [f32; 2]) -> (f32, f32) {
        let w = self.img.width() as f32 - 1.0;
        let h = self.img.height() as f32 - 1.0;
        ((p[0] - self.x0) / (self.x1 - self.x0) * w, (self.y1 - p[1]) / (self.y1 - self.y0) * h)
    }

    fn put(&mut self, px: f32, py: f32, colour: Colour) {
        // A 2 × 2 pen, so that lines are easy to see.
        for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
            let (x, y) = (px.round() as isize + dx, py.round() as isize + dy);
            if self.img.contains(x, y) {
                self.img[(x as usize, y as usize)] = colour;
            }
        }
    }

    pub fn line(&mut self, a: [f32; 2], b: [f32; 2], colour: Colour) {
        let (ax, ay) = self.to_pixel(a);
        let (bx, by) = self.to_pixel(b);
        let steps = ((bx - ax).abs().max((by - ay).abs()) * 2.0).ceil().max(1.0) as usize;
        if steps > 100_000 {
            return; // far outside the picture
        }
        for s in 0..=steps {
            let t = s as f32 / steps as f32;
            self.put(ax + t * (bx - ax), ay + t * (by - ay), colour);
        }
    }

    /// A polyline through the points.
    pub fn curve(&mut self, points: &[[f32; 2]], colour: Colour) {
        for w in points.windows(2) {
            self.line(w[0], w[1], colour);
        }
    }

    /// A dashed polyline (every second piece drawn).
    pub fn dashed(&mut self, points: &[[f32; 2]], colour: Colour) {
        for (i, w) in points.windows(2).enumerate() {
            if (i / 3) % 2 == 0 {
                self.line(w[0], w[1], colour);
            }
        }
    }

    /// A filled dot of `radius` pixels.
    pub fn dot(&mut self, p: [f32; 2], radius: f32, colour: Colour) {
        let (cx, cy) = self.to_pixel(p);
        let r = radius.ceil() as isize;
        for dy in -r..=r {
            for dx in -r..=r {
                if ((dx * dx + dy * dy) as f32) <= radius * radius {
                    let (x, y) = (cx.round() as isize + dx, cy.round() as isize + dy);
                    if self.img.contains(x, y) {
                        self.img[(x as usize, y as usize)] = colour;
                    }
                }
            }
        }
    }

    pub fn save(&self, path: impl AsRef<std::path::Path>) {
        save_rgb(&self.img, path);
    }
}
