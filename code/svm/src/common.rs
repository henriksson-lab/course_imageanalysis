//! Helpers shared by the programs (`run.rs`) of the units of day 35: the class colours and a
//! picture of a 2D feature space with a decision function.

use imgcourse::Image;

/// Colours for class −1 (blue) and class +1 (red).
pub const BLUE: [f32; 3] = [0.25, 0.45, 0.85];
pub const RED: [f32; 3] = [0.85, 0.3, 0.25];

/// A picture of the 2D feature space `lo..hi` (x1 to the right, x2 up), `size × size` pixels:
/// the background tinted by the sign of `f`, with `street` the band `−1 ≤ f ≤ 1` in grey, the
/// line `f = 0` in black (where `f` changes sign within one pixel), the points as dots in the
/// colour of their label, and the points marked in `support` (if not empty) with a black ring.
#[allow(clippy::too_many_arguments)]
pub fn plot<P: AsRef<[f32]>>(
    points: &[P],
    labels: &[f32],
    support: &[bool],
    street: bool,
    f: impl Fn(&[f32]) -> f32,
    lo: [f32; 2],
    hi: [f32; 2],
    size: usize,
) -> Image<[f32; 3]> {
    let scale = [(hi[0] - lo[0]) / size as f32, (hi[1] - lo[1]) / size as f32];
    let values = Image::from_fn(size + 1, size + 1, |px, py| f(&[lo[0] + (px as f32 + 0.5) * scale[0], hi[1] - (py as f32 + 0.5) * scale[1]]));
    let mut img = Image::from_fn(size, size, |px, py| {
        let v = values[(px, py)];
        if v * values[(px + 1, py)] <= 0.0 || v * values[(px, py + 1)] <= 0.0 {
            [0.0; 3]
        } else if street && v.abs() < 1.0 {
            [0.82; 3]
        } else {
            (if v > 0.0 { RED } else { BLUE }).map(|c| 0.8 + 0.2 * c)
        }
    });
    for (i, p) in points.iter().enumerate() {
        let p = p.as_ref();
        let (cx, cy) = ((p[0] - lo[0]) / scale[0], (hi[1] - p[1]) / scale[1]);
        let ring = support.get(i).copied().unwrap_or(false);
        for dy in -6..=6isize {
            for dx in -6..=6isize {
                let (x, y) = (cx as isize + dx, cy as isize + dy);
                let r2 = dx * dx + dy * dy;
                if !img.contains(x, y) {
                    continue;
                }
                if r2 <= 9 {
                    img[(x as usize, y as usize)] = if r2 >= 7 { [0.0; 3] } else if labels[i] > 0.0 { RED } else { BLUE };
                } else if ring && (25..=36).contains(&r2) {
                    img[(x as usize, y as usize)] = [0.0; 3];
                }
            }
        }
    }
    img
}
