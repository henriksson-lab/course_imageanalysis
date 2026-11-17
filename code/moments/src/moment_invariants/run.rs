// Hu's invariants of an L that is rotated, scaled and mirrored; results in out/moment-invariants/.
//   cargo run -p moments -- moment-invariants                      (your code)
//   cargo run -p moments --features solution -- moment-invariants  (the model solution)

use imgcourse::*;
use crate::moment_invariants::*;

/// Is the point (u, v) inside the L (arm length 1, arm width 0.3, corner at the origin)?
fn in_ell(u: f32, v: f32) -> bool {
    (0.0..0.3).contains(&u) && (0.0..1.0).contains(&v) || (0.0..0.7).contains(&u) && (0.0..0.3).contains(&v)
}

/// The L drawn on a 96 × 96 canvas: `size` pixels long, turned by `angle` degrees
/// (counter-clockwise on the screen), optionally mirrored top–bottom. With `grey`, each pixel
/// holds the fraction of it covered by the L (8 × 8 samples), otherwise 0 or 1 (its centre).
fn draw_ell(size: f32, angle: f32, mirror: bool, grey: bool) -> Image<f32> {
    let (s, c) = angle.to_radians().sin_cos();
    let n = if grey { 8 } else { 1 };
    Image::from_fn(96, 96, |x, y| {
        let mut hits = 0;
        for i in 0..n {
            for j in 0..n {
                // sample position relative to the canvas centre, y up
                let px = x as f32 + (i as f32 + 0.5) / n as f32 - 48.0;
                let mut py = 48.0 - (y as f32 + (j as f32 + 0.5) / n as f32);
                if mirror {
                    py = -py;
                }
                // turn back by the angle, then scale and move the L's centre to the origin
                let (u, v) = ((c * px + s * py) / size + 0.3, (-s * px + c * py) / size + 0.4);
                if in_ell(u, v) {
                    hits += 1;
                }
            }
        }
        hits as f32 / (n * n) as f32
    })
}

/// Hu's invariants on the log scale of the slide "In practice": -sign(phi) log10 |phi|.
fn logscale(phi: &[f64; 7]) -> String {
    phi.iter().map(|&p| format!("{:7.2}", -p.signum() * p.abs().log10())).collect::<Vec<_>>().join(" ")
}

fn change(a: &[f64; 7], b: &[f64; 7]) -> String {
    a.iter().zip(b).map(|(a, b)| format!("{:7.2}%", 100.0 * (b - a).abs() / a.abs())).collect::<Vec<_>>().join(" ")
}

pub fn main() {
    let dir = "out/moment-invariants";
    let reference = draw_ell(60.0, 0.0, false, false);
    let phi0 = hu(&reference);
    println!("Hu's invariants of the L (60 pixels), as -sign(phi) log10|phi|:");
    println!("{:>28} {}", "", logscale(&phi0));
    println!();
    println!("relative change of phi1 .. phi7 against the reference:");
    let mut strip = Image::new(7 * 98, 96, 0.5f32);
    let cases = [
        ("moved by (8, 6) pixels", None),
        ("scaled to 45 pixels", Some((45.0, 0.0, false, false))),
        ("scaled to 45, grey edges", Some((45.0, 0.0, false, true))),
        ("rotated 50°", Some((60.0, 50.0, false, false))),
        ("rotated 50°, grey edges", Some((60.0, 50.0, false, true))),
        ("mirrored", Some((60.0, 0.0, true, false))),
        ("small (18 px), rotated 45°", Some((18.0, 45.0, false, false))),
    ];
    for (k, (name, how)) in cases.iter().enumerate() {
        let (img, phi_ref) = match how {
            None => (Image::from_fn(96, 96, |x, y| if x >= 8 && y >= 6 { reference[(x - 8, y - 6)] } else { 0.0 }), phi0),
            // the small L is compared with a small L that is not rotated
            Some((size, angle, mirror, grey)) => (draw_ell(*size, *angle, *mirror, *grey), if *size < 20.0 { hu(&draw_ell(*size, 0.0, false, false)) } else { phi0 }),
        };
        let phi = hu(&img);
        println!("{name:>28} {}", change(&phi_ref, &phi));
        if *name == "mirrored" {
            println!("{:>28} phi7: {:.3e} before, {:.3e} after", "", phi_ref[6], phi[6]);
        }
        for (x, y) in img.coords() {
            strip[(k * 98 + x, y)] = img[(x, y)];
        }
    }
    save_gray(&strip, format!("{dir}/shapes.png"));
    println!("wrote {dir}/shapes.png (moved, scaled, scaled with grey edges, rotated, rotated with grey edges, mirrored, small)");
}
