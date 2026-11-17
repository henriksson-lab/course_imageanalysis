// Acceptor photobleaching on simulated cells: E from the donor before and after, and the
// corrections; pictures in out/acceptor-bleach/.
//   cargo run -p day45 -- acceptor-bleach                      (your code)
//   cargo run -p day45 --features solution -- acceptor-bleach  (the model solution)

use crate::acceptor_bleach::sim::{self, Bleach};
use crate::acceptor_bleach::*;
use imgcourse::*;

/// The mean of the pixel values of `img` in the mask.
fn mean_in(img: &Image<f32>, mask: &Image<bool>) -> f32 {
    let v: Vec<f32> = img.coords().filter(|&p| mask[p]).map(|p| img[p]).collect();
    v.iter().sum::<f32>() / v.len() as f32
}

pub fn main() {
    let dir = "out/acceptor-bleach";
    let base = Bleach::default();
    let scenarios = [
        ("ideal", base.clone()),
        ("stop early", Bleach { pulses: 4, ..base.clone() }),
        ("donor bleaching", Bleach { donor_loss: 0.015, ..base.clone() }),
        ("the cells moved", Bleach { drift: (3, -2), ..base.clone() }),
        ("few photons", Bleach { photons: 40.0, ..base.clone() }),
        ("half paired", Bleach { f: 0.5, ..base.clone() }),
    ];
    println!("true E = {} (all donors paired unless stated)", base.e);
    for (name, p) in &scenarios {
        let ex = sim::experiment(p);
        // the region: cell pixels in the box; the control: cell pixels outside the box
        let region = ex.bleached.zip_map(&ex.cells, |b, c| b && c);
        let control = ex.bleached.zip_map(&ex.cells, |b, c| !b && c);
        let mut post = ex.donor_post.clone();
        let mut acc_post = ex.acceptor_post.clone();
        let (dx, dy) = find_shift(&ex.donor_pre, &post, 5);
        if (dx, dy) != (0, 0) {
            post = shift(&post, dx, dy);
            acc_post = shift(&acc_post, dx, dy);
        }
        let e_raw = region_efficiency(&ex.donor_pre, &ex.donor_post, &region);
        let ratio = control_ratio(&ex.donor_pre, &post, &control);
        let e_aligned = region_efficiency(&ex.donor_pre, &post, &region);
        let e_donor = region_efficiency(&ex.donor_pre, &post.map(|v| v / ratio), &region);
        // acceptor left in the box, measured from the acceptor images
        let left = control_ratio(&ex.acceptor_pre, &acc_post, &region); // Σ after / Σ before in the box
        let e_full = correct_incomplete(e_donor, 1.0 - left);
        let map = efficiency_map(&ex.donor_pre, &post);
        println!("\n{name}: pulses {}, true acceptor left {:.1} %, photons {}, f = {}", p.pulses, 100.0 * ex.acceptor_left, p.photons, p.f);
        println!("  shift found ({dx}, {dy}); E raw {e_raw:.3}, after alignment {e_aligned:.3}");
        println!("  control ratio {ratio:.3}; E after donor-bleach correction {e_donor:.3}");
        println!("  acceptor left (measured) {:.1} %; E after incomplete-bleach correction {e_full:.3}", 100.0 * left);
        println!("  mean of the pixel E in the box {:.3}, outside {:.3}", mean_in(&map, &region), mean_in(&map, &control));
        let file = name.replace(' ', "-");
        save_gray_signed(&efficiency_map(&ex.donor_pre, &ex.donor_post).map_xy(|x, y, v| if ex.cells[(x, y)] { v.clamp(-0.6, 0.6) } else { 0.0 }), format!("{dir}/{file}-e-raw.png"));
        if name == &"ideal" {
            save_gray_stretched(&ex.donor_pre, format!("{dir}/donor-before.png"));
            save_gray_stretched(&ex.donor_post, format!("{dir}/donor-after.png"));
            save_gray_stretched(&acc_post, format!("{dir}/acceptor-after.png"));
        }
    }
    println!("\nwrote the pictures to {dir}/ (E maps: 0 = grey, +0.6 white, −0.6 black)");
}
