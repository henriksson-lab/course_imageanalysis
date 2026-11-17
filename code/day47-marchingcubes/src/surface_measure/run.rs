// Measures surfaces with your code: voxel faces against the mesh for balls of growing radius, grey
// images against binary masks, volume against c, the voxel spacing, and topology.
//   cargo run -p day47 -- surface-measure                      (your code)
//   cargo run -p day47 --features solution -- surface-measure  (the model solution)

use imgcourse::*;
use std::f32::consts::PI;
use crate::common::blur3;
use crate::surface_measure::*;

/// A ball of radius `r` voxels in a cube of n³ voxels, as a camera would record it: every voxel
/// holds the fraction of it that lies inside (4 × 4 × 4 samples), scaled along z by `stretch`
/// (an ellipsoid with semi-axes r, r, stretch · r).
fn ball(n: usize, r: f32, stretch: f32) -> Volume<f32> {
    let c = (n as f32 - 1.0) / 2.0;
    Volume::from_fn(n, n, n, |x, y, z| {
        let mut hits = 0;
        for s in 0..64 {
            let p = [x as f32 + ((s & 3) as f32 - 1.5) / 4.0, y as f32 + (((s >> 2) & 3) as f32 - 1.5) / 4.0, z as f32 + ((s >> 4) as f32 - 1.5) / 4.0];
            let d2 = (p[0] - c).powi(2) + (p[1] - c).powi(2) + ((p[2] - c) / stretch).powi(2);
            hits += (d2 <= r * r) as usize;
        }
        hits as f32 / 64.0
    })
}

fn pct(got: f32, truth: f32) -> String {
    format!("{:+6.1} %", 100.0 * (got / truth - 1.0))
}

pub fn main() {
    let one = [1.0, 1.0, 1.0];

    // 1. Voxel faces against the mesh, for balls of growing radius (errors against 4πR², 4/3 πR³).
    println!("ball of radius R: errors of the area and the volume");
    println!("   R    voxel faces   mesh (grey)   mesh (mask)   |   voxel count   mesh (grey)");
    for r in [4.0, 6.0, 8.0, 12.0, 16.0] {
        let n = (2.0 * r) as usize + 6;
        let grey = ball(n, r, 1.0);
        let mask_vol = grey.map(|v| v > 0.5);
        let mask = mask_vol.map(|b| if b { 1.0 } else { 0.0 });
        let (vox_v, vox_a) = voxel_estimates(&mask_vol, one);
        let (m_grey, m_mask) = (surface(&grey, one, 0.5), surface(&mask, one, 0.5));
        let (true_a, true_v) = (4.0 * PI * r * r, 4.0 / 3.0 * PI * r.powi(3));
        println!("{r:4.0}    {}      {}      {}      |   {}      {}",
            pct(vox_a, true_a), pct(mesh_area(&m_grey), true_a), pct(mesh_area(&m_mask), true_a), pct(vox_v, true_v), pct(mesh_volume(&m_grey), true_v));
    }

    // 2. Volume against c: a binary ball of radius 8, blurred.
    println!("\nbinary ball of radius 8 (true volume {:.1}), blurred: volume at c = 0.45, 0.5, 0.55", 4.0 / 3.0 * PI * 512.0);
    let binary = ball(26, 8.0, 1.0).map(|v| if v > 0.5 { 1.0 } else { 0.0 });
    for sigma in [0.5, 1.5, 2.5] {
        let b = blur3(&binary, sigma);
        let v: Vec<f32> = [0.45, 0.5, 0.55].iter().map(|&c| mesh_volume(&surface(&b, one, c))).collect();
        println!("  σ = {sigma}:  {:7.1}  {:7.1}  {:7.1}   (c ± 0.05 changes it by ±{:.1} %)", v[0], v[1], v[2], 100.0 * (v[0] - v[2]) / (2.0 * v[1]));
    }

    // 3. Spacing: an ellipsoid with semi-axes 10, 10, 4 voxels, imaged with voxels of 1 × 1 × 2.5
    //    (so 10 µm in every direction: a ball of radius 10).
    let stack = Volume::from_fn(26, 26, 13, {
        let fine = ball(26, 10.0, 0.4);
        move |x, y, z| fine[(x, y, (z + 7).min(25))]
    });
    let (right, wrong) = (surface(&stack, [1.0, 1.0, 2.5], 0.5), surface(&stack, one, 0.5));
    println!("\nball of radius 10 imaged with Δz = 2.5 (true area {:.0}, volume {:.0})", 4.0 * PI * 100.0, 4.0 / 3.0 * PI * 1000.0);
    println!("  with the spacing:    area {:7.1}  volume {:7.1}", mesh_area(&right), mesh_volume(&right));
    println!("  ignoring it:         area {:7.1}  volume {:7.1}  (volume ratio {:.3})", mesh_area(&wrong), mesh_volume(&wrong), mesh_volume(&wrong) / mesh_volume(&right));

    // 4. Topology.
    println!("\ntopology: V, E, F, boundary edges, χ");
    let two = Volume::from_fn(34, 18, 18, |x, y, z| {
        let d = |cx: f32| ((x as f32 - cx).powi(2) + (y as f32 - 8.5).powi(2) + (z as f32 - 8.5).powi(2)).sqrt();
        (6.5 - d(8.5).min(d(24.5))).clamp(0.0, 1.0)
    });
    let torus = Volume::from_fn(34, 34, 14, |x, y, z| {
        let ring = (((x as f32 - 16.5).powi(2) + (y as f32 - 16.5).powi(2)).sqrt() - 11.0).hypot(z as f32 - 6.5);
        (4.5 - ring).clamp(0.0, 1.0)
    });
    let one_ball = ball(18, 6.0, 1.0);
    let cut = Volume::from_fn(18, 18, 13, |x, y, z| one_ball[(x, y, z + 5)]); // the top of the ball cut off
    for (name, vol) in [("one ball", one_ball.clone()), ("two balls", two), ("torus", torus), ("a ball cut open", cut)] {
        let (v, e, f, b, chi) = topology(&surface(&vol, one, 0.5));
        println!("  {name:16} {v:6} {e:6} {f:6} {b:5}   χ = {chi}");
    }
}
