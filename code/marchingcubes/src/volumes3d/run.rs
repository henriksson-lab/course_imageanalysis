// Slices, an oblique plane, a MIP and object counts of a synthetic stack of nuclei, with your
// code; pictures in out/volumes3d/.
//   cargo run -p marchingcubes -- volumes3d                      (your code)
//   cargo run -p marchingcubes --features solution -- volumes3d  (the model solution)

use imgcourse::*;
use crate::volumes3d::*;

/// An ellipsoid in world coordinates (µm, relative to the origin): centre, semi-axes, and a
/// tilt in the x–z plane (degrees).
struct Nucleus {
    centre: [f32; 3],
    axes: [f32; 3],
    tilt: f32,
}

/// The stack of the demo: 64 × 64 × 20 voxels of 0.2 × 0.2 × 0.5 µm, origin (10, 20, 4) µm.
/// Nuclei A and B overlap in x–y at different depths, C is long and tilted by 35° in the x–z
/// plane, D is small. Values: background 10/255, nuclei 200/255, a little blur and noise.
fn nuclei_stack(grid: &Grid) -> Volume<f32> {
    let nuclei = [
        Nucleus { centre: [4.6, 6.2, 2.8], axes: [1.6, 1.6, 1.2], tilt: 0.0 },  // A
        Nucleus { centre: [5.4, 6.2, 7.2], axes: [1.7, 1.4, 1.2], tilt: 0.0 },  // B
        Nucleus { centre: [9.0, 3.4, 5.0], axes: [3.6, 1.2, 1.2], tilt: 35.0 }, // C
        Nucleus { centre: [10.0, 10.0, 5.0], axes: [0.8, 0.8, 0.8], tilt: 0.0 }, // D
    ];
    let inside = |p: [f32; 3]| {
        nuclei.iter().any(|n| {
            let (c, s) = (n.tilt.to_radians().cos(), n.tilt.to_radians().sin());
            let d = [p[0] - n.centre[0], p[1] - n.centre[1], p[2] - n.centre[2]];
            let (a, b) = (c * d[0] + s * d[2], -s * d[0] + c * d[2]); // along and across the long axis (c, 0, s)
            (a / n.axes[0]).powi(2) + (d[1] / n.axes[1]).powi(2) + (b / n.axes[2]).powi(2) <= 1.0
        })
    };
    let mut rng = Rng::new(4);
    Volume::from_fn(64, 64, 20, |i, j, k| {
        // the fraction of the voxel inside (4 × 4 × 4 samples), as an optical blur would mix it
        let mut hits = 0;
        for a in 0..4 {
            for b in 0..4 {
                for c in 0..4 {
                    let v = [i as f32 + (a as f32 - 1.5) / 4.0, j as f32 + (b as f32 - 1.5) / 4.0, k as f32 + (c as f32 - 1.5) / 4.0];
                    let w = voxel_to_world(&Grid { spacing: grid.spacing, origin: [0.0; 3] }, v);
                    hits += inside(w) as usize;
                }
            }
        }
        (10.0 + 190.0 * hits as f32 / 64.0 + 4.0 * rng.normal()) / 255.0
    })
}

pub fn main() {
    let dir = "out/volumes3d";
    std::fs::create_dir_all(dir).unwrap();
    let grid = Grid { spacing: [0.2, 0.2, 0.5], origin: [10.0, 20.0, 4.0] };
    let vol = nuclei_stack(&grid);
    let w = voxel_to_world(&grid, [23.0, 30.0, 6.0]);
    println!("voxel (23, 30, 6): value {:.0}, world position ({:.1}, {:.1}, {:.1}) µm", 255.0 * vol[(23, 30, 6)], w[0], w[1], w[2]);

    // Orthogonal slices. The xz slice is drawn with its physical aspect: pixels of 0.2 µm.
    save_gray(&vol.slice(6), format!("{dir}/slice-xy-k6.png"));
    let c = voxel_to_world(&grid, [31.5, 31.0, 9.5]); // centre of the box, in row j = 31
    let (ex, ey, ez) = ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]);
    save_gray(&oblique_slice(&vol, &grid, c, ex, ez, 0.2, 64, 48), format!("{dir}/slice-xz-j31.png"));
    // The same slice when the header wrongly says Δz = 0.2 µm: the stack is squashed.
    let squashed = Grid { spacing: [0.2, 0.2, 0.2], origin: grid.origin };
    let cs = voxel_to_world(&squashed, [31.5, 31.0, 9.5]);
    save_gray(&oblique_slice(&vol, &squashed, cs, ex, ez, 0.2, 64, 20), format!("{dir}/slice-xz-j31-squashed.png"));

    // An oblique plane through C's centre, tilted by 35° in the x–z plane, along the cell.
    let t = 35f32.to_radians();
    let u = [t.cos(), 0.0, t.sin()];
    let c_centre = [grid.origin[0] + 9.0, grid.origin[1] + 3.4, grid.origin[2] + 5.0];
    let oblique = oblique_slice(&vol, &grid, c_centre, u, ey, 0.2, 61, 41);
    save_gray(&oblique, format!("{dir}/oblique-C.png"));
    let length = (0..61).filter(|&p| oblique[(p, 20)] > 100.0 / 255.0).count();
    let flat = oblique_slice(&vol, &grid, c_centre, ex, ey, 0.2, 61, 41);
    save_gray(&flat, format!("{dir}/flat-C.png"));
    let flat_length = (0..61).filter(|&p| flat[(p, 20)] > 100.0 / 255.0).count();
    println!("cell C along the tilted plane: {length} pixels of 0.2 µm above 100 ({:.1} µm); in the xy plane: {flat_length} ({:.1} µm)", 0.2 * length as f32, 0.2 * flat_length as f32);

    // MIP, and counting objects above 100 in the MIP and in 3D.
    let mip = mip_z(&vol);
    save_gray(&mip, format!("{dir}/mip-z.png"));
    let mask = vol.map(|v| v > 100.0 / 255.0);
    let mip_mask = Volume::from_fn(64, 64, 1, |x, y, _| mip[(x, y)] > 100.0 / 255.0);
    let (_, n_mip) = label_3d(&mip_mask, 26);
    let (labels, n6) = label_3d(&mask, 6);
    let (_, n26) = label_3d(&mask, 26);
    println!("objects above 100: MIP {n_mip} (8-connected); 3D {n6} (6-connected), {n26} (26-connected)");
    let voxel_volume = grid.spacing[0] * grid.spacing[1] * grid.spacing[2];
    for l in 1..=n6 {
        let count = labels.voxels().iter().filter(|&&v| v == l).count();
        println!("  object {l}: {count} voxels = {:.1} µm³", count as f32 * voxel_volume);
    }
    println!("wrote the pictures to {dir}/");
}
