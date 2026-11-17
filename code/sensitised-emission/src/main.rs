// Sensitised emission on a simulated field of cells: coefficients from the controls,
// F_c and E_app per cell; pictures in out/sensitised-emission/.
//   cargo run -p sensitised-emission                      (your code)
//   cargo run -p sensitised-emission --features solution  (the model solution)

use imgcourse::*;
use sensitised_emission::sim::{self, Kind};
use sensitised_emission::*;

fn main() {
    let dir = "out/sensitised-emission";
    let photons = 300.0;
    let minus_offset = |img: &Image<f32>| img.map(|v| v - sim::OFFSET);

    // 1. The controls: a field of donor-only cells and one of acceptor-only cells.
    let donor = sim::field(&[Kind::DonorOnly], 10, photons, 1.0, 11);
    let d = slope_through_origin(&minus_offset(&donor.idd), &minus_offset(&donor.ida), &sim::cell_mask(&donor.labels));
    let acc = sim::field(&[Kind::AcceptorOnly], 10, photons, 1.0, 12);
    let a = slope_through_origin(&minus_offset(&acc.iaa), &minus_offset(&acc.ida), &sim::cell_mask(&acc.labels));
    println!("from the controls: d = {d:.3} (true {}), a = {a:.3} (true {})", sim::D_TRUE, sim::A_TRUE);

    // 2. The field: five kinds of cells.
    let kinds = [Kind::DonorOnly, Kind::AcceptorOnly, Kind::NoFret, Kind::Fusion, Kind::Binding];
    let run = |name: &str, acceptor_scale: f32, d_used: f32, save: bool| {
        let f = sim::field(&kinds, 15, photons, acceptor_scale, 5);
        let (idd, ida, iaa) = (minus_offset(&f.idd), minus_offset(&f.ida), minus_offset(&f.iaa));
        let fc = corrected_fret(&idd, &ida, &iaa, d_used, a);
        let cells = per_cell(&fc, &idd, &iaa, &f.labels, sim::G_TRUE);
        println!("\n{name} (d = {d_used:.3}):");
        println!("  cell  kind           area  F_c/pixel  true fE   E_app  F_c/I_AA  F_c/sqrt(I_DD I_AA)");
        for (l, v) in cells.iter().enumerate().skip(1) {
            if v.area < 30 {
                continue; // (almost) covered by other cells
            }
            let kind = f.kinds[l - 1];
            let (ff, e) = kind.f_and_e();
            let fc_mean: f32 = f.labels.coords().filter(|&p| f.labels[p] == l as u32).map(|p| fc[p]).sum::<f32>() / v.area as f32;
            // with only one dye the efficiency or the indices divide by (almost) nothing: not shown
            let show = |x: f32, ok: bool| if ok { format!("{x:.3}") } else { "-".to_string() };
            let (has_d, has_a) = (kind != Kind::AcceptorOnly, kind != Kind::DonorOnly);
            println!("  {l:>4}  {:<14} {:>4} {:>9.1} {:>8.2} {:>7} {:>9} {:>12}", kind.name(), v.area, fc_mean, ff * e,
                show(v.e_app, has_d), show(v.per_acceptor, has_a), show(v.symmetric, has_d && has_a));
        }
        if save {
            let mask = sim::cell_mask(&f.labels);
            let ea = e_app(&fc, &idd, sim::G_TRUE).map_xy(|x, y, v| if mask[(x, y)] { v } else { 0.0 });
            save_gray_stretched(&f.ida, format!("{dir}/fret-channel.png"));
            save_gray_signed(&fc, format!("{dir}/fc.png"));
            save_gray(&ea.map(|v| v / 0.4), format!("{dir}/e-app.png")); // black 0, white 0.4 and above
            save_labels(&f.labels, format!("{dir}/cells.png"));
            // the noise of one pixel and of a whole cell, for the first fusion cell
            let l = (1..cells.len()).find(|&l| f.kinds[l - 1] == Kind::Fusion && cells[l].area >= 30).unwrap();
            let pix: Vec<f32> = f.labels.coords().filter(|&p| f.labels[p] == l as u32).map(|p| ea[p]).collect();
            let m = pix.iter().sum::<f32>() / pix.len() as f32;
            let sd = (pix.iter().map(|v| (v - m).powi(2)).sum::<f32>() / pix.len() as f32).sqrt();
            println!("  cell {l} (fusion): pixel E_app {m:.3} ± {sd:.3} (sd over {} pixels), cell value {:.3}", pix.len(), cells[l].e_app);
        }
    };
    run("measured coefficients", 1.0, d, true);
    run("d too small", 1.0, 0.4, false);
    run("extra free acceptor in the FRET cells (3 ×)", 3.0, d, false);
    println!("\nwrote the pictures to {dir}/");
}
