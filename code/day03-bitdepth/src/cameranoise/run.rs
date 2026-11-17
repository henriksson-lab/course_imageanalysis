// Simulates a camera imaging the course micrograph, measures its noise on flat patches, and
// computes SNR and PSNR; results in out/cameranoise/.
//   cargo run -p day03 -- cameranoise                      (your code)
//   cargo run -p day03 --features solution -- cameranoise  (the model solution)

use crate::cameranoise::*;
use imgcourse::*;

pub fn main() {
    let dir = "out/cameranoise";
    let mut rng = Rng::new(42);
    let cell = samples::cell();

    // Shot noise: the micrograph as the mean count, scaled to 10, 100 and 10 000 electrons on
    // average, with an ideal camera (no read noise, a well too large to fill, many bits).
    let ideal = Camera { full_well: 1.0e6, read_noise: 0.0, bits: 24 };
    let cell_mean = cell.mean();
    for n in [10.0, 100.0, 10000.0] {
        let electrons = cell.map(|v| v / cell_mean * n);
        let shot = expose(&ideal, &electrons, &mut rng);
        // As a picture in [0, 1]: divide by the brightest mean count, which is then the peak.
        let peak = electrons.max();
        let picture = shot.map(|a| a as f32 * gain(&ideal) / peak);
        save_gray(&picture, format!("{dir}/cell-{n}-electrons.png"));
        let p = psnr(&picture, &electrons.map(|e| e / peak), 1.0);
        let diff: Vec<f32> = shot.pixels().iter().zip(electrons.pixels()).map(|(&a, e)| a as f32 * gain(&ideal) - e).collect();
        let rms = (diff.iter().map(|d| d * d).sum::<f32>() / diff.len() as f32).sqrt();
        println!("micrograph, {n:5} electrons per pixel on average: RMS noise {rms:.1}, SNR {:.1} ({:.1} dB), PSNR {p:.1} dB", n / rms, amplitude_db(n / rms));
    }

    // The photon-transfer curve: noise against signal on flat 64 × 64 patches.
    let scmos = Camera { full_well: 30000.0, read_noise: 1.5, bits: 16 };
    let scmos8 = Camera { bits: 8, ..scmos };
    for cam in [scmos, scmos8] {
        println!(
            "\nfull well {}, read noise {}, {} bits: gain {:.2} electrons/ADU, quantisation noise {:.2}, dynamic range {:.0} : 1 = {:.1} dB = {:.1} bits",
            cam.full_well,
            cam.read_noise,
            cam.bits,
            gain(&cam),
            gain(&cam) / 12f32.sqrt(),
            dynamic_range(&cam),
            amplitude_db(dynamic_range(&cam)),
            dynamic_range(&cam).log2()
        );
        println!("  signal (e)   measured noise (e)   predicted (e)   SNR (dB)");
        for s in [1.0, 10.0, 100.0, 1000.0, 10000.0, 30000.0] {
            let adu = expose(&cam, &Image::new(64, 64, s), &mut rng);
            let e: Vec<f32> = adu.pixels().iter().map(|&a| a as f32 * gain(&cam)).collect();
            let (m, sd) = mean_and_sd(&e);
            // A patch whose pixels all read the same number has no measurable noise (and no SNR).
            let snr = if sd > 0.01 { format!("{:8.1}", amplitude_db(m / sd)) } else { "       –".to_string() };
            println!("  {s:10}   {:18.2}   {:13.2}   {snr}", if sd > 0.01 { sd } else { 0.0 }, predicted_noise(&cam, s));
        }
    }

    // Low light: how many different ADU values does the picture use with 8 and with 12 bits?
    for bits in [8, 12] {
        let cam = Camera { full_well: 30000.0, read_noise: 1.5, bits };
        let electrons = cell.map(|v| 300.0 * v);
        let adu = expose(&cam, &electrons, &mut rng);
        let mut used: Vec<u32> = adu.pixels().to_vec();
        used.sort();
        used.dedup();
        let top = *used.last().unwrap();
        println!("\nlow light (at most 300 electrons), {bits} bits: {} different ADU values, brightest {top}", used.len());
        save_gray_stretched(&adu.map(|a| a as f32), format!("{dir}/lowlight-{bits}bit.png"));
    }

    // Averaging frames of a dim flat patch: 400 electrons, read noise 3.
    let cam = Camera { full_well: 15000.0, read_noise: 3.0, bits: 16 };
    println!();
    for frames in [1, 4, 7, 16] {
        let mut sum = Image::new(64, 64, 0.0f32);
        for _ in 0..frames {
            let adu = expose(&cam, &Image::new(64, 64, 400.0), &mut rng);
            sum = sum.zip_map(&adu, |s, a| s + a as f32 * gain(&cam));
        }
        let (m, sd) = mean_and_sd(sum.map(|v| v / frames as f32).pixels());
        println!("400 electrons, read noise 3, average of {frames:2} frames: SNR {:.1} ({:.1} dB)", m / sd, amplitude_db(m / sd));
    }
    println!("wrote the results to {dir}/");
}
