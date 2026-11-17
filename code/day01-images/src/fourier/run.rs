// Runs your Fourier transforms on the course micrograph; results in out/fourier/.
//   cargo run -p day01 -- fourier                      (your code)
//   cargo run -p day01 --features solution -- fourier  (the model solution)

use crate::fourier::*;
use imgcourse::*;

/// The spectrum as a picture: log of the magnitude, frequency 0 moved to the centre.
fn spectrum_picture(spectrum: &Image<Complex>) -> Image<f32> {
    let (w, h) = (spectrum.width(), spectrum.height());
    Image::from_fn(w, h, |x, y| (1.0 + spectrum[((x + w / 2) % w, (y + h / 2) % h)].abs()).ln())
}

pub fn main() {
    let dir = "out/fourier";
    let img = samples::cell();
    save_gray(&img, format!("{dir}/input.png"));
    save_gray_stretched(&spectrum_picture(&dft2(&img.map(Complex::from), false)), format!("{dir}/spectrum.png"));

    for radius in [5.0, 30.0, 90.0] {
        save_gray(&lowpass(&img, radius), format!("{dir}/lowpass-{radius}.png"));
    }

    // A stripe artefact: two bright dots in the spectrum.
    let striped = img.map_xy(|x, y, v| v + 0.15 * (2.0 * std::f32::consts::PI * (16.0 * x as f32 + 10.0 * y as f32) / 256.0).cos());
    save_gray(&striped, format!("{dir}/striped.png"));
    save_gray_stretched(&spectrum_picture(&dft2(&striped.map(Complex::from), false)), format!("{dir}/striped-spectrum.png"));

    // Frequency responses of the two kernels of exercise B3, on signals of length 8.
    for (name, w) in [("blur (1 2 1)/4", vec![0.25, 0.5, 0.25]), ("second difference (1 -2 1)", vec![1.0, -2.0, 1.0])] {
        let r: Vec<String> = frequency_response(&w, 8).iter().map(|c| format!("{c:.2}")).collect();
        println!("{name}: {}", r.join("  "));
    }
    println!("wrote the results to {dir}/");
}
