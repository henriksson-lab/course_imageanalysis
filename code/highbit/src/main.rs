// Shows a synthetic 12-bit fluorescence image and a synthetic CT slice with different
// windows, and subtracts a background in three kinds of arithmetic; results in out/highbit/.
//   cargo run -p highbit                      (your code)
//   cargo run -p highbit --features solution  (the model solution)

use highbit::*;
use imgcourse::*;

/// A synthetic 12-bit fluorescence image (values 0 … 4095, stored as u16): nuclei on a
/// background of about 160 with noise, and two small saturated spots.
fn fluorescence() -> (Image<u16>, Image<u32>) {
    let (img, labels) = samples::nuclei(256, 8, 4);
    let mut rng = Rng::new(9);
    let raw = Image::from_fn(256, 256, |x, y| {
        let mean = 160.0 + 2500.0 * (img[(x, y)] - 0.2).max(0.0);
        let spot = (x as f32 - 200.0).hypot(y as f32 - 60.0) < 3.0 || (x as f32 - 70.0).hypot(y as f32 - 180.0) < 3.0;
        if spot {
            5000.0
        } else {
            mean + mean.sqrt() * rng.normal()
        }
    });
    (raw.map(|v| v.round().clamp(0.0, 4095.0) as u16), labels)
}

/// Show raw values through a window, as a grey picture.
fn show(img: &Image<f32>, lo: f32, hi: f32) -> Image<f32> {
    img.map(|v| window(v, lo, hi) as f32 / 255.0)
}

fn main() {
    let dir = "out/highbit";
    let (fluo, labels) = fluorescence();
    let raw = fluo.map(|v| v as f32);
    let (min, max) = (raw.min(), raw.max());
    let lo = percentile(fluo.pixels(), 0.005);
    let hi = percentile(fluo.pixels(), 0.995);
    println!("12-bit image: values {min} … {max}; 0.5 % and 99.5 % percentiles {lo} and {hi}");
    save_gray(&show(&raw, 0.0, 65535.0), format!("{dir}/fluo-0-65535.png"));
    save_gray(&show(&raw, 0.0, 4095.0), format!("{dir}/fluo-0-4095.png"));
    save_gray(&show(&raw, lo as f32, hi as f32), format!("{dir}/fluo-percentiles.png"));
    println!("brightest pixel {max} shown as level {} with the window 0 … 65 535", window(max, 0.0, 65535.0));

    // MSB-aligned: the same values times 16.
    let distinct = |v: &mut Vec<u16>| {
        v.sort();
        v.dedup();
        v.len()
    };
    let lsb = distinct(&mut fluo.pixels().to_vec());
    let mut msb: Vec<u16> = fluo.pixels().iter().map(|&v| v * 16).collect();
    let n_msb = distinct(&mut msb);
    let comb = msb.iter().all(|v| v % 16 == 0);
    println!("different values: {lsb} LSB-aligned, {n_msb} MSB-aligned (from {} to {}); all MSB values multiples of 16: {comb}", msb[0], msb[n_msb - 1]);
    println!("brightest MSB-aligned pixel with the window 0 … 65 535: level {}", window(65520.0, 0.0, 65535.0));

    // Subtract a background of 160, three ways; show all with the window −60 … 600.
    let wrap = fluo.map(|v| sub_wrap(v, 160) as f32);
    let saturate = fluo.map(|v| sub_saturate(v, 160) as f32);
    let float = raw.map(|v| v - 160.0);
    let background: Vec<(usize, usize)> = labels.coords().filter(|&(x, y)| labels[(x, y)] == 0 && x > 8 && y > 8).collect();
    for (name, img) in [("wrap", &wrap), ("saturate", &saturate), ("float", &float)] {
        let m = background.iter().map(|&p| img[p]).sum::<f32>() / background.len() as f32;
        let high = img.pixels().iter().filter(|&&v| v > 4095.0).count() as f32 / img.pixels().len() as f32;
        println!("background − 160, {name:8}: background mean {m:9.1}; {:4.1} % of the pixels above 4095", 100.0 * high);
        save_gray(&show(img, -60.0, 600.0), format!("{dir}/subtract-{name}.png"));
    }

    // A synthetic CT slice in Hounsfield units: air, a body of soft tissue (40) with a fat
    // layer (−100), two lungs (−850) and a bone (700); a faint lesion (60) in the soft tissue.
    let ct = Image::from_fn(256, 256, |x, y| {
        let (u, v) = ((x as f32 - 128.0) / 110.0, (y as f32 - 128.0) / 80.0);
        let r = u.hypot(v);
        let lung = |cx: f32| ((x as f32 - cx) / 35.0).hypot((y as f32 - 120.0) / 50.0) < 1.0;
        if r > 1.0 {
            -1000.0
        } else if r > 0.93 {
            -100.0
        } else if lung(80.0) || lung(176.0) {
            -850.0
        } else if (x as f32 - 128.0).hypot(y as f32 - 185.0) < 14.0 {
            700.0
        } else if (x as f32 - 128.0).hypot(y as f32 - 95.0) < 8.0 {
            60.0
        } else {
            40.0
        }
    });
    for (name, lo, hi) in [("soft", -160.0, 240.0), ("lung", -1350.0, 150.0), ("bone", -500.0, 1300.0), ("narrow", -5.0, 145.0)] {
        save_gray(&show(&ct, lo, hi), format!("{dir}/ct-{name}.png"));
        println!("CT, window {lo} … {hi}: soft tissue (40 HU) level {}, lesion (60 HU) level {}", window(40.0, lo, hi), window(60.0, lo, hi));
    }
    // The same slice stored as HU + 1024, shown with the soft-tissue window by mistake.
    let stored = ct.map(|v| v + 1024.0);
    save_gray(&show(&stored, -160.0, 240.0), format!("{dir}/ct-stored-soft.png"));
    println!("stored as HU + 1024 with the soft-tissue window: air (stored 24) shows as level {}", window(24.0, -160.0, 240.0));
    println!("wrote the results to {dir}/");
}
