// A counting pipeline on a synthetic nuclei image; pictures in out/label-measure/.
//   cargo run -p labelling -- label-measure                      (your code)
//   cargo run -p labelling --features solution -- label-measure  (the model solution)

use imgcourse::*;
use crate::label_measure::*;

pub fn main() {
    let dir = "out/label-measure";
    // the middle of a larger synthetic image, so that nuclei are cut by the image border
    let (big, _) = samples::nuclei(320, 25, 7);
    let img = big.crop(32, 32, 256, 256);
    save_gray(&img, format!("{dir}/nuclei.png"));

    // 1. A low threshold: the nuclei, and specks of noise where the background is brightest.
    let (labels, n) = label_components(&img.map(|v| v > 0.24));
    save_labels(&labels, format!("{dir}/1-threshold.png"));
    let rows = table(&labels);
    let small = rows.iter().filter(|r| r.area < 30).count();
    println!("threshold 0.24: {n} objects, {small} of them smaller than 30 pixels");

    // 2. Two ways to remove the specks.
    let (opened, n_open) = area_opening(&labels, 30);
    save_labels(&opened, format!("{dir}/2-area-opening.png"));
    let (hyst, n_hyst) = hysteresis(&img, 0.24, 0.5);
    save_labels(&hyst, format!("{dir}/2-hysteresis.png"));
    let same = opened == hyst;
    println!("area opening (≥ 30 pixels): {n_open} objects; hysteresis (0.24, 0.5): {n_hyst} objects; same label image: {same}");

    // 3. Border rules on the cleaned label image.
    let (w, h) = (opened.width(), opened.height());
    let t = table(&opened);
    let inside = t.iter().filter(|r| !r.touches_border(w, h)).count();
    let margin = 20;
    let counted = counting_frame(&opened, margin);
    println!("keep all: {n_open}; remove those touching the border: {inside}; counting frame (margin {margin}): {}", counted.len());

    // The counting frame: counted objects in colour, the others grey; inclusion edges green,
    // exclusion line red.
    let mut keep = vec![false; n_open as usize + 1];
    for &l in &counted {
        keep[l as usize] = true;
    }
    let colours = io::label_colours(&opened);
    let mut picture = Image::from_fn(w, h, |x, y| {
        let l = opened[(x, y)] as usize;
        if l == 0 {
            [0.0; 3]
        } else if keep[l] {
            colours[(x, y)]
        } else {
            [0.45; 3]
        }
    });
    for i in 0..=(w - margin) {
        picture[(i, h - margin)] = [1.0, 0.0, 0.0]; // bottom edge, extended to the left
        picture[(w - margin, i)] = [1.0, 0.0, 0.0]; // right edge, extended to the top
    }
    for i in margin..(w - margin) {
        picture[(i, margin)] = [0.0, 1.0, 0.0]; // top inclusion edge
        picture[(margin, i)] = [0.0, 1.0, 0.0]; // left inclusion edge
    }
    save_rgb(&picture, format!("{dir}/3-counting-frame.png"));

    // Sizes: the mean area of the objects each rule keeps.
    let mean = |ls: &[u32]| ls.iter().map(|&l| t[l as usize - 1].area).sum::<usize>() as f32 / ls.len() as f32;
    let all: Vec<u32> = (1..=n_open).collect();
    let not_border: Vec<u32> = (1..=n_open).filter(|&l| !t[l as usize - 1].touches_border(w, h)).collect();
    let cut = counted.iter().filter(|&&l| t[l as usize - 1].touches_border(w, h)).count();
    println!("objects counted by the frame that touch the image border (and are cut): {cut}");
    println!("mean area: all {:.0}, not touching the border {:.0}, counting frame {:.0}", mean(&all), mean(&not_border), mean(&counted));
    println!("wrote the results to {dir}/");
}
