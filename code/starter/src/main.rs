// A first program: read an image, change every pixel, save the result.
// Run it from the folder with Cargo.toml in it:   cargo run -p starter
// The result is written to out/starter/.


fn main() {
    // The course micrograph, built into the library. To read your own file instead:
    // let img = load_gray("my-image.png");
    let img = samples::cell();
    println!("the image is {} × {} pixels", img.width(), img.height());
    println!("darkest {:.3}, brightest {:.3}, mean {:.3}", img.min(), img.max(), img.mean());

    // A new image of the same size, filled in pixel by pixel.
    let mut out = Image::new(img.width(), img.height(), 0.0);
    for (x, y) in img.coords() {
        out[(x, y)] = 1.0 - img[(x, y)]; // the negative
    }
    save_gray(&out, "out/starter/negative.png");

    // The same with map: one function applied to every pixel value.
    let brighter = img.map(|v| (1.5 * v).min(1.0));
    save_gray(&brighter, "out/starter/brighter.png");

    // A small image written out by hand, and printed with two decimals.
    let tiny = Image::from_rows(&[[0.0, 0.5, 1.0], [0.2, 0.4, 0.6]]);
    println!("{tiny:.2}");

    println!("wrote out/starter/negative.png and out/starter/brighter.png");
}
