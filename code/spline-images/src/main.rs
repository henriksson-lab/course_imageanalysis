// Interpolators on an impulse, a step and the micrograph (zoom, ten rotations), and a free-form
// deformation of a checkerboard; pictures in out/spline-images/.
//   cargo run -p spline-images                      (your code)
//   cargo run -p spline-images --features solution  (the model solution)

use imgcourse::*;
use spline_images::*;

/// The interpolators compared here. Nearest and bilinear are plumbing from the images unit.
#[derive(Clone, Copy)]
enum Method {
    Nearest,
    Bilinear,
    BsplineNoPrefilter,
    CubicConvolution,
    Bspline,
}

const METHODS: [(Method, &str); 5] = [
    (Method::Nearest, "nearest"),
    (Method::Bilinear, "bilinear"),
    (Method::BsplineNoPrefilter, "B-spline, no prefilter"),
    (Method::CubicConvolution, "cubic convolution"),
    (Method::Bspline, "cubic B-spline"),
];

fn nearest(img: &Image<f32>, x: f32, y: f32) -> f32 {
    img.get_border(x.round() as isize, y.round() as isize, Border::Mirror)
}

fn bilinear(img: &Image<f32>, x: f32, y: f32) -> f32 {
    let (x0, y0) = (x.floor(), y.floor());
    let (fx, fy) = (x - x0, y - y0);
    let (x0, y0) = (x0 as isize, y0 as isize);
    let p = |dx, dy| img.get_border(x0 + dx, y0 + dy, Border::Mirror);
    (1.0 - fy) * ((1.0 - fx) * p(0, 0) + fx * p(1, 0)) + fy * ((1.0 - fx) * p(0, 1) + fx * p(1, 1))
}

/// Resample `img` at the positions `pos(x, y)` with one interpolator.
fn resample(img: &Image<f32>, method: Method, width: usize, height: usize, pos: impl Fn(usize, usize) -> (f32, f32)) -> Image<f32> {
    let coef = match method {
        Method::Bspline => prefilter(img),
        _ => img.clone(),
    };
    Image::from_fn(width, height, |x, y| {
        let (u, v) = pos(x, y);
        match method {
            Method::Nearest => nearest(img, u, v),
            Method::Bilinear => bilinear(img, u, v),
            Method::BsplineNoPrefilter | Method::Bspline => bspline_value(&coef, u, v),
            Method::CubicConvolution => cubic_convolution(img, u, v),
        }
    })
}

fn main() {
    let dir = "out/spline-images";

    // An impulse and a step, as rows (images of height 1).
    let mut impulse = Image::new(41, 1, 0.0f32);
    impulse[(20, 0)] = 1.0;
    let step = Image::from_fn(40, 1, |x, _| if x <= 20 { 0.0 } else { 100.0 });
    let c = prefilter_1d(impulse.row(0));
    println!("impulse: prefiltered coefficients c0..c3 = {:.4}, {:.4}, {:.4}, {:.4}", c[20], c[21], c[22], c[23]);
    println!("                           at the pixel   halfway   range near a step 0 → 100");
    for (method, name) in METHODS {
        let at = resample(&impulse, method, 1, 1, |_, _| (20.0, 0.0))[(0, 0)];
        let half = resample(&impulse, method, 1, 1, |_, _| (20.5, 0.0))[(0, 0)];
        let fine = resample(&step, method, 400, 1, |x, _| (15.0 + x as f32 / 40.0, 0.0));
        println!("  {name:24} {at:10.3}   {half:7.3}   {:7.2} to {:6.2}", fine.min(), fine.max());
    }

    // Zoom a 16 × 16 patch of the micrograph 8 times with each interpolator, side by side.
    let cell = samples::cell();
    let patch = cell.crop(100, 60, 16, 16);
    let mut strip = Image::new(5 * 128 + 4 * 4, 128, 1.0f32);
    for (i, (method, _)) in METHODS.iter().enumerate() {
        let big = resample(&patch, *method, 128, 128, |x, y| ((x as f32 - 3.5) / 8.0, (y as f32 - 3.5) / 8.0));
        for (x, y) in big.coords() {
            strip[(i * 132 + x, y)] = big[(x, y)].clamp(0.0, 1.0);
        }
    }
    save_gray(&strip, format!("{dir}/zoom.png"));

    // Rotate the micrograph ten times by 36° about its centre; compare with the original inside
    // a disc of radius 100 (grey levels 0..255).
    println!("\nmicrograph rotated 10 × 36°: RMS change inside the central disc (grey levels)");
    let (n, centre) = (cell.width(), 127.5f32);
    let (s, co) = 36f32.to_radians().sin_cos();
    let mut strip = Image::new(5 * 200 + 4 * 4, 200, 1.0f32);
    for (i, (method, name)) in METHODS.iter().enumerate() {
        let mut img = cell.clone();
        for _ in 0..10 {
            // Inverse mapping: the output pixel (x, y) reads the input at the position turned back.
            img = resample(&img, *method, n, n, |x, y| {
                let (dx, dy) = (x as f32 - centre, y as f32 - centre);
                (centre + co * dx + s * dy, centre - s * dx + co * dy)
            });
        }
        let (mut sum, mut count) = (0.0, 0);
        for (x, y) in img.coords() {
            if (x as f32 - centre).hypot(y as f32 - centre) < 100.0 {
                sum += (255.0 * (img[(x, y)] - cell[(x, y)])).powi(2);
                count += 1;
            }
        }
        println!("  {name:24} {:5.1}", (sum / count as f32).sqrt());
        let crop = img.crop(28, 28, 200, 200);
        for (x, y) in crop.coords() {
            strip[(i * 204 + x, y)] = crop[(x, y)].clamp(0.0, 1.0);
        }
    }
    save_gray(&strip, format!("{dir}/rotate10.png"));

    // Free-form deformation of a checkerboard, control spacing 32 (a 9 × 9 grid).
    let checker = samples::checker(256, 256, 16);
    for (name, moves) in [("one", vec![((4, 4), [9.0, 0.0])]), ("fold", vec![((4, 4), [44.0, 0.0]), ((5, 4), [-44.0, 0.0])])] {
        let mut d = Image::new(9, 9, [0.0f32, 0.0]);
        for ((k, l), v) in &moves {
            d[(*k, *l)] = *v;
        }
        // Warp: the pixel at x shows the input at x − u(x) (inverse mapping, approximately).
        let coef = prefilter(&checker);
        let warped = Image::from_fn(256, 256, |x, y| {
            let u = ffd_displacement(&d, 32.0, x as f32, y as f32);
            bspline_value(&coef, x as f32 - u[0], y as f32 - u[1]).clamp(0.0, 1.0)
        });
        save_gray(&warped, format!("{dir}/ffd-{name}.png"));
        // The Jacobian determinant of x ↦ x + u(x), by central differences.
        let jac = Image::from_fn(256, 256, |x, y| {
            let (x, y) = (x as f32, y as f32);
            let (ux1, ux0) = (ffd_displacement(&d, 32.0, x + 0.5, y), ffd_displacement(&d, 32.0, x - 0.5, y));
            let (uy1, uy0) = (ffd_displacement(&d, 32.0, x, y + 0.5), ffd_displacement(&d, 32.0, x, y - 0.5));
            (1.0 + ux1[0] - ux0[0]) * (1.0 + uy1[1] - uy0[1]) - (ux1[1] - ux0[1]) * (uy1[0] - uy0[0])
        });
        let folded = jac.pixels().iter().filter(|&&j| j <= 0.0).count();
        println!("\nFFD '{name}': smallest Jacobian determinant {:.3}, folded pixels {folded}", jac.min());
        if name == "one" {
            let at = |x: f32, y: f32| ffd_displacement(&d, 32.0, x, y)[0];
            println!("  pixels at (128,128), (160,128), (160,160) move by {:.2}, {:.2}, {:.2}", at(128.0, 128.0), at(160.0, 128.0), at(160.0, 160.0));
        }
        save_gray_signed(&jac.map(|j| j - 1.0), format!("{dir}/ffd-{name}-jacobian.png"));
    }
    println!("wrote zoom.png, rotate10.png and the ffd-*.png pictures to {dir}/");
}
