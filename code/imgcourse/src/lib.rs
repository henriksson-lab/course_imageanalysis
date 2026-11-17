//! `imgcourse`: a small image library for the image analysis course.
//!
//! It is written to be **read**, not to be fast. An image is a grid of numbers,
//! `Image<T>`, indexed as `img[(x, y)]`:
//!
//! - `x` is the column, counted from the left, `0..width`;
//! - `y` is the row, counted from the top, `0..height`;
//! - grey values are `f32` between 0 (black) and 1 (white).
//!
//! The library only does the plumbing: storing pixels (`Image`, and `Volume` for 3D),
//! reading and writing files, the border rules, complex numbers, random numbers and test
//! images. The image analysis itself is what you write in the exercises.
//!
//! For later days it also has tools that earlier days taught you to write yourself:
//! a fast Fourier transform (`fft`) and small linear algebra (`linalg`: solving equations,
//! least squares, eigenvectors of a symmetric matrix), and `filters` (Gaussian blur, bilinear
//! sampling, RMSE and PSNR).
//!
//! ```
//! use imgcourse::*;
//!
//! let img = samples::cell();                       // the course micrograph, 256 × 256
//! let inverted = img.map(|v| 1.0 - v);              // the same function on every pixel
//! let mut out = Image::new(img.width(), img.height(), 0.0);
//! for (x, y) in img.coords() {                       // every pixel position, row by row
//!     out[(x, y)] = 0.5 * (img[(x, y)] + inverted[(x, y)]);
//! }
//! assert!((out[(10, 20)] - 0.5).abs() < 1e-6);
//! ```

mod border;
mod complex;
pub mod fft;
pub mod filters;
mod image;
pub mod io;
pub mod linalg;
mod random;
pub mod samples;
mod volume;

pub use border::Border;
pub use complex::Complex;
pub use image::Image;
pub use io::{gray_to_rgb, load_gray, load_rgb, save_gray, save_gray_signed, save_gray_stretched, save_labels, save_rgb};
pub use random::Rng;
pub use volume::Volume;
