//! Coding exercises for the day "Images, filters and frequencies": one module per unit.
//!
//! - `images`: Images as arrays & geometric transforms
//! - `kernels`: Filtering & convolution
//! - `fourier`: Images in frequency space
//! - `pca`: Principal component analysis
//!
//! Run a unit's program with `cargo run -p images -- <unit>`, check it with
//! `cargo test -p images <unit>::` (see the course page "Coding exercises: getting started").

pub mod images;
pub mod kernels;
pub mod fourier;
pub mod pca;
