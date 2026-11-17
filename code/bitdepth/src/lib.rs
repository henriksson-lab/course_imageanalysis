//! Coding exercises for the day "Bit depth and noise": one module per unit.
//!
//! - `quantisation`: Quantisation and bit depth
//! - `cameranoise`: Camera noise, SNR and PSNR
//! - `highbit`: 12- and 16-bit images in practice
//! - `contrast`: Contrast and intensity transforms
//! - `floatingpoint`: Floating point
//!
//! Run a unit's program with `cargo run -p bitdepth -- <unit>`, check it with
//! `cargo test -p bitdepth <unit>::` (see the course page "Coding exercises: getting started").

pub mod quantisation;
pub mod cameranoise;
pub mod highbit;
pub mod contrast;
pub mod floatingpoint;
