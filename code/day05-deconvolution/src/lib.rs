//! Coding exercises for day 5, "Deconvolution": one module per unit.
//!
//! - `deconv`: Linear deconvolution: inverse, Wiener, Tikhonov
//! - `regularise`: Regularisation: what we add, and how much
//! - `richardson_lucy`: Richardson–Lucy, 3D, and the PSF
//! - `denoise`: Denoising and what you may measure
//!
//! Run a unit's program with `cargo run -p day05 -- <unit>`, check it with
//! `cargo test -p day05 <unit>::` (see the course page "Coding exercises: getting started").

pub mod common;
pub mod deconv;
pub mod regularise;
pub mod richardson_lucy;
pub mod denoise;
