//! Coding exercises for the day "Wavelets": one module per unit.
//!
//! - `haar`: Localised basis functions and the Haar transform
//! - `filterbank`: Multiresolution, filter banks and Daubechies wavelets
//! - `dwt2d`: The 2D wavelet transform and sparsity of images
//! - `shrinkage`: Denoising by wavelet shrinkage
//!
//! Run a unit's program with `cargo run -p wavelets -- <unit>`, check it with
//! `cargo test -p wavelets <unit>::` (see the course page "Coding exercises: getting started").

pub mod common;
pub mod haar;
pub mod filterbank;
pub mod dwt2d;
pub mod shrinkage;
