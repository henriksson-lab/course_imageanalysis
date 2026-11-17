//! Coding exercises for day 9, "Radon transform and tomography": one module per unit.
//!
//! - `radon`: Projections and the sinogram
//! - `backprojection`: Back-projection and the Fourier slice theorem
//! - `fbp`: Filtered back-projection and sampling
//! - `algebraic_recon`: Reconstruction as a linear system
//!
//! Run a unit's program with `cargo run -p day09 -- <unit>`, check it with
//! `cargo test -p day09 <unit>::` (see the course page "Coding exercises: getting started").

pub mod radon;
pub mod backprojection;
pub mod fbp;
pub mod algebraic_recon;
