//! Coding exercises for day 4, "Image formation": one module per unit.
//!
//! - `psf`: Image formation and the point spread function
//! - `resolution`: Resolution, sampling and measuring the PSF
//! - `illposed`: Running the model backwards: an ill-posed problem
//!
//! Run a unit's program with `cargo run -p day04 -- <unit>`, check it with
//! `cargo test -p day04 <unit>::` (see the course page "Coding exercises: getting started").

pub mod psf;
pub mod resolution;
pub mod illposed;
