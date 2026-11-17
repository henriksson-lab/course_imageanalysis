//! Coding exercises for the day "Gaussian mixtures": one module per unit.
//!
//! - `kmeans`: Feature spaces & k-means
//! - `gaussian_mixture`: The multivariate Gaussian & mixtures
//! - `em`: Fitting a mixture: EM
//! - `gmm_images`: Mixtures on images
//!
//! Run a unit's program with `cargo run -p gmm -- <unit>`, check it with
//! `cargo test -p gmm <unit>::` (see the course page "Coding exercises: getting started").

pub mod common;
pub mod kmeans;
pub mod gaussian_mixture;
pub mod em;
pub mod gmm_images;
