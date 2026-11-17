//! Coding exercises for day 11, "Sparse representations": one module per unit.
//!
//! - `sparsity`: Sparsity and the ℓ1 penalty
//! - `ista`: Solving with ℓ1: thresholding iterations and pursuit
//! - `compressed_sensing`: Compressed sensing
//! - `cs_mri`: Fast MRI with compressed sensing
//! - `dictionaries`: Sparse coding and dictionary learning
//!
//! Run a unit's program with `cargo run -p day11 -- <unit>`, check it with
//! `cargo test -p day11 <unit>::` (see the course page "Coding exercises: getting started").

pub mod common;
pub mod sparsity;
pub mod ista;
pub mod compressed_sensing;
pub mod cs_mri;
pub mod dictionaries;
