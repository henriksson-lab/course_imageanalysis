//! Coding exercises for the day "Moments": one module per unit.
//!
//! - `moments_basis`: Moments as projections onto monomials
//! - `moment_invariants`: Normalisation and moment invariants
//! - `orthogonal_moments`: Better bases: Legendre moments and reconstruction
//! - `zernike`: Zernike moments on the unit disc
//!
//! Run a unit's program with `cargo run -p moments -- <unit>`, check it with
//! `cargo test -p moments <unit>::` (see the course page "Coding exercises: getting started").

pub mod moments_basis;
pub mod moment_invariants;
pub mod orthogonal_moments;
pub mod zernike;
