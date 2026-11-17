//! Coding exercises for the day "Ising model and Markov random fields": one module per unit.
//!
//! - `ising`: Energy and probability: the Ising model
//! - `mcmc`: Sampling, temperature and the phase transition
//! - `mrf_denoise`: Bayesian denoising with an Ising prior
//! - `gmrf`: Quadratic smoothing and total variation
//!
//! Run a unit's program with `cargo run -p ising -- <unit>`, check it with
//! `cargo test -p ising <unit>::` (see the course page "Coding exercises: getting started").

pub mod ising;
pub mod mcmc;
pub mod mrf_denoise;
pub mod gmrf;
