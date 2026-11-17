//! Coding exercises for day 41, "Diffusion models": one module per unit.
//!
//! - `score_matching`: Noise levels, denoisers and the score
//! - `diffusion_sampling`: Samplers, steps, and judging the samples
//! - `guidance`: Conditioning, guidance and latent diffusion
//! - `diffusion_inverse`: Diffusion priors for inverse problems
//!
//! Run a unit's program with `cargo run -p day41 -- <unit>`, check it with
//! `cargo test -p day41 <unit>::` (see the course page "Coding exercises: getting started").

pub mod common;
pub mod score_matching;
pub mod diffusion_sampling;
pub mod guidance;
pub mod diffusion_inverse;
