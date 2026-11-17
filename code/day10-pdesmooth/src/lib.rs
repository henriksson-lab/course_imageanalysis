//! Coding exercises for day 10, "Edge-preserving smoothing and PDEs": one module per unit.
//!
//! - `heat_equation`: Smoothing as heat flow: the heat equation and explicit schemes
//! - `perona_malik`: Edge-stopping diffusion: Perona–Malik
//! - `tv_flow`: Total variation flow: smoothing as energy descent
//! - `bilateral_nlm`: Neighbourhood filters in depth: bilateral and non-local means
//! - `inpainting`: Filling holes: diffusion and exemplar inpainting
//! - `smoothers`: All smoothers on one image
//!
//! Run a unit's program with `cargo run -p day10 -- <unit>`, check it with
//! `cargo test -p day10 <unit>::` (see the course page "Coding exercises: getting started").

pub mod common;
pub mod heat_equation;
pub mod perona_malik;
pub mod tv_flow;
pub mod bilateral_nlm;
pub mod inpainting;
pub mod smoothers;
