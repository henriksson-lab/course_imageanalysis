//! Coding exercises for day 15, "Watershed": one module per unit.
//!
//! - `ws_landscape`: Images as landscapes: watershed by flooding
//! - `ws_flooding`: The flooding algorithm on a grid
//! - `ws_gradient`: Watershed segmentation of a real image
//! - `ws_distance`: Splitting touching objects with the distance transform
//!
//! Run a unit's program with `cargo run -p day15 -- <unit>`, check it with
//! `cargo test -p day15 <unit>::` (see the course page "Coding exercises: getting started").

pub mod common;
pub mod ws_landscape;
pub mod ws_flooding;
pub mod ws_gradient;
pub mod ws_distance;
