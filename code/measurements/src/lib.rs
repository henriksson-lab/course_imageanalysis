//! Coding exercises for the day "Measurements": one module per unit.
//!
//! - `intensity_features`: Intensity inside an object
//! - `texture_features`: Texture: co-occurrence matrices
//! - `relations`: Neighbours, parents and children
//!
//! Run a unit's program with `cargo run -p measurements -- <unit>`, check it with
//! `cargo test -p measurements <unit>::` (see the course page "Coding exercises: getting started").

pub mod intensity_features;
pub mod texture_features;
pub mod relations;
