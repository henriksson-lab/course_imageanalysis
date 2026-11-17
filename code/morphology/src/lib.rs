//! Coding exercises for the day "Morphology": one module per unit.
//!
//! - `morph_binary`: Erosion and dilation
//! - `morph_openclose`: Opening, closing & the distance transform
//! - `morph_grey`: Grey-level morphology: top-hat & gradient
//! - `morph_reconstruct`: Reconstruction & h-extrema
//!
//! Run a unit's program with `cargo run -p morphology -- <unit>`, check it with
//! `cargo test -p morphology <unit>::` (see the course page "Coding exercises: getting started").

pub mod common;
pub mod morph_binary;
pub mod morph_openclose;
pub mod morph_grey;
pub mod morph_reconstruct;
