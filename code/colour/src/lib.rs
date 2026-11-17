//! Coding exercises for the day "Colour": one module per unit.
//!
//! - `colour_vision`: From a spectrum to three numbers
//! - `rgb_linear`: RGB: gamma, linear maps & white balance
//! - `colour_perceptual`: HSV, CIELAB & colour segmentation
//! - `colour_display`: Colour for display: maps, merges & decorrelation
//!
//! Run a unit's program with `cargo run -p colour -- <unit>`, check it with
//! `cargo test -p colour <unit>::` (see the course page "Coding exercises: getting started").

pub mod common;
pub mod colour_vision;
pub mod rgb_linear;
pub mod colour_perceptual;
pub mod colour_display;
