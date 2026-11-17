//! Coding exercises for day 22, "Integral images": one module per unit.
//!
//! - `integral_image`: Integral images: any rectangle in four lookups
//! - `box_filters`: Box filters & local statistics
//! - `rect_features`: Rectangle features
//! - `viola_jones`: Boosted detectors & the cascade
//!
//! Run a unit's program with `cargo run -p day22 -- <unit>`, check it with
//! `cargo test -p day22 <unit>::` (see the course page "Coding exercises: getting started").

pub mod common;
pub mod integral_image;
pub mod box_filters;
pub mod rect_features;
pub mod viola_jones;
