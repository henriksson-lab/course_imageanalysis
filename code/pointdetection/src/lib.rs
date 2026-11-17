//! Coding exercises for the day "Point detection": one module per unit.
//!
//! - `spot_filters`: Finding spots: filter, then local maxima
//! - `spot_threshold`: How many false spots? Thresholds from noise
//! - `spot_fit`: Sub-pixel localisation and the Cramér–Rao bound
//! - `spot_crowd`: Crowded spots and evaluating a detector
//!
//! Run a unit's program with `cargo run -p pointdetection -- <unit>`, check it with
//! `cargo test -p pointdetection <unit>::` (see the course page "Coding exercises: getting started").

pub mod common;
pub mod spot_filters;
pub mod spot_threshold;
pub mod spot_fit;
pub mod spot_crowd;
