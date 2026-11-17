//! Coding exercises for the day "Optical flow": one module per unit.
//!
//! - `flow_constraint`: Motion between frames: brightness constancy and the aperture problem
//! - `lucas_kanade`: Lucas–Kanade: local least squares and pyramids
//! - `horn_schunck`: Horn–Schunck and evaluating flow
//! - `nonrigid_reg`: Non-rigid registration
//!
//! Run a unit's program with `cargo run -p opticalflow -- <unit>`, check it with
//! `cargo test -p opticalflow <unit>::` (see the course page "Coding exercises: getting started").

pub mod common;
pub mod flow_constraint;
pub mod lucas_kanade;
pub mod horn_schunck;
pub mod nonrigid_reg;
