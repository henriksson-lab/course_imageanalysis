//! Coding exercises for day 50, "Ray tracing": one module per unit.
//!
//! - `raycast`: Rays from the camera, and where they hit
//! - `whitted`: Shadows, mirrors and glass
//! - `bvh`: Tracing fast: boxes, trees and grids
//! - `pathtrace`: Path tracing: Monte Carlo and noise
//!
//! Run a unit's program with `cargo run -p day50 -- <unit>`, check it with
//! `cargo test -p day50 <unit>::` (see the course page "Coding exercises: getting started").

pub mod common;
pub mod raycast;
pub mod whitted;
pub mod bvh;
pub mod pathtrace;
