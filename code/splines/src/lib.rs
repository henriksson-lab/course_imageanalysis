//! Coding exercises for the day "Splines and NURBS": one module per unit.
//!
//! - `bezier`: Polynomial wiggles, piecewise cubics and Bézier curves
//! - `bsplines`: B-splines and NURBS
//! - `spline_fit`: Interpolating, approximating and smoothing splines
//! - `spline_images`: Splines on images: interpolation and free-form deformation
//!
//! Run a unit's program with `cargo run -p splines -- <unit>`, check it with
//! `cargo test -p splines <unit>::` (see the course page "Coding exercises: getting started").

pub mod common;
pub mod bezier;
pub mod bsplines;
pub mod spline_fit;
pub mod spline_images;
