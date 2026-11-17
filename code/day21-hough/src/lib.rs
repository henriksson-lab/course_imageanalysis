//! Coding exercises for day 21, "Hough transform": one module per unit.
//!
//! - `hough_lines`: Hough transform: lines
//! - `hough_circles`: Hough transform: circles
//! - `ght`: The generalised Hough transform
//! - `ransac`: Hypothesise and verify: RANSAC
//!
//! Run a unit's program with `cargo run -p day21 -- <unit>`, check it with
//! `cargo test -p day21 <unit>::` (see the course page "Coding exercises: getting started").

pub mod common;
pub mod hough_lines;
pub mod hough_circles;
pub mod ght;
pub mod ransac;
