//! Coding exercises for day 32, "Dynamic programming": one module per unit.
//!
//! - `dp_paths`: The DP principle: minimal paths on a grid
//! - `polar_boundary`: Closed boundaries: DP in polar coordinates
//! - `livewire`: When there is no order: Dijkstra and live-wire
//! - `fastmarching`: From Dijkstra to fast marching
//!
//! Run a unit's program with `cargo run -p day32 -- <unit>`, check it with
//! `cargo test -p day32 <unit>::` (see the course page "Coding exercises: getting started").

pub mod common;
pub mod dp_paths;
pub mod polar_boundary;
pub mod livewire;
pub mod fastmarching;
