//! Coding exercises for day 33, "Random forests": one module per unit.
//!
//! - `dtree`: Decision trees
//! - `forest`: Bagging and random forests
//! - `forest_objects`: Feature tables & feature importance
//! - `pixelclass`: Pixel classification with a forest
//!
//! Run a unit's program with `cargo run -p day33 -- <unit>`, check it with
//! `cargo test -p day33 <unit>::` (see the course page "Coding exercises: getting started").

pub mod common;
pub mod dtree;
pub mod forest;
pub mod forest_objects;
pub mod pixelclass;
