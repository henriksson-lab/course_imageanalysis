//! Coding exercises for the day "Random forests": one module per unit.
//!
//! - `dtree`: Decision trees
//! - `forest`: Bagging and random forests
//! - `forest_objects`: Feature tables & feature importance
//! - `pixelclass`: Pixel classification with a forest
//!
//! Run a unit's program with `cargo run -p randomforest -- <unit>`, check it with
//! `cargo test -p randomforest <unit>::` (see the course page "Coding exercises: getting started").

pub mod common;
pub mod dtree;
pub mod forest;
pub mod forest_objects;
pub mod pixelclass;
