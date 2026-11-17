//! Coding exercises for the day "Snakes and active shape models": one module per unit.
//!
//! - `snakes`: Active contours (snakes)
//! - `snake_matrix`: The snake as a linear system
//! - `shape_models`: Shape models: landmarks, Procrustes and PCA
//! - `asm`: Active shape models
//!
//! Run a unit's program with `cargo run -p snakes -- <unit>`, check it with
//! `cargo test -p snakes <unit>::` (see the course page "Coding exercises: getting started").

pub mod common;
pub mod snakes;
pub mod snake_matrix;
pub mod shape_models;
pub mod asm;
