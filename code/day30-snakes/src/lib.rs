//! Coding exercises for day 30, "Snakes and active shape models": one module per unit.
//!
//! - `snakes`: Active contours (snakes)
//! - `snake_matrix`: The snake as a linear system
//! - `shape_models`: Shape models: landmarks, Procrustes and PCA
//! - `asm`: Active shape models
//!
//! Run a unit's program with `cargo run -p day30 -- <unit>`, check it with
//! `cargo test -p day30 <unit>::` (see the course page "Coding exercises: getting started").

pub mod snakes;
pub mod snake_matrix;
pub mod shape_models;
pub mod asm;
