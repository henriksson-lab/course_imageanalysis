//! Coding exercises for day 31, "Level sets": one module per unit.
//!
//! - `implicit`: Curves as level sets
//! - `levelset_eq`: Moving a curve: the level set equation
//! - `geodesic`: Edge-based segmentation: geodesic active contours
//! - `chanvese`: Region-based segmentation: Chan–Vese
//!
//! Run a unit's program with `cargo run -p day31 -- <unit>`, check it with
//! `cargo test -p day31 <unit>::` (see the course page "Coding exercises: getting started").

pub mod common;
pub mod implicit;
pub mod levelset_eq;
pub mod geodesic;
pub mod chanvese;
