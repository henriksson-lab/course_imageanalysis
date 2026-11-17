//! Coding exercises for the day "Marching cubes": one module per unit.
//!
//! - `volumes3d`: 3D images
//! - `isocontour`: Level sets & marching squares
//! - `marchingcubes`: Marching cubes
//! - `surface_measure`: Measuring surfaces
//!
//! Run a unit's program with `cargo run -p marchingcubes -- <unit>`, check it with
//! `cargo test -p marchingcubes <unit>::` (see the course page "Coding exercises: getting started").

pub mod common;
pub mod volumes3d;
pub mod isocontour;
pub mod marchingcubes;
pub mod surface_measure;
