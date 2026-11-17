//! Coding exercises for day 48, "3D graphics": one module per unit.
//!
//! - `pipeline3d`: Meshes, coordinate spaces and projection
//! - `rasterise`: From triangles to shaded pixels
//! - `gpuformats`: How graphics cards store colour
//! - `volrender`: Seeing voxels: projections and volume rendering
//!
//! Run a unit's program with `cargo run -p day48 -- <unit>`, check it with
//! `cargo test -p day48 <unit>::` (see the course page "Coding exercises: getting started").

pub mod pipeline3d;
pub mod rasterise;
pub mod gpuformats;
pub mod volrender;
