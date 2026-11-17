//! Coding exercises for day 44, "Colocalisation": one module per unit.
//!
//! - `coloc`: Colocalisation of two channels
//! - `coloc_null`: Is it more than chance?
//! - `coloc_objects`: Objects instead of pixels
//! - `coloc_confounds`: What the microscope does to colocalisation
//!
//! Run a unit's program with `cargo run -p day44 -- <unit>`, check it with
//! `cargo test -p day44 <unit>::` (see the course page "Coding exercises: getting started").

pub mod common;
pub mod coloc;
pub mod coloc_null;
pub mod coloc_objects;
pub mod coloc_confounds;
