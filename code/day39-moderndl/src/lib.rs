//! Coding exercises for day 39, "Modern deep learning": one module per unit.
//!
//! - `instances`: From masks to objects
//! - `detection`: Finding objects as boxes
//! - `attention`: Attention, transformers and pretrained models
//! - `diffusion`: Diffusion models and generated detail
//!
//! Run a unit's program with `cargo run -p day39 -- <unit>`, check it with
//! `cargo test -p day39 <unit>::` (see the course page "Coding exercises: getting started").

pub mod instances;
pub mod detection;
pub mod attention;
pub mod diffusion;
