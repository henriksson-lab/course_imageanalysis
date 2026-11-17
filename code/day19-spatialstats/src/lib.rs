//! Coding exercises for day 19, "Spatial statistics of objects": one module per unit.
//!
//! - `point_patterns`: Point patterns and complete spatial randomness
//! - `nn_functions`: Nearest neighbours and empty space: G, F and J
//! - `ripley_k`: Ripley's K, the L function and the pair correlation
//! - `envelopes`: Is it more than chance? Simulation envelopes
//! - `marked_patterns`: Two cell types and an uneven density
//!
//! Run a unit's program with `cargo run -p day19 -- <unit>`, check it with
//! `cargo test -p day19 <unit>::` (see the course page "Coding exercises: getting started").

pub mod point_patterns;
pub mod nn_functions;
pub mod ripley_k;
pub mod envelopes;
pub mod marked_patterns;
