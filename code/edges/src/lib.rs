//! Coding exercises for the day "Edges, blobs and ridges": one module per unit.
//!
//! - `canny`: Edges: Gaussian derivatives and Canny
//! - `scale_space`: Scale space, blobs and pyramids
//! - `hessian`: The Hessian: blobs, ridges and saddles
//! - `structure_tensor`: The structure tensor: orientation and corners
//!
//! Run a unit's program with `cargo run -p edges -- <unit>`, check it with
//! `cargo test -p edges <unit>::` (see the course page "Coding exercises: getting started").

pub mod canny;
pub mod scale_space;
pub mod hessian;
pub mod structure_tensor;
