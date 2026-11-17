//! Coding exercises for day 36, "Neural networks": one module per unit.
//!
//! - `neuralnet`: Neural networks: matrix, bend, matrix
//! - `backprop`: Learning: loss, gradient descent and backpropagation
//! - `cnn`: Convolutional networks
//! - `unet`: Segmentation networks
//!
//! Run a unit's program with `cargo run -p day36 -- <unit>`, check it with
//! `cargo test -p day36 <unit>::` (see the course page "Coding exercises: getting started").

pub mod neuralnet;
pub mod backprop;
pub mod cnn;
pub mod unet;
