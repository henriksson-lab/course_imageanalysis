//! Coding exercises for the day "Neural networks": one module per unit.
//!
//! - `neuralnet`: Neural networks: matrix, bend, matrix
//! - `backprop`: Learning: loss, gradient descent and backpropagation
//! - `cnn`: Convolutional networks
//! - `unet`: Segmentation networks
//!
//! Run a unit's program with `cargo run -p neuralnets -- <unit>`, check it with
//! `cargo test -p neuralnets <unit>::` (see the course page "Coding exercises: getting started").

pub mod neuralnet;
pub mod backprop;
pub mod cnn;
pub mod unet;
