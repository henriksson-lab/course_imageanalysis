//! Coding exercises for the day "Training networks in practice": one module per unit.
//!
//! - `optimisers`: Optimisers and learning rates
//! - `normalisation`: Initialisation, normalisation and residual connections
//! - `augmentation`: Regularisation and data augmentation
//! - `training_curves`: Reading training curves
//!
//! Run a unit's program with `cargo run -p training -- <unit>`, check it with
//! `cargo test -p training <unit>::` (see the course page "Coding exercises: getting started").
//! `common` holds the small network and the data sets that several units use.

pub mod common;
pub mod optimisers;
pub mod normalisation;
pub mod augmentation;
pub mod training_curves;
