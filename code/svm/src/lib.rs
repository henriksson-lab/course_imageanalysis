//! Coding exercises for the day "Support vector machines": one module per unit.
//!
//! - `classify`: Linear classifiers
//! - `maxmargin`: Maximum margin and support vectors
//! - `kernel_trick`: Kernel functions and the kernel trick
//! - `pixel_classify`: SVMs for pixel classification
//!
//! Run a unit's program with `cargo run -p svm -- <unit>`, check it with
//! `cargo test -p svm <unit>::` (see the course page "Coding exercises: getting started").

pub mod common;
pub mod classify;
pub mod maxmargin;
pub mod kernel_trick;
pub mod pixel_classify;
