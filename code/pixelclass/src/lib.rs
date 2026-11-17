//! Coding exercises for the unit "Pixel classification with a forest".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p pixelclass`, and run it with
//! `cargo run -p pixelclass` (pictures in `out/pixelclass/`).
//! `src/solution.rs` is a model solution: `cargo test -p pixelclass --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

// A small random forest (from the forest unit; filled in).
pub mod forest;
pub use forest::{argmax, Forest, Settings, Tree};
