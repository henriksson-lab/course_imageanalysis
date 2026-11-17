//! Coding exercises for the unit "Pixel classification with a forest".
//!
//! Write your code in `src/pixelclass/exercise.rs`. Check it with `cargo test -p day33 pixelclass::`, and run it with
//! `cargo run -p day33 -- pixelclass` (pictures in `out/pixelclass/`).
//! `src/pixelclass/solution.rs` is a model solution: `cargo test -p day33 --features solution pixelclass::`.

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

/// The program of this unit: `cargo run -p day33 -- pixelclass`.
pub mod run;
