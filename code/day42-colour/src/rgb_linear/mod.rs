//! Coding exercises for the unit "RGB: gamma, linear maps and white balance".
//!
//! Write your code in `src/rgb_linear/exercise.rs`. Check it with `cargo test -p day42 rgb_linear::`, and run it
//! with `cargo run -p day42 -- rgb-linear` (pictures in `out/rgb-linear/`).
//! `src/rgb_linear/solution.rs` is a model solution: `cargo test -p day42 --features solution rgb_linear::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day42 -- rgb-linear`.
pub mod run;
