//! Coding exercises for the unit "Edge-stopping diffusion (Perona–Malik)".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p perona-malik`, and run it on
//! the images with `cargo run -p perona-malik` (pictures in `out/perona-malik/`).
//! `src/solution.rs` is a model solution: `cargo test -p perona-malik --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
