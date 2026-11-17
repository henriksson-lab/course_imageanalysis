//! Coding exercises for the unit "Texture: co-occurrence matrices".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p texture-features`, and run it with
//! `cargo run -p texture-features` (pictures in `out/texture-features/`).
//! `src/solution.rs` is a model solution: `cargo test -p texture-features --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
