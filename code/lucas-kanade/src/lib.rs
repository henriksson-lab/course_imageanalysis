//! Coding exercises for the unit "Lucas–Kanade".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p lucas-kanade`, and run it on
//! the micrograph with `cargo run -p lucas-kanade` (pictures in `out/lucas-kanade/`).
//! `src/solution.rs` is a model solution: `cargo test -p lucas-kanade --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
