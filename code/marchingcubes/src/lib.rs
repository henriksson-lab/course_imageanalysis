//! Coding exercises for the unit "Marching cubes".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p marchingcubes`, and run it
//! with `cargo run -p marchingcubes` (pictures in `out/marchingcubes/`).
//! `src/solution.rs` is a model solution: `cargo test -p marchingcubes --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
