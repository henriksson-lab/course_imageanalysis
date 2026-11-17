//! Coding exercises for the unit "Marching cubes".
//!
//! Write your code in `src/marchingcubes/exercise.rs`. Check it with `cargo test -p marchingcubes marchingcubes::`, and run it
//! with `cargo run -p marchingcubes -- marchingcubes` (pictures in `out/marchingcubes/`).
//! `src/marchingcubes/solution.rs` is a model solution: `cargo test -p marchingcubes --features solution marchingcubes::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p marchingcubes -- marchingcubes`.
pub mod run;
