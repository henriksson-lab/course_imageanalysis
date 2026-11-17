//! Coding exercises for the unit "Tracking: gaps, divisions and lineage trees".
//!
//! Write your code in `src/track_events/exercise.rs`. Check it with `cargo test -p day26 track_events::`, and run the
//! examples with `cargo run -p day26 -- track-events` (pictures in `out/track-events/`).
//! `src/track_events/solution.rs` is a model solution: `cargo test -p day26 --features solution track_events::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

// Provided: global assignment with alternative costs (not an exercise; see track-linking).
mod assign;
pub use assign::{hungarian, join_with_alternatives};

/// The program of this unit: `cargo run -p day26 -- track-events`.
pub mod run;
