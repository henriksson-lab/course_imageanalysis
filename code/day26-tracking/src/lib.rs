//! Coding exercises for day 26, "Tracking": one module per unit.
//!
//! - `track_linking`: From detections to tracks
//! - `track_predict`: Motion models and the Kalman filter
//! - `track_events`: Gaps, divisions and lineage trees
//! - `track_analysis`: Speed, MSD and modes of motion
//!
//! Run a unit's program with `cargo run -p day26 -- <unit>`, check it with
//! `cargo test -p day26 <unit>::` (see the course page "Coding exercises: getting started").

pub mod track_linking;
pub mod track_predict;
pub mod track_events;
pub mod track_analysis;
