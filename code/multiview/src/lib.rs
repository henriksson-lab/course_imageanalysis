//! Coding exercises for the day "Geometry from multiple images": one module per unit.
//!
//! - `cameras`: The pinhole camera and calibration
//! - `two_view`: Two views: epipolar geometry, triangulation
//! - `stereo`: Stereo: from disparity to depth
//! - `sfm`: Structure from motion and bundle adjustment
//!
//! Run a unit's program with `cargo run -p multiview -- <unit>`, check it with
//! `cargo test -p multiview <unit>::` (see the course page "Coding exercises: getting started").

pub mod common;
pub mod cameras;
pub mod two_view;
pub mod stereo;
pub mod sfm;
