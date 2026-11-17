//! Coding exercises for day 24, "Keypoints and descriptors": one module per unit.
//!
//! - `keypoints`: Keypoints: corners and blobs across scale
//! - `descriptors`: Describing a keypoint: gradient histograms and bits
//! - `feature_matching`: Matching descriptors: nearest, ratio and mutual
//! - `feature_alignment`: From matches to a transform: RANSAC, stitching and registration
//!
//! Run a unit's program with `cargo run -p day24 -- <unit>`, check it with
//! `cargo test -p day24 <unit>::` (see the course page "Coding exercises: getting started").

pub mod common;
pub mod keypoints;
pub mod descriptors;
pub mod feature_matching;
pub mod feature_alignment;
