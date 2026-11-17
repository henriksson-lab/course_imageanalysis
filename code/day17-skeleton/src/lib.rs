//! Coding exercises for day 17, "Skeletons": one module per unit.
//!
//! - `skel_thinning`: Hit-or-miss and thinning
//! - `skel_medialaxis`: Medial axis, distance transform and reconstruction
//! - `skel_graph`: From skeleton to graph: pruning and length
//! - `skel_neurovessel`: Measuring neurons and vessel networks
//!
//! Run a unit's program with `cargo run -p day17 -- <unit>`, check it with
//! `cargo test -p day17 <unit>::` (see the course page "Coding exercises: getting started").

pub mod skel_thinning;
pub mod skel_medialaxis;
pub mod skel_graph;
pub mod skel_neurovessel;
