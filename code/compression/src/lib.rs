//! Coding exercises for the day "Image compression": one module per unit.
//!
//! - `entropy`: Information, entropy and entropy codes
//! - `lossless`: Prediction, dictionaries and lossless formats
//! - `dct`: Transform coding, the DCT and the KLT
//! - `jpeg`: The JPEG pipeline, artefacts and quality metrics
//! - `compress_science`: Compression and measurements, JPEG 2000, formats
//!
//! Run a unit's program with `cargo run -p compression -- <unit>`, check it with
//! `cargo test -p compression <unit>::` (see the course page "Coding exercises: getting started").

pub mod entropy;
pub mod lossless;
pub mod dct;
pub mod jpeg;
pub mod compress_science;
