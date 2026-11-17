//! Coding exercises for day 23, "FFT registration": one module per unit.
//!
//! - `matching`: Finding a shift by sliding and scoring
//! - `xcorr_fft`: Cross-correlation with the FFT
//! - `phasecorr`: Phase correlation
//! - `stitching`: From a peak to a mosaic
//!
//! Run a unit's program with `cargo run -p day23 -- <unit>`, check it with
//! `cargo test -p day23 <unit>::` (see the course page "Coding exercises: getting started").

pub mod common;
pub mod matching;
pub mod xcorr_fft;
pub mod phasecorr;
pub mod stitching;
