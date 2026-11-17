// The programs of day 24, one per unit:  cargo run -p day24 -- <unit>
// (add --features solution before -- to run the model solution). `--help` lists the units.

use clap::{Parser, Subcommand};
use day24::*;

/// Day 24: Keypoints and descriptors. Runs one unit's program; the pictures go to out/<unit>/.
#[derive(Parser)]
#[command(name = "day24")]
struct Cli {
    #[command(subcommand)]
    unit: Unit,
}

#[derive(Subcommand)]
enum Unit {
    /// Keypoints: corners and blobs across scale
    #[command(name = "keypoints")]
    Keypoints,
    /// Describing a keypoint: gradient histograms and bits
    #[command(name = "descriptors")]
    Descriptors,
    /// Matching descriptors: nearest, ratio and mutual
    #[command(name = "feature-matching")]
    FeatureMatching,
    /// From matches to a transform: RANSAC, stitching and registration
    #[command(name = "feature-alignment")]
    FeatureAlignment,
}

fn main() {
    match Cli::parse().unit {
        Unit::Keypoints => keypoints::run::main(),
        Unit::Descriptors => descriptors::run::main(),
        Unit::FeatureMatching => feature_matching::run::main(),
        Unit::FeatureAlignment => feature_alignment::run::main(),
    }
}
