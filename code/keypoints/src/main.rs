// The programs of this day, one per unit:  cargo run -p keypoints -- <unit>
// (add --features solution before -- to run the model solution). `--help` lists the units.

use clap::{Parser, Subcommand};

/// Keypoints and descriptors. Runs one unit's program; the pictures go to out/<unit>/.
#[derive(Parser)]
#[command(name = "keypoints")]
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
    /// Aligning point sets: iterative closest point
    #[command(name = "icp")]
    Icp,
}

fn main() {
    match Cli::parse().unit {
        Unit::Keypoints => keypoints::keypoints::run::main(),
        Unit::Descriptors => keypoints::descriptors::run::main(),
        Unit::FeatureMatching => keypoints::feature_matching::run::main(),
        Unit::FeatureAlignment => keypoints::feature_alignment::run::main(),
        Unit::Icp => keypoints::icp::run::main(),
    }
}
