// The programs of day 35, one per unit:  cargo run -p day35 -- <unit>
// (add --features solution before -- to run the model solution). `--help` lists the units.

use clap::{Parser, Subcommand};
use day35::*;

/// Day 35: Gaussian mixtures. Runs one unit's program; the pictures go to out/<unit>/.
#[derive(Parser)]
#[command(name = "day35")]
struct Cli {
    #[command(subcommand)]
    unit: Unit,
}

#[derive(Subcommand)]
enum Unit {
    /// Feature spaces & k-means
    #[command(name = "kmeans")]
    Kmeans,
    /// The multivariate Gaussian & mixtures
    #[command(name = "gaussian-mixture")]
    GaussianMixture,
    /// Fitting a mixture: EM
    #[command(name = "em")]
    Em,
    /// Mixtures on images
    #[command(name = "gmm-images")]
    GmmImages,
}

fn main() {
    match Cli::parse().unit {
        Unit::Kmeans => kmeans::run::main(),
        Unit::GaussianMixture => gaussian_mixture::run::main(),
        Unit::Em => em::run::main(),
        Unit::GmmImages => gmm_images::run::main(),
    }
}
