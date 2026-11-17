// The programs of day 36, one per unit:  cargo run -p day36 -- <unit>
// (add --features solution before -- to run the model solution). `--help` lists the units.

use clap::{Parser, Subcommand};
use day36::*;

/// Day 36: Neural networks. Runs one unit's program; the pictures go to out/<unit>/.
#[derive(Parser)]
#[command(name = "day36")]
struct Cli {
    #[command(subcommand)]
    unit: Unit,
}

#[derive(Subcommand)]
enum Unit {
    /// Neural networks: matrix, bend, matrix
    #[command(name = "neuralnet")]
    Neuralnet,
    /// Learning: loss, gradient descent and backpropagation
    #[command(name = "backprop")]
    Backprop,
    /// Convolutional networks
    #[command(name = "cnn")]
    Cnn,
    /// Segmentation networks
    #[command(name = "unet")]
    Unet,
}

fn main() {
    match Cli::parse().unit {
        Unit::Neuralnet => neuralnet::run::main(),
        Unit::Backprop => backprop::run::main(),
        Unit::Cnn => cnn::run::main(),
        Unit::Unet => unet::run::main(),
    }
}
