// The programs of day 2, one per unit:  cargo run -p day02 -- <unit>
// (add --features solution before -- to run the model solution). `--help` lists the units.

use clap::{Parser, Subcommand};
use day02::*;

/// Day 2: Edges, blobs and ridges. Runs one unit's program; the pictures go to out/<unit>/.
#[derive(Parser)]
#[command(name = "day02")]
struct Cli {
    #[command(subcommand)]
    unit: Unit,
}

#[derive(Subcommand)]
enum Unit {
    /// Edges: Gaussian derivatives and Canny
    #[command(name = "canny")]
    Canny,
    /// Scale space, blobs and pyramids
    #[command(name = "scale-space")]
    ScaleSpace,
    /// The Hessian: blobs, ridges and saddles
    #[command(name = "hessian")]
    Hessian,
    /// The structure tensor: orientation and corners
    #[command(name = "structure-tensor")]
    StructureTensor,
}

fn main() {
    match Cli::parse().unit {
        Unit::Canny => canny::run::main(),
        Unit::ScaleSpace => scale_space::run::main(),
        Unit::Hessian => hessian::run::main(),
        Unit::StructureTensor => structure_tensor::run::main(),
    }
}
