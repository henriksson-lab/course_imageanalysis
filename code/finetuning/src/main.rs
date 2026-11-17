// The programs of this day, one per unit:  cargo run -p finetuning -- <unit>
// (add --features solution before -- to run the model solution). `--help` lists the units.

use clap::{Parser, Subcommand};
use finetuning::*;

/// Fine-tuning and transfer learning. Runs one unit's program; the pictures go to out/<unit>/.
#[derive(Parser)]
#[command(name = "finetuning")]
struct Cli {
    #[command(subcommand)]
    unit: Unit,
}

#[derive(Subcommand)]
enum Unit {
    /// Transfer learning
    #[command(name = "transfer")]
    Transfer,
    /// Fine-tuning strategies
    #[command(name = "finetune-strategies")]
    FinetuneStrategies,
    /// Few labels: annotation and the human in the loop
    #[command(name = "few-labels")]
    FewLabels,
    /// Domain shift and adaptation
    #[command(name = "domain-shift")]
    DomainShift,
}

fn main() {
    match Cli::parse().unit {
        Unit::Transfer => transfer::run::main(),
        Unit::FinetuneStrategies => finetune_strategies::run::main(),
        Unit::FewLabels => few_labels::run::main(),
        Unit::DomainShift => domain_shift::run::main(),
    }
}
