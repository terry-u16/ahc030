use anyhow::Context;
use clap::Parser;
use itertools::Itertools;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use serde::Deserialize;
use std::{collections::HashMap, path::PathBuf, str::FromStr};

#[derive(Debug, Parser)]
struct Args {
    #[clap(short = 'd', long = "dir")]
    dir: PathBuf,
    #[clap(short = 'f', long = "file")]
    file: Option<PathBuf>,
    #[clap(short = 'o', long = "opt", default_value = "max")]
    opt: Opt,
}

#[derive(Debug, Clone, Copy)]
enum Opt {
    Max,
    Min,
}

impl FromStr for Opt {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "max" => Ok(Self::Max),
            "min" => Ok(Self::Min),
            _ => Err(anyhow::anyhow!("Invalid opt")),
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "PascalCase")]
struct Trial {
    time_stamp: String,
    _comment: String,
    results: Vec<TestCase>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "PascalCase")]
struct TestCase {
    seed: u64,
    score: i64,
    _elapsed: String,
    _message: String,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let (best_scores, last_file) = load_all_scores(&args)?;
    show_relative_score(&args, last_file, best_scores)?;

    Ok(())
}

fn load_all_scores(args: &Args) -> Result<(HashMap<u64, i64>, PathBuf), anyhow::Error> {
    let entries = std::fs::read_dir(&args.dir)?.collect_vec();
    let mut best_scores = HashMap::new();

    let entries = entries
        .iter()
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| path.extension().unwrap_or_default() == "json")
        .filter_map(|path| {
            std::fs::read_to_string(&path)
                .ok()
                .and_then(|s| Some((path, s)))
        })
        .collect_vec();

    let trials = entries
        .par_iter()
        .filter_map(|(path, s)| {
            serde_json::from_str(s)
                .ok()
                .map(|trial: Trial| (path, trial))
        })
        .collect::<Vec<_>>();
    let mut last_file = None;

    for (path, trial) in trials {
        for case in trial.results {
            // 0点はスキップ
            if case.score == 0 {
                continue;
            }

            let score = best_scores.entry(case.seed).or_insert(case.score);
            let pred = match args.opt {
                Opt::Max => case.score > *score,
                Opt::Min => case.score < *score,
            };

            if pred {
                *score = case.score;
            }
        }

        last_file = Some(path);
    }

    let last_file = last_file.context("No json file found")?;
    let last_file = args.file.as_ref().unwrap_or(&last_file).clone();
    Ok((best_scores, last_file))
}

fn show_relative_score(
    args: &Args,
    last_file: PathBuf,
    best_scores: HashMap<u64, i64>,
) -> Result<(), anyhow::Error> {
    let trial: Trial = serde_json::from_reader(std::fs::File::open(&last_file)?)?;
    println!("[Trial {}]", trial.time_stamp);

    let mut total_score = 0.0;

    for case in &trial.results {
        let score = if case.score == 0 {
            0.0
        } else {
            let best_score = best_scores.get(&case.seed).unwrap_or(&case.score);
            let relative_score = match args.opt {
                Opt::Max => case.score as f64 / *best_score as f64,
                Opt::Min => *best_score as f64 / case.score as f64,
            };
            relative_score * 100.0
        };

        total_score += score;
        println!("Seed: {:4} | Score: {:7.3}", case.seed, score);
    }

    println!(
        "Average Score: {:.3}",
        total_score / trial.results.len() as f64
    );

    Ok(())
}
