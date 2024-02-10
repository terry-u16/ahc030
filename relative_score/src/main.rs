use anyhow::Context;
use clap::Parser;
use serde::Deserialize;
use std::{collections::HashMap, path::PathBuf};

#[derive(Debug, Parser)]
struct Args {
    #[clap(short = 'd', long = "dir")]
    dir: PathBuf,
    #[clap(short = 'f', long = "file")]
    file: Option<PathBuf>,
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
    let (best_scores, last_file) = load_all_scores(args)?;

    show_relative_score(last_file, best_scores)?;

    Ok(())
}

fn load_all_scores(args: Args) -> Result<(HashMap<u64, i64>, PathBuf), anyhow::Error> {
    let entries = std::fs::read_dir(args.dir)?;
    let mut last_file = None;
    let mut best_scores = HashMap::new();
    for entry in entries {
        let path = entry?.path();

        let Some(extension) = path.extension() else {
            continue;
        };

        if extension != "json" {
            continue;
        }

        let trial: Trial = serde_json::from_reader(std::fs::File::open(&path)?)?;

        for case in trial.results {
            // 0点はスキップ
            if case.score == 0 {
                continue;
            }

            let score = best_scores.entry(case.seed).or_insert(case.score);

            if *score < case.score {
                *score = case.score;
            }
        }

        last_file = Some(path);
    }
    let last_file = last_file.context("No json file found")?;
    let last_file = args.file.unwrap_or(last_file);
    Ok((best_scores, last_file))
}

fn show_relative_score(
    last_file: PathBuf,
    best_scores: HashMap<u64, i64>,
) -> Result<(), anyhow::Error> {
    let trial: Trial = serde_json::from_reader(std::fs::File::open(&last_file)?)?;
    println!("[Trial {}]", trial.time_stamp);
    
    let mut total_score = 0.0;

    for case in &trial.results {
        let best_score = best_scores.get(&case.seed).unwrap_or(&case.score);
        let relative_score = case.score as f64 / *best_score as f64 * 100.0;
        total_score += relative_score;
        println!("Seed: {:4} | Score: {:7.3}", case.seed, relative_score);
    }

    println!(
        "Average Score: {:.3}",
        total_score / trial.results.len() as f64
    );

    Ok(())
}
