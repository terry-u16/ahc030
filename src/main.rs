mod common;
mod grid;
mod problem;
mod solver;

#[allow(unused_imports)]
use proconio::*;
#[allow(unused_imports)]
use rand::prelude::*;
use solver::solve;

use crate::problem::Judge;

fn main() {
    let mut judge = Judge::new();
    let input = judge.read_input();

    solve(judge, &input);
}
