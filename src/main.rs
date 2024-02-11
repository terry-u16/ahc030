mod common;
mod grid;
mod problem;
mod solver;

#[allow(unused_imports)]
use proconio::*;
#[allow(unused_imports)]
use rand::prelude::*;
use solver::Solver as _;

use crate::problem::Judge;

fn main() {
    let mut judge = Judge::new();
    let input = judge.read_input();

    let solver = solver::single_dig::SingleDigSolver::new(judge);
    solver.solve(&input);
}
