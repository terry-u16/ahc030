mod common;
mod data_structures;
mod distributions;
mod grid;
mod problem;
mod solver;

#[allow(unused_imports)]
use proconio::*;
#[allow(unused_imports)]
use rand::prelude::*;
use solver::Solver;

use crate::problem::Judge;

fn main() {
    let mut judge = Judge::new();
    let input = judge.read_input();

    let mut solver: Box<dyn Solver> = if input.oil_count <= 10 {
        Box::new(solver::multi_dig::MultiDigSolver::new(judge))
    } else {
        Box::new(solver::single_dig::SingleDigSolver::new(judge))
    };

    solver.solve(&input);
}
