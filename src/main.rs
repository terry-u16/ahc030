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

    eprintln!(
        "k={:.3}, b={:.3}, r={:.3}, multi={:.3}",
        input.time_conductor.k(),
        input.time_conductor.b(),
        input.time_conductor.phase_ratio(),
        input.params.use_multi_dig_solver
    );

    let mut solver: Box<dyn Solver> = if input.params.use_multi_dig_solver() {
        Box::new(solver::multi_dig::MultiDigSolver::new(judge))
    } else {
        Box::new(solver::single_dig::SingleDigSolver::new(judge))
    };

    solver.solve(&input);
}
