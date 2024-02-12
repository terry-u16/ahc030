mod common;
mod distributions;
mod grid;
mod problem;
mod solver;

use std::time::Duration;

use grid::Coord;
#[allow(unused_imports)]
use proconio::*;
#[allow(unused_imports)]
use rand::prelude::*;
use solver::Solver;

use crate::problem::Judge;

fn main() {
    let mut judge = Judge::new();
    let input = judge.read_input();

    judge.set_query_limit(input.map_size * input.map_size - 10);

    if input.eps <= 0.15 && input.oil_count <= 12 {
        let mut solver = solver::multi_dig::MultiDigSolver::new(Duration::from_millis(2000));

        if solver.solve(&input, &mut judge).is_ok() {
            return;
        }
    }

    judge.set_query_limit(input.map_size * input.map_size * 2);
    let duration = Duration::from_millis(2900) - input.since.elapsed();
    let mut solver = solver::single_dig::SingleDigSolver::new(duration);

    if solver.solve(&input, &mut judge).is_ok() {
        return;
    }

    while judge.can_query() {
        judge.query_single(Coord::new(0, 0));
    }
}
