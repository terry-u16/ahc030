mod common;
mod grid;
mod problem;

#[allow(unused_imports)]
use proconio::*;
#[allow(unused_imports)]
use rand::prelude::*;

use crate::{
    grid::{Coord, Map2d},
    problem::Judge,
};

fn main() {
    let mut judge = Judge::new();
    let input = judge.read_input();

    let mut results = vec![];
    let mut map = Map2d::new(vec![0.0; input.map_size * input.map_size], input.map_size);

    for row in 0..input.map_size {
        for col in 0..input.map_size {
            let c = Coord::new(row, col);

            judge.comment_colors(&map);
            let response = judge.query_single(c);

            if response != 0 {
                results.push(c);
            }

            map[c] = response as f64;
        }
    }

    judge.answer(&results).unwrap();
}
