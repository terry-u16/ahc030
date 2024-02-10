use std::io::{self, BufReader};

use itertools::Itertools;
use proconio::source::line::LineSource;
#[allow(unused_imports)]
use proconio::*;
#[allow(unused_imports)]
use rand::prelude::*;

pub trait ChangeMinMax {
    fn change_min(&mut self, v: Self) -> bool;
    fn change_max(&mut self, v: Self) -> bool;
}

impl<T: PartialOrd> ChangeMinMax for T {
    fn change_min(&mut self, v: T) -> bool {
        *self > v && {
            *self = v;
            true
        }
    }

    fn change_max(&mut self, v: T) -> bool {
        *self < v && {
            *self = v;
            true
        }
    }
}

#[derive(Debug, Clone)]
struct Input {}

fn main() {
    let mut source = LineSource::new(BufReader::new(io::stdin()));

    input! {
        from &mut source,
        n: usize,
        m: usize,
        eps: f64,
    }

    let fields = vec![vec![0; n]; n];

    for _ in 0..m {
        input! {
            from &mut source,
            d: usize,
        }

        for _ in 0..d {
            input! {
                from &mut source,
                i: usize,
                j: usize,
            }
        }
    }

    let mut results = vec![];

    for i in 0..n {
        for j in 0..n {
            println!("q 1 {} {}", i, j);

            input! {
                from &mut source,
                response: i32,
            }

            if response != 0 {
                results.push((i, j));
            }
        }
    }

    let pos = results
        .iter()
        .map(|(i, j)| format!("{} {}", i, j))
        .join(" ");

    println!("a {} {}", results.len(), pos);
}
