pub mod multi_dig;
pub mod single_dig;
pub mod bayesian;

use crate::problem::Input;

pub trait Solver {
    fn solve(&mut self, input: &Input);
}
