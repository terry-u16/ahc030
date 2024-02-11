pub mod single_dig;

use crate::problem::Input;

pub trait Solver {
    fn solve(self, input: &Input);
}
