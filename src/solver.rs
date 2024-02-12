pub mod multi_dig;
pub mod single_dig;

use crate::problem::{Input, Judge};

pub trait Solver {
    fn solve(&mut self, input: &Input, judge: &mut Judge) -> Result<(), ()>;
}
