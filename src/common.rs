use std::time::{Duration, Instant};

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

#[derive(Debug, Clone, Copy)]
pub struct DurationCorrector {
    mul: f64,
}

#[allow(dead_code)]
impl DurationCorrector {
    pub fn new(mul: f64) -> Self {
        Self { mul }
    }

    pub fn from_env() -> Self {
        let mul = std::env::var("DURATION_MUL")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1.0);
        eprintln!("DURATION_MUL = {}", mul);

        Self::new(mul)
    }

    pub fn elapsed(&self, since: Instant) -> Duration {
        let elapsed = since.elapsed();
        elapsed.mul_f64(self.mul)
    }

    pub fn elapsed_f64(&self, since: Instant) -> f64 {
        let elapsed = since.elapsed();
        elapsed.as_secs_f64() * self.mul
    }
}
