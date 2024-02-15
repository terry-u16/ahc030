mod generator;
mod observation;
mod sampler;

use crate::{
    grid::{Coord, CoordDiff, Map2d},
    problem::{Input, Judge},
    solver::multi_dig::{generator::State, observation::Observation, sampler::ProbTable},
};
use rand::seq::SliceRandom;
use rand_core::SeedableRng;
use rand_pcg::Pcg64Mcg;
use std::time::Instant;
use std::vec;

use self::observation::ObservationManager;

use super::Solver;

pub struct MultiDigSolver {
    judge: Judge,
}

impl MultiDigSolver {
    pub fn new(judge: Judge) -> Self {
        Self { judge }
    }

    fn answer_all(&mut self, states: &Vec<State>, input: &Input) -> Result<(), ()> {
        // 全部順番に答える
        for state in states.iter() {
            let answer = state.to_answer(input);
            if self.judge.answer(&answer).is_ok() {
                return Ok(());
            }
        }

        Err(())
    }
}

impl Solver for MultiDigSolver {
    fn solve(&mut self, input: &crate::problem::Input) {
        let mut env = Env::new(input);
        let mut rng = Pcg64Mcg::from_entropy();
        let all_time_mul = if input.oil_count <= 8 && input.eps <= 0.15 {
            2.0
        } else {
            1.0
        };
        let turn_duration = all_time_mul * 2.0 / ((input.map_size as f64).powi(2) * 2.0);
        let since = Instant::now();
        let mut states = vec![State::new(
            vec![CoordDiff::new(0, 0); input.oil_count],
            &env,
        )];
        let mut prob_table = ProbTable::new(input);

        const ANSWER_THRESHOLD_RATIO: f64 = 100.0;

        for turn in 0..input.map_size * input.map_size * 2 {
            // TLE緊急回避モード
            if input.duration_corrector.elapsed(since).as_secs_f64() >= 2.95 {
                if self.answer_all(&states, input).is_ok() {
                    return;
                }

                for _ in 0..input.map_size * input.map_size * 2 {
                    self.judge.query_single(Coord::new(0, 0));
                }

                return;
            }

            // 新たな置き方を生成
            states = generator::generate_states(&env, states, turn_duration * 0.7, &mut rng);
            states.sort_unstable();
            states.dedup();
            states.shuffle(&mut rng);
            states
                .sort_unstable_by(|a, b| b.log_likelihood.partial_cmp(&a.log_likelihood).unwrap());

            let state = &states[0];
            let ratio = if states.len() >= 2 {
                (state.log_likelihood - states[1].log_likelihood).exp()
            } else {
                f64::INFINITY
            };

            self.judge.comment(&format!(
                "found: {} log_likelihood: {:.3}, ratio: {:.3}",
                states.len(),
                state.log_likelihood,
                ratio
            ));

            let mut map = Map2d::new_with(0.0, input.map_size);

            for i in 0..input.oil_count {
                for p in input.oils[i].pos.iter() {
                    map[*p + state.shift[i]] += 1.0 / input.oil_count as f64;
                }
            }

            self.judge.comment_colors(&map);

            if ratio >= ANSWER_THRESHOLD_RATIO {
                if self.judge.answer(&state.to_answer(input)).is_ok() {
                    return;
                }
            } else if self.judge.remaining_query_count() <= 10
                && self.judge.remaining_query_count() <= states.len()
            {
                if self.answer_all(&states, input).is_ok() {
                    return;
                }
            }

            let time_mul = if turn < 10 {
                20.0
            } else if turn < 20 {
                5.0
            } else if turn < 50 {
                2.0
            } else {
                1.0
            };
            let max_sample_count = if turn < input.map_size * input.map_size {
                input.map_size * input.map_size
            } else {
                input.map_size * input.map_size / 8
            };

            let targets = sampler::select_sample_points(
                input,
                &mut prob_table,
                states.clone(),
                max_sample_count,
                turn_duration * 0.3 * time_mul,
                &mut rng,
            );

            let sampled = self.judge.query_multiple(&targets);
            let observation = Observation::new(targets, sampled, input);

            env.obs.add_observation(input, observation);

            for state in states.iter_mut() {
                state.add_last_observation(&env);
            }

            let max_log_likelihood = states
                .iter()
                .map(|s| s.log_likelihood)
                .fold(f64::MIN, f64::max);

            let retain_threshold = ANSWER_THRESHOLD_RATIO.ln();
            states.retain(|s| max_log_likelihood - s.log_likelihood <= retain_threshold);
        }
    }
}

#[derive(Debug, Clone)]
struct Env<'a> {
    input: &'a Input,
    obs: ObservationManager,
}

impl<'a> Env<'a> {
    fn new(input: &'a Input) -> Self {
        let obs = ObservationManager::new(input);

        Self { input, obs }
    }
}
