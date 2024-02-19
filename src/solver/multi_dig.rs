mod generator;
mod observation;
mod sampler;

use crate::{
    grid::{Coord, CoordDiff, Map2d},
    problem::{Input, Judge},
    solver::multi_dig::{generator::State, observation::Observation, sampler::ProbTable},
};
use bitset_fixed::BitSet;
use itertools::Itertools;
use ordered_float::OrderedFloat;
use rand::seq::SliceRandom;
use rand_core::SeedableRng;
use rand_pcg::Pcg64Mcg;
use std::{cmp::Reverse, vec};

use self::observation::ObservationManager;

use super::Solver;

pub struct MultiDigSolver<'a> {
    judge: Judge<'a>,
}

impl<'a> MultiDigSolver<'a> {
    pub fn new(judge: Judge<'a>) -> Self {
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

impl<'a> Solver for MultiDigSolver<'a> {
    fn solve(&mut self, input: &crate::problem::Input) {
        let mut env = Env::new(input);
        let mut rng = Pcg64Mcg::from_entropy();

        let mut states = vec![State::new(
            vec![CoordDiff::new(0, 0); input.oil_count],
            &env,
        )];
        let mut prob_table = ProbTable::new(input);

        let elapsed = input.duration_corrector.elapsed(input.since);
        let time_table = input.time_conductor.get_time_table(
            elapsed,
            Input::TIME_LIMIT,
            self.judge.max_query_count(),
        );

        const ANSWER_THRESHOLD_RATIO: f64 = 100.0;

        while self.judge.can_query() {
            let turn = self.judge.query_count() + 1;
            eprintln!(
                "===== turn {} / {} =====",
                turn,
                self.judge.max_query_count()
            );

            // TLE緊急回避モード
            if input.duration_corrector.elapsed(input.since) >= Input::TIME_LIMIT {
                if self.answer_all(&states, input).is_ok() {
                    return;
                }

                for _ in 0..input.map_size * input.map_size * 2 {
                    self.judge.query_single(Coord::new(0, 0));
                }

                return;
            }

            let time_limit_turn = time_table[turn as usize - 1];

            // 新たな置き方を生成
            let duration = (time_limit_turn
                .saturating_sub(input.duration_corrector.elapsed(input.since)))
            .mul_f64(input.time_conductor.phase_ratio());
            states = generator::generate_states(&env, states, duration.as_secs_f64(), &mut rng);
            states.sort_unstable();
            states.dedup();
            states.shuffle(&mut rng);
            states
                .sort_unstable_by(|a, b| b.log_likelihood.partial_cmp(&a.log_likelihood).unwrap());

            let mut likelihoods = states
                .iter()
                .group_by(|s| {
                    let mut map = BitSet::new(input.map_size * input.map_size);
                    for i in 0..input.oil_count {
                        map |= &input.oils[i].get_shifted_bitset(input, s.shift[i]);
                    }
                    map
                })
                .into_iter()
                .map(|(_, ss)| {
                    let mut likelihood = 0.0;
                    let mut hash = None;

                    for p in ss.into_iter() {
                        likelihood += (p.log_likelihood - states[0].log_likelihood).exp();
                        if hash.is_none() {
                            hash = Some(p.hash);
                        }
                    }

                    (likelihood, hash.unwrap())
                })
                .collect_vec();
            likelihoods.sort_unstable_by_key(|(likelihood, _)| Reverse(OrderedFloat(*likelihood)));

            let state = &states[0];
            let ratio = if likelihoods.len() >= 2 {
                likelihoods[0].0 / likelihoods[1].0
            } else {
                f64::INFINITY
            };

            self.judge.comment(&format!(
                "found: {} log_likelihood: {:.3}, ratio: {:.3}",
                likelihoods.len(),
                likelihoods[0].0,
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
                let state = states.iter().find(|s| s.hash == likelihoods[0].1).unwrap();

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

            let max_sample_count = input.map_size * input.map_size;
            let duration =
                time_limit_turn.saturating_sub(input.duration_corrector.elapsed(input.since));

            let targets = sampler::select_sample_points(
                input,
                &mut prob_table,
                states.clone(),
                max_sample_count,
                duration.as_secs_f64(),
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
