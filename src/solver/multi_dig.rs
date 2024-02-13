mod mcmc;
mod sampler;

use crate::{
    distributions::GaussianDistribution,
    grid::{Coord, CoordDiff, Map2d},
    problem::{Input, Judge},
    solver::multi_dig::{mcmc::State, sampler::ProbTable},
};
use itertools::Itertools;
use rand::{seq::SliceRandom, Rng};
use rand_core::SeedableRng;
use rand_distr::{Distribution, WeightedIndex};
use rand_pcg::Pcg64Mcg;
use std::time::Instant;
use std::vec;

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
        let all_coords = (0..input.map_size)
            .flat_map(|row| (0..input.map_size).map(move |col| Coord::new(row, col)))
            .collect_vec();
        let mut rng = Pcg64Mcg::from_entropy();
        let turn_duration = 2.0 / ((input.map_size as f64).powi(2) * 2.0);
        let since = Instant::now();
        let mut states = vec![State::new(
            vec![CoordDiff::new(0, 0); input.oil_count],
            &env,
        )];
        let mut prob_table = ProbTable::new(input);

        const ANSWER_THRESHOLD_RATIO: f64 = 100.0;

        for turn in 0..input.map_size * input.map_size * 2 {
            // TLE緊急回避モード
            if since.elapsed().as_secs_f64() >= 2.8 {
                if self.answer_all(&states, input).is_ok() {
                    return;
                }

                for _ in 0..input.map_size * input.map_size * 2 {
                    self.judge.query_single(Coord::new(0, 0));
                }

                return;
            }

            // 尤度の比に応じてMCMCの初期状態をサンプリング
            let max_log_likelihood = states
                .iter()
                .map(|s| s.log_likelihood)
                .fold(f64::MIN, f64::max);
            let weights = states
                .iter()
                .map(|s| (s.log_likelihood - max_log_likelihood).exp())
                .collect_vec();
            let dist = WeightedIndex::new(weights).unwrap();
            let state_i = dist.sample(&mut rng);
            let state = states[state_i].clone();

            let mut sampled_states = mcmc::mcmc(&env, state, turn_duration * 0.8, &mut rng);
            states.append(&mut sampled_states);
            states.sort_unstable();
            states.dedup();
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

            let time_mul = if turn < 20 { 5.0 } else { 1.0 };
            let max_sample_count = if turn < input.map_size * input.map_size {
                input.map_size * input.map_size
            } else if turn < input.map_size * input.map_size * 3 / 2 {
                input.map_size * input.map_size / 8
            } else {
                input.map_size * input.map_size / 32
            };

            let targets = sampler::select_sample_points(
                input,
                &mut prob_table,
                states.clone(),
                max_sample_count,
                turn_duration * 0.2 * time_mul,
                &mut rng,
            );
            let sampled = self.judge.query_multiple(&targets);
            let observation = Observation::new(targets, sampled, input);

            env.add_observation(input, observation);

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
    observations: Vec<Observation>,
    /// indices[oil_i][shift] := 影響する (obs_i, count) のVec
    relative_observation_indices: Vec<Map2d<Vec<(usize, usize)>>>,
    /// indices[obs_i] := 影響する (oil_i, count, shift) のVec
    inv_relative_observation_indices: Vec<Vec<(usize, usize, CoordDiff)>>,
    /// matrix[obs_i][oil_i][shift] := oil_iをshiftだけ動かした領域とobs_iの重なりの数
    overlaps: Vec<Vec<Map2d<usize>>>,
}

impl<'a> Env<'a> {
    fn new(input: &'a Input) -> Self {
        let relative_observation_indices = (0..input.oil_count)
            .map(|_| Map2d::new_with(vec![], input.map_size))
            .collect();
        let observations = vec![];
        let inv_relative_observation_indices = vec![];
        let overlaps = vec![];

        Self {
            input,
            observations,
            relative_observation_indices,
            inv_relative_observation_indices,
            overlaps,
        }
    }

    fn add_observation(&mut self, input: &Input, observation: Observation) {
        let obs_id = self.observations.len();
        let mut observed_map = Map2d::new_with(false, input.map_size);

        for &p in observation.pos.iter() {
            observed_map[p] = true;
        }

        let mut inv_relative_observation_indices = vec![];
        let mut overlaps = vec![];

        for (oil_i, oil) in input.oils.iter().enumerate() {
            let mut overlap = Map2d::new_with(0, input.map_size);

            for row in 0..=input.map_size - oil.height {
                for col in 0..=input.map_size - oil.width {
                    let c = Coord::new(row, col);
                    let shift = CoordDiff::new(row as isize, col as isize);
                    let mut count = 0;

                    for &p in oil.pos.iter() {
                        let p = p + shift;

                        if observed_map[p] {
                            count += 1;
                        }
                    }

                    overlap[c] = count;

                    if count > 0 {
                        self.relative_observation_indices[oil_i][c].push((obs_id, count));
                        inv_relative_observation_indices.push((oil_i, count, shift));
                    }
                }
            }

            overlaps.push(overlap);
        }

        self.inv_relative_observation_indices
            .push(inv_relative_observation_indices);
        self.observations.push(observation);
        self.overlaps.push(overlaps);
    }
}

#[derive(Debug, Clone)]
struct Observation {
    pos: Vec<Coord>,
    sampled: i32,
    /// k番目の要素はΣv(pi)=kとなる対数尤度を表す
    log_likelihoods: Vec<f64>,
}

impl Observation {
    fn new(pos: Vec<Coord>, sampled: i32, input: &Input) -> Self {
        assert!(pos.len() > 0);

        let likelihoods_len = pos.len() * input.oil_count + 1;
        let k = pos.len() as f64;
        let x = sampled as f64;

        // k = 1のときは特別扱い
        if pos.len() == 1 {
            let mut log_likelihoods = vec![f64::MIN_POSITIVE.ln(); likelihoods_len];
            log_likelihoods[sampled as usize] = 0.0;

            return Self {
                pos,
                sampled,
                log_likelihoods,
            };
        }

        let mut log_likelihoods = Vec::with_capacity(likelihoods_len);

        for true_v in 0..likelihoods_len {
            let v = true_v as f64;
            let mean = (k - v) * input.eps + v * (1.0 - input.eps);
            let variance = k * input.eps * (1.0 - input.eps);
            let std_dev = variance.sqrt();
            let gauss = GaussianDistribution::new(mean, std_dev);

            let likelihood = if sampled == 0 {
                gauss.calc_cumulative_dist(x + 0.5)
            } else {
                gauss.calc_cumulative_dist(x + 0.5) - gauss.calc_cumulative_dist(x - 0.5)
            }
            .max(f64::MIN_POSITIVE);
            let log_likelihood = likelihood.ln();

            log_likelihoods.push(log_likelihood);
        }

        Self {
            pos,
            sampled,
            log_likelihoods,
        }
    }
}
