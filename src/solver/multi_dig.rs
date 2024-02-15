mod generator;
mod sampler;

use crate::{
    distributions::{GaussianBayesianEstimator, GaussianDistribution},
    grid::{Coord, CoordDiff, Map2d},
    problem::{Input, Judge},
    solver::multi_dig::{generator::State, sampler::ProbTable},
};
use itertools::Itertools;
use ordered_float::OrderedFloat;
use rand::seq::SliceRandom;
use rand_core::SeedableRng;
use rand_pcg::Pcg64Mcg;
use std::vec;
use std::{cmp::Reverse, time::Instant};

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
            if input.duration_corrector.elapsed(since).as_secs_f64() >= 2.8 {
                if self.answer_all(&states, input).is_ok() {
                    return;
                }

                for _ in 0..input.map_size * input.map_size * 2 {
                    self.judge.query_single(Coord::new(0, 0));
                }

                return;
            }

            eprintln!(
                "candidates: {}",
                env.shift_candidates.iter().map(|v| v.len()).join(" ")
            );

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

            let time_mul = if turn < 20 {
                5.0
            } else if turn < 50 {
                2.0
            } else {
                1.0
            };
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
                turn_duration * 0.3 * time_mul,
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
    /// matrix[oil_i] := oil_iの移動先の候補リスト
    shift_candidates: Vec<Vec<CoordDiff>>,
    bayesian_estimator: GaussianBayesianEstimator,
}

impl<'a> Env<'a> {
    fn new(input: &'a Input) -> Self {
        let relative_observation_indices = (0..input.oil_count)
            .map(|_| Map2d::new_with(vec![], input.map_size))
            .collect();
        let observations = vec![];
        let inv_relative_observation_indices = vec![];
        let overlaps = vec![];

        let shift_candidates = input
            .oils
            .iter()
            .map(|oil| {
                let mut candidates = vec![];

                for dr in 0..=input.map_size - oil.height {
                    for dc in 0..=input.map_size - oil.width {
                        candidates.push(CoordDiff::new(dr as isize, dc as isize));
                    }
                }

                candidates
            })
            .collect_vec();

        let bayesian_estimator = GaussianBayesianEstimator::new(input);

        Self {
            input,
            observations,
            relative_observation_indices,
            inv_relative_observation_indices,
            overlaps,
            shift_candidates,
            bayesian_estimator,
        }
    }

    fn add_observation(&mut self, input: &Input, observation: Observation) {
        self.update_shift_candidates(&observation);

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

    fn update_shift_candidates(&mut self, observation: &Observation) {
        //if self.shift_candidates[0].len()
        //    != (self.input.map_size - self.input.oils[0].height + 1)
        //        * (self.input.map_size - self.input.oils[0].width + 1)
        //{
        //    return;
        //}

        let sampled = observation.sampled;
        self.bayesian_estimator
            .update(&self.input, &observation.pos, sampled);

        let mut shift_candidates = vec![];

        for oil in self.input.oils.iter() {
            let size =
                (self.input.map_size - oil.height + 1) * (self.input.map_size - oil.width + 1);
            let mut candidates = Vec::with_capacity(size);
            let mut log_likelihoods = Vec::with_capacity(size);

            for dr in 0..=self.input.map_size - oil.height {
                for dc in 0..=self.input.map_size - oil.width {
                    let shift = CoordDiff::new(dr as isize, dc as isize);
                    let mut log_likelihood = 0.0;

                    for &p in oil.pos.iter() {
                        let p = p + shift;
                        let dist = self
                            .bayesian_estimator
                            .get_marginal_distribution(&self.input, p);

                        // 真の値が0でない確率を求める
                        let p = 1.0 - dist.calc_cumulative_dist(0.5);
                        log_likelihood += p.max(f64::MIN_POSITIVE).ln();
                    }

                    candidates.push(shift);
                    log_likelihoods.push(log_likelihood);
                }
            }

            let max_log_likelihoods = log_likelihoods
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            let mut probs = log_likelihoods
                .iter()
                .map(|&v| (v - max_log_likelihoods).exp())
                .collect_vec();
            let prob_sum = probs.iter().sum::<f64>();

            for p in probs.iter_mut() {
                *p /= prob_sum;
            }

            let mut indices = (0..size).collect_vec();
            indices.sort_unstable_by_key(|&i| Reverse(OrderedFloat(probs[i])));

            let mut cum_sum = 0.0;
            let mut result = vec![];

            for (i, &index) in indices.iter().enumerate() {
                result.push(candidates[index]);
                cum_sum += probs[index];

                // 99.999%を超えたら打ち切る
                if i >= 10 && cum_sum >= 0.99999 {
                    break;
                }
            }

            shift_candidates.push(result);
        }

        self.shift_candidates = shift_candidates;
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
