use std::{cmp::Reverse, ops::Range};

use itertools::Itertools;
use ordered_float::OrderedFloat;

use crate::{
    common::ChangeMinMax,
    distributions::GaussianDistribution,
    grid::{Coord, CoordDiff, Map2d},
    problem::Input,
};

#[derive(Debug, Clone)]
pub(super) struct ObservationManager {
    pub observations: Vec<Observation>,
    /// indices[oil_i][shift] := 影響する (obs_i, count) のVec
    pub relative_observation_indices: Vec<Map2d<Vec<(usize, usize)>>>,
    /// indices[obs_i] := 影響する (oil_i, count, shift) のVec
    pub inv_relative_observation_indices: Vec<Vec<(usize, usize, CoordDiff)>>,
    /// matrix[obs_i][oil_i][shift] := oil_iをshiftだけ動かした領域とobs_iの重なりの数
    pub overlaps: Vec<Vec<Map2d<usize>>>,
    pub shift_candidates: Vec<Vec<CoordDiff>>,
    /// matrix[oil_i][shift] := oil_iをshiftだけ動かすときの対数尤度
    shift_log_likelihoods: Vec<Map2d<f64>>,
    /// matrix[obs_i][oil_i] := oil_i以外の油田とobs_iとの重なりの最小値と最大値
    overlap_min_max: Vec<Vec<(usize, usize)>>,
}

impl ObservationManager {
    pub(super) fn new(input: &Input) -> Self {
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
                for row in 0..=input.map_size - oil.height {
                    for col in 0..=input.map_size - oil.width {
                        candidates.push(CoordDiff::new(row as isize, col as isize));
                    }
                }

                candidates
            })
            .collect_vec();

        let shift_log_likelihoods = (0..input.oil_count)
            .map(|_| Map2d::new_with(0.0, input.map_size))
            .collect_vec();
        let overlap_min_max = vec![];

        Self {
            observations,
            relative_observation_indices,
            inv_relative_observation_indices,
            overlaps,
            shift_candidates,
            shift_log_likelihoods,
            overlap_min_max,
        }
    }

    pub(super) fn add_observation(&mut self, input: &Input, observation: Observation) {
        let obs_id = self.observations.len();

        // 観測による条件式の更新
        let mut observed_map = Map2d::new_with(false, input.map_size);

        for &p in observation.pos.iter() {
            observed_map[p] = true;
        }

        let mut inv_relative_observation_indices = vec![];
        let mut overlaps = vec![];
        let mut overlap_mins = vec![];
        let mut overlap_maxs = vec![];

        for (oil_i, oil) in input.oils.iter().enumerate() {
            let mut overlap = Map2d::new_with(0, input.map_size);
            let mut overlap_min = usize::MAX;
            let mut overlap_max = usize::MIN;

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
                    overlap_min.change_min(count);
                    overlap_max.change_max(count);

                    if count > 0 {
                        self.relative_observation_indices[oil_i][c].push((obs_id, count));
                        inv_relative_observation_indices.push((oil_i, count, shift));
                    }
                }
            }

            overlaps.push(overlap);
            overlap_mins.push(overlap_min);
            overlap_maxs.push(overlap_max);
        }

        self.overlaps.push(overlaps);

        let overlap_min_sum = overlap_mins.iter().sum::<usize>();
        let overlap_max_sum = overlap_maxs.iter().sum::<usize>();
        let overlap_min_max = (0..input.oil_count)
            .map(|i| {
                let min = overlap_min_sum - overlap_mins[i];
                let max = overlap_max_sum - overlap_maxs[i];
                (min, max)
            })
            .collect_vec();
        self.overlap_min_max.push(overlap_min_max);

        self.inv_relative_observation_indices
            .push(inv_relative_observation_indices);
        self.observations.push(observation);

        // 左右からDP
        let overlap_probs = Self::dp_overlap(input, &self.overlaps[obs_id]);

        // 候補をアップデート
        let mut shift_candidates = vec![];

        for (oil_i, oil) in input.oils.iter().enumerate() {
            let mut all_shifts = vec![];
            let mut log_likelihoods = vec![];
            let (other_min, other_max) = self.overlap_min_max[obs_id][oil_i];

            for row in 0..=input.map_size - oil.height {
                for col in 0..=input.map_size - oil.width {
                    let c = Coord::new(row, col);
                    let shift = CoordDiff::new(row as isize, col as isize);
                    let overlap = self.overlaps[obs_id][oil_i][c];

                    let mut likelihood = 0.0;

                    let min = overlap + other_min;
                    let max = overlap + other_max;
                    let p0 = &overlap_probs[oil_i][other_min..=other_max];
                    let p1 = &self.observations[obs_id].likelihoods[min..=max];

                    for (p0, p1) in p0.iter().zip(p1) {
                        likelihood += p0 * p1;
                    }

                    self.shift_log_likelihoods[oil_i][c] += likelihood.ln();

                    all_shifts.push(shift);
                    log_likelihoods.push(self.shift_log_likelihoods[oil_i][c]);
                }
            }

            let max_log_likelihood = log_likelihoods.iter().copied().fold(f64::MIN, f64::max);
            let mut likelihoods = log_likelihoods
                .iter()
                .map(|&v| (v - max_log_likelihood).exp())
                .collect_vec();
            let likelihood_sum = likelihoods.iter().sum::<f64>();

            for l in likelihoods.iter_mut() {
                *l /= likelihood_sum
            }

            let mut indices = (0..all_shifts.len()).collect_vec();
            indices.sort_unstable_by_key(|&i| Reverse(OrderedFloat(likelihoods[i])));

            let mut result = vec![];
            let mut sum = 0.0;

            for (i, &index) in indices.iter().enumerate() {
                sum += likelihoods[index];
                result.push(all_shifts[index]);

                if sum >= 0.99999 {
                    break;
                }
            }

            shift_candidates.push(result);
        }

        self.shift_candidates = shift_candidates;
    }

    fn dp_overlap(input: &Input, overlaps: &[Map2d<usize>]) -> Vec<Vec<f64>> {
        // 重なる個数をメモしておく
        let mut overlap_probs = vec![];

        for (oil_i, oil) in input.oils.iter().enumerate() {
            let mut overlap_vec = vec![];
            let mut max_overlap = 0;

            for row in 0..=input.map_size - oil.height {
                for col in 0..=input.map_size - oil.width {
                    let c = Coord::new(row, col);
                    overlap_vec.push(overlaps[oil_i][c]);
                    max_overlap.change_max(overlaps[oil_i][c]);
                }
            }

            let candidate_count = overlap_vec.len();
            let max_overlap = overlap_vec.iter().copied().max().unwrap();
            let mut overlap_prob = vec![0.0; max_overlap + 1];

            for &v in overlap_vec.iter() {
                overlap_prob[v] += 1.0 / candidate_count as f64;
            }

            overlap_probs.push(overlap_prob);
        }

        let mut prefix_dp = vec![vec![1.0]];

        for overlap_prob in overlap_probs.iter() {
            let pre_dp = prefix_dp.last().unwrap();
            let mut next_dp = vec![0.0; pre_dp.len() + overlap_prob.len() - 1];

            for j in 0..pre_dp.len() {
                for k in 0..overlap_prob.len() {
                    next_dp[j + k] += pre_dp[j] * overlap_prob[k];
                }
            }

            prefix_dp.push(next_dp);
        }

        let mut suffix_dp = vec![vec![1.0]];

        for overlap_prob in overlap_probs.iter().rev() {
            let pre_dp = suffix_dp.last().unwrap();
            let mut next_dp = vec![0.0; pre_dp.len() + overlap_prob.len() - 1];

            for j in 0..pre_dp.len() {
                for k in 0..overlap_prob.len() {
                    next_dp[j + k] += pre_dp[j] * overlap_prob[k];
                }
            }

            suffix_dp.push(next_dp);
        }

        suffix_dp.reverse();

        let mut result = vec![];

        for i in 0..input.oil_count {
            let pre_dp = &prefix_dp[i];
            let suf_dp = &suffix_dp[i + 1];

            let mut dp = vec![0.0; pre_dp.len() + suf_dp.len() - 1];

            for j in 0..pre_dp.len() {
                for k in 0..suf_dp.len() {
                    dp[j + k] += pre_dp[j] * suf_dp[k];
                }
            }

            result.push(dp);
        }

        result
    }
}

#[derive(Debug, Clone)]
pub(super) struct Observation {
    pub pos: Vec<Coord>,
    /// k番目の要素はΣv(pi)=kとなる尤度を表す
    pub likelihoods: Vec<f64>,
    /// k番目の要素はΣv(pi)=kとなる対数尤度を表す
    pub log_likelihoods: Vec<f64>,
    likelihood_prefix_sum: Vec<f64>,
}

impl Observation {
    pub(super) fn new(pos: Vec<Coord>, sampled: i32, input: &Input) -> Self {
        assert!(pos.len() > 0);

        let likelihoods_len = pos.len() * input.oil_count + 1;
        let k = pos.len() as f64;
        let x = sampled as f64;

        // k = 1のときは特別扱い
        let log_likelihoods = if pos.len() == 1 {
            let mut log_likelihoods = vec![f64::MIN_POSITIVE.ln(); likelihoods_len];
            log_likelihoods[sampled as usize] = 0.0;
            log_likelihoods
        } else {
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

            log_likelihoods
        };

        let likelihoods = log_likelihoods.iter().copied().map(f64::exp).collect_vec();

        let mut likelihood_prefix_sum = vec![0.0];

        for i in 0..log_likelihoods.len() {
            likelihood_prefix_sum.push(likelihood_prefix_sum[i] + likelihoods[i]);
        }

        Self {
            pos,
            likelihoods,
            log_likelihoods,
            likelihood_prefix_sum,
        }
    }

    fn sum_likelihood(&self, range: Range<usize>) -> f64 {
        self.likelihood_prefix_sum[range.end] - self.likelihood_prefix_sum[range.start]
    }
}
