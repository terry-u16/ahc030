use crate::{
    common::ChangeMinMax,
    distributions::GaussianDistribution,
    grid::{Coord, CoordDiff, Map2d},
    problem::Input,
};
use itertools::Itertools;

#[derive(Debug, Clone)]
pub(super) struct ObservationManager {
    pub observations: Vec<Observation>,
    pub obs_log_likelihoods: Vec<f32>,
    pub obs_log_likelihoods_offsets: Vec<u32>,
    /// indices[oil_i][shift] := 影響するobs_iのVec
    pub relative_observation_indices: Vec<Map2d<Vec<usize>>>,
    /// indices[oil_i][shift] := 影響するcountのVec
    pub relative_observation_cnt: Vec<Map2d<Vec<usize>>>,
    pub relative_observation_cnt_u32: Vec<Map2d<Vec<u32>>>,
    /// indices[obs_i] := 影響する (oil_i, count, shift) のVec
    pub inv_relative_observation_indices: Vec<Vec<(usize, usize, CoordDiff)>>,
    /// matrix[obs_i][oil_i][shift] := oil_iをshiftだけ動かした領域とobs_iの重なりの数
    pub overlaps: Vec<Vec<Map2d<usize>>>,
    pub shift_candidates: Vec<Vec<CoordDiff>>,
    /// matrix[obs_i][oil_i] := oil_i以外の油田とobs_iとの重なりの最小値と最大値
    overlap_min_max: Vec<Vec<(usize, usize)>>,
}

impl ObservationManager {
    pub(super) fn new(input: &Input) -> Self {
        let relative_observation_indices = (0..input.oil_count)
            .map(|_| Map2d::new_with(vec![], input.map_size))
            .collect();
        let relative_observation_cnt = (0..input.oil_count)
            .map(|_| Map2d::new_with(vec![], input.map_size))
            .collect();
        let relative_observation_cnt_u32 = (0..input.oil_count)
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

        let overlap_min_max = vec![];

        Self {
            observations,
            obs_log_likelihoods: vec![],
            obs_log_likelihoods_offsets: vec![],
            relative_observation_indices,
            relative_observation_cnt,
            relative_observation_cnt_u32,
            inv_relative_observation_indices,
            overlaps,
            shift_candidates,
            overlap_min_max,
        }
    }

    pub(super) fn add_observation(&mut self, input: &Input, observation: Observation) {
        let obs_id = self.observations.len();

        self.obs_log_likelihoods_offsets
            .push(self.obs_log_likelihoods.len() as u32);

        for &v in observation.log_likelihoods.iter() {
            self.obs_log_likelihoods.push(v as f32);
        }

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

            let relative_observation_indices = &mut self.relative_observation_indices[oil_i];
            let relative_observation_cnt = &mut self.relative_observation_cnt[oil_i];
            let relative_observation_cnt_u32 = &mut self.relative_observation_cnt_u32[oil_i];

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
                    relative_observation_indices[c].push(obs_id);
                    relative_observation_cnt[c].push(count);
                    relative_observation_cnt_u32[c].push(count as u32);
                    inv_relative_observation_indices.push((oil_i, count, shift));
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
    }
}

#[derive(Debug, Clone)]
pub(super) struct Observation {
    pub pos: Vec<Coord>,
    /// k番目の要素はΣv(pi)=kとなる対数尤度を表す
    pub log_likelihoods: Vec<f64>,
}

impl Observation {
    pub(super) fn new(pos: Vec<Coord>, sampled: i32, input: &Input) -> Self {
        assert!(pos.len() > 0);

        let likelihoods_len = input.total_oil_tiles + 1;
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
                let (mean, variance) = if pos.len() == 1 {
                    (v as f64, 1e-5)
                } else {
                    let mean = (k - v) * input.eps + v * (1.0 - input.eps);
                    let variance = k * input.eps * (1.0 - input.eps);
                    (mean, variance)
                };
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
            log_likelihoods,
        }
    }
}
