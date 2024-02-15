use crate::{
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
}

impl ObservationManager {
    pub(super) fn new(input: &Input) -> Self {
        let relative_observation_indices = (0..input.oil_count)
            .map(|_| Map2d::new_with(vec![], input.map_size))
            .collect();
        let observations = vec![];
        let inv_relative_observation_indices = vec![];
        let overlaps = vec![];

        Self {
            observations,
            relative_observation_indices,
            inv_relative_observation_indices,
            overlaps,
        }
    }

    pub(super) fn add_observation(&mut self, input: &Input, observation: Observation) {
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
pub(super) struct Observation {
    pub pos: Vec<Coord>,
    /// k番目の要素はΣv(pi)=kとなる対数尤度を表す
    pub log_likelihoods: Vec<f64>,
}

impl Observation {
    pub(super) fn new(pos: Vec<Coord>, sampled: i32, input: &Input) -> Self {
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
            log_likelihoods,
        }
    }
}
