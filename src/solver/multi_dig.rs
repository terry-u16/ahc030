use itertools::Itertools;
use rand::{
    seq::{IteratorRandom, SliceRandom},
    Rng,
};
use rand_core::SeedableRng;
use rand_distr::{Distribution, WeightedAliasIndex, WeightedIndex};
use rand_pcg::Pcg64Mcg;
use rustc_hash::FxHashMap;
use std::hash::Hash;
use std::vec;

use crate::{
    common::ChangeMinMax,
    distributions::GaussianDistribution,
    grid::{Coord, CoordDiff, Map2d},
    problem::{Input, Judge},
};

use super::Solver;

pub struct MultiDigSolver {
    judge: Judge,
}

impl MultiDigSolver {
    pub fn new(judge: Judge) -> Self {
        Self { judge }
    }
}

impl Solver for MultiDigSolver {
    fn solve(&mut self, input: &crate::problem::Input) {
        let mut env = Env::new(input);
        let all_coords = (0..input.map_size)
            .flat_map(|row| (0..input.map_size).map(move |col| Coord::new(row, col)))
            .collect_vec();
        let mut rng = Pcg64Mcg::from_entropy();
        let mcmc_duration = 2.0 / ((input.map_size as f64).powi(2) * 2.0);
        let mut next_start = vec![CoordDiff::new(0, 0); input.oil_count];

        for _ in 0..input.map_size * input.map_size * 2 {
            let state = State::new(next_start, &env);

            let mut states = mcmc(&env, state, mcmc_duration, &mut rng);
            states.sort_unstable();
            states.dedup();
            states
                .sort_unstable_by(|a, b| b.log_likelihood.partial_cmp(&a.log_likelihood).unwrap());

            let state = &states[0];
            next_start = state.shift.clone();
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

            if ratio >= 100.0 {
                if self.judge.answer(&state.to_answer(input)).is_ok() {
                    return;
                }
            }

            let count = rng.gen_range(
                input.map_size * input.map_size / 4..=input.map_size * input.map_size / 2,
            );
            let targets = all_coords
                .choose_multiple(&mut rng, count)
                .copied()
                .collect_vec();
            let sampled = self.judge.query_multiple(&targets);
            let observation = Observation::new(targets, sampled, input);
            env.add_observation(input, observation);
        }
    }
}

#[derive(Debug, Clone)]
struct Env<'a> {
    input: &'a Input,
    observations: Vec<Observation>,
    relative_observation_indices: Vec<Map2d<Vec<(usize, usize)>>>,
    inv_relative_observation_indices: Vec<Vec<(usize, usize, CoordDiff)>>,
}

impl<'a> Env<'a> {
    fn new(input: &'a Input) -> Self {
        let relative_observation_indices = (0..input.oil_count)
            .map(|_| Map2d::new_with(vec![], input.map_size))
            .collect();
        let observations = vec![];
        let inv_relative_observation_indices = vec![];

        Self {
            input,
            observations,
            relative_observation_indices,
            inv_relative_observation_indices,
        }
    }

    fn add_observation(&mut self, input: &Input, observation: Observation) {
        let obs_id = self.observations.len();
        let mut observed_map = Map2d::new_with(false, input.map_size);

        for &p in observation.pos.iter() {
            observed_map[p] = true;
        }

        let mut inv_relative_observation_indices = vec![];

        for (oil_i, oil) in input.oils.iter().enumerate() {
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

                    if count > 0 {
                        self.relative_observation_indices[oil_i][c].push((obs_id, count));
                        inv_relative_observation_indices.push((oil_i, count, shift));
                    }
                }
            }
        }

        self.inv_relative_observation_indices
            .push(inv_relative_observation_indices);
        self.observations.push(observation);
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
        let likelihoods_len = pos.len() * input.oil_count + 1;
        let mut log_likelihoods = Vec::with_capacity(likelihoods_len);
        let k = pos.len() as f64;
        let x = sampled as f64;

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

#[derive(Debug, Clone)]
struct State {
    shift: Vec<CoordDiff>,
    log_likelihood: f64,
    counts: Vec<usize>,
    hash: u64,
}

impl State {
    fn new(shift: Vec<CoordDiff>, env: &Env) -> Self {
        let mut log_likelihood = 0.0;
        let counts = vec![0; env.observations.len()];

        for (obs, &count) in env.observations.iter().zip(counts.iter()) {
            log_likelihood += obs.log_likelihoods[count];
        }

        let mut hash = 0;

        for (i, shift) in shift.iter().enumerate() {
            hash ^= env.input.hashes[i][Coord::try_from(*shift).unwrap()];
        }

        let mut state = Self {
            shift,
            log_likelihood,
            counts,
            hash,
        };

        let shift = state.shift.clone();

        for (i, &shift) in shift.iter().enumerate() {
            state.add_oil(env, i, shift);
        }

        state
    }

    fn add_oil(&mut self, env: &Env, oil_i: usize, shift: CoordDiff) {
        self.shift[oil_i] = shift;
        let indices = &env.relative_observation_indices[oil_i][Coord::try_from(shift).unwrap()];
        self.hash ^= env.input.hashes[oil_i][Coord::try_from(shift).unwrap()];

        for &(obs_i, cnt) in indices {
            let observation = &env.observations[obs_i];
            let count = &mut self.counts[obs_i];
            self.log_likelihood -= observation.log_likelihoods[*count];
            *count += cnt;
            self.log_likelihood += observation.log_likelihoods[*count];
        }
    }

    fn remove_oil(&mut self, env: &Env, oil_i: usize) {
        let shift = self.shift[oil_i];
        let indices = &env.relative_observation_indices[oil_i][Coord::try_from(shift).unwrap()];
        self.hash ^= env.input.hashes[oil_i][Coord::try_from(shift).unwrap()];

        for &(obs_i, cnt) in indices.iter() {
            let observation = &env.observations[obs_i];
            let count = &mut self.counts[obs_i];
            self.log_likelihood -= observation.log_likelihoods[*count];
            *count -= cnt;
            self.log_likelihood += observation.log_likelihoods[*count];
        }
    }

    fn add_oil_whatif(&self, env: &Env, oil_i: usize, shift: CoordDiff) -> f64 {
        let indices = &env.relative_observation_indices[oil_i][Coord::try_from(shift).unwrap()];
        let mut log_likelihood = self.log_likelihood;

        for &(obs_i, cnt) in indices {
            let observation = &env.observations[obs_i];
            let mut count = self.counts[obs_i];
            log_likelihood -= observation.log_likelihoods[count];
            count += cnt;
            log_likelihood += observation.log_likelihoods[count];
        }

        log_likelihood
    }

    fn calc_log_likelihood(&self) -> f64 {
        self.log_likelihood
    }

    fn neigh(mut self, env: &Env, rng: &mut impl Rng, choose_cnt: usize) -> Self {
        let mut oil_indices = (0..env.input.oil_count).choose_multiple(rng, choose_cnt);
        oil_indices.shuffle(rng);

        for &oil_i in oil_indices.iter() {
            self.remove_oil(env, oil_i);
        }

        // ランダム性を入れた貪欲で場所を決めていく
        for &oil_i in oil_indices.iter() {
            let height = env.input.map_size - env.input.oils[oil_i].height + 1;
            let width = env.input.map_size - env.input.oils[oil_i].width + 1;
            let candidate_count = height * width;
            let mut shifts = Vec::with_capacity(candidate_count);
            let mut log_likelihoods = Vec::with_capacity(candidate_count);
            let mut max_log_likelihood = f64::NEG_INFINITY;

            for row in 0..height {
                for col in 0..width {
                    let shift = CoordDiff::new(row as isize, col as isize);
                    let log_likelihood = self.add_oil_whatif(env, oil_i, shift);
                    shifts.push(shift);
                    log_likelihoods.push(log_likelihood);
                    max_log_likelihood.change_max(log_likelihood);
                }
            }

            let likelihoods = log_likelihoods
                .iter()
                .map(|&log_likelihood| f64::exp(log_likelihood - max_log_likelihood))
                .collect_vec();

            let dist = WeightedIndex::new(likelihoods).unwrap();
            let sample_i = dist.sample(rng);
            let shift = shifts[sample_i];

            self.add_oil(env, oil_i, shift);
        }

        self.normalize(&env.input);

        self
    }

    fn normalize(&mut self, input: &Input) {
        let mut groups = FxHashMap::default();

        for (i, shift) in self.shift.iter().enumerate() {
            self.hash ^= input.hashes[i][Coord::try_from(*shift).unwrap()];
        }

        for (i, (oil, shift)) in input.oils.iter().zip(&self.shift).enumerate() {
            let entry = groups.entry(&oil.pos).or_insert_with(|| vec![]);
            entry.push((i, shift));
        }

        let mut new_shift = vec![CoordDiff::new(0, 0); input.oil_count];

        for group in groups.values() {
            let mut shifts = group.iter().map(|&(_, shift)| shift).collect_vec();
            shifts.sort_unstable();

            for (i, shift) in group.iter().map(|&(i, _)| i).zip(shifts) {
                new_shift[i] = *shift;
            }
        }

        self.shift = new_shift;

        for (i, shift) in self.shift.iter().enumerate() {
            self.hash ^= input.hashes[i][Coord::try_from(*shift).unwrap()];
        }
    }

    fn to_answer(&self, input: &Input) -> Vec<Coord> {
        let mut map = Map2d::new_with(0, input.map_size);

        for (oil, &shift) in input.oils.iter().zip(self.shift.iter()) {
            for &p in oil.pos.iter() {
                let pos = p + shift;

                map[pos] = 1;
            }
        }

        let mut results = vec![];

        for row in 0..input.map_size {
            for col in 0..input.map_size {
                let c = Coord::new(row, col);

                if map[c] != 0 {
                    results.push(c);
                }
            }
        }

        results
    }
}

impl Hash for State {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.hash.hash(state);
    }
}

impl PartialEq for State {
    fn eq(&self, other: &Self) -> bool {
        self.hash == other.hash
    }
}

impl Eq for State {}

impl PartialOrd for State {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.hash.partial_cmp(&other.hash)
    }
}

impl Ord for State {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.hash.cmp(&other.hash)
    }
}

fn mcmc(env: &Env, mut state: State, duration: f64, rng: &mut impl Rng) -> Vec<State> {
    let mut sampled_states = vec![state.clone()];

    let mut all_iter = 0;
    let mut valid_iter = 0;
    let mut accepted_count = 0;

    let duration_inv = 1.0 / duration;
    let since = std::time::Instant::now();

    let oil_count_dist = WeightedAliasIndex::new(vec![10, 60, 20, 10]).unwrap();

    loop {
        all_iter += 1;
        if (all_iter & ((1 << 4) - 1)) == 0 {
            let time = (std::time::Instant::now() - since).as_secs_f64() * duration_inv;

            if time >= 1.0 {
                break;
            }
        }

        // 変形
        let oil_count = (oil_count_dist.sample(rng) + 1).min(env.input.oil_count);
        let new_state = state.clone().neigh(env, rng, oil_count);

        // スコア計算
        let log_likelihood_diff = new_state.calc_log_likelihood() - state.calc_log_likelihood();

        if log_likelihood_diff >= 0.0 || rng.gen_bool(f64::exp(log_likelihood_diff)) {
            // サンプリングされた解を保存
            // 解の集合が分かれば良いので、遷移しなかったときは不要
            state = new_state;
            sampled_states.push(state.clone());
            accepted_count += 1;
        }

        valid_iter += 1;
    }

    eprintln!(
        "all_iter: {} valid_iter: {} accepted_count: {}",
        all_iter, valid_iter, accepted_count
    );

    sampled_states
}
