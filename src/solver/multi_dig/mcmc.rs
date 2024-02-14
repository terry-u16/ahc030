use crate::{
    common::ChangeMinMax as _,
    grid::{Coord, CoordDiff, Map2d, ADJACENTS},
    problem::Input,
};
use itertools::Itertools as _;
use rand::{
    seq::{IteratorRandom as _, SliceRandom as _},
    Rng,
};
use rand_distr::{Distribution as _, WeightedAliasIndex, WeightedIndex};
use rustc_hash::FxHashMap;
use std::hash::Hash;

use super::Env;

#[derive(Debug, Clone)]
pub(super) struct State {
    pub shift: Vec<CoordDiff>,
    pub log_likelihood: f64,
    counts: Vec<usize>,
    hash: u64,
}

impl State {
    pub(super) fn new(shift: Vec<CoordDiff>, env: &Env) -> Self {
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

    pub(super) fn add_last_observation(&mut self, env: &Env) {
        assert!(self.counts.len() + 1 == env.observations.len());

        let mut count = 0;
        let overlaps = env.overlaps.last().unwrap();
        let observation = env.observations.last().unwrap();

        for (&shift, overlap) in self.shift.iter().zip(overlaps.iter()) {
            count += overlap[Coord::try_from(shift).unwrap()];
        }

        self.log_likelihood += observation.log_likelihoods[count];

        self.counts.push(count);
    }

    fn calc_log_likelihood(&self) -> f64 {
        self.log_likelihood
    }

    fn neigh1(self, env: &Env, rng: &mut impl Rng, choose_cnt: usize) -> Self {
        let mut oil_indices = (0..env.input.oil_count).choose_multiple(rng, choose_cnt);
        oil_indices.shuffle(rng);
        self.break_and_reconstruct(env, rng, &oil_indices)
    }

    fn neigh2(self, env: &Env, rng: &mut impl Rng) -> Self {
        let mut candidates = vec![];

        for (&shift, oil) in self.shift.iter().zip(env.input.oils.iter()) {
            for &p in oil.pos.iter() {
                candidates.push(p + shift);
            }
        }

        let start = candidates.choose(rng).copied().unwrap();
        let mut seen = Map2d::new_with(false, env.input.map_size);
        seen[start] = true;

        let mut stack = vec![start];

        while let Some(pos) = stack.pop() {
            for &adj in ADJACENTS.iter() {
                let next = pos + adj;

                if next.in_map(env.input.map_size) && !seen[next] {
                    seen[next] = true;
                    stack.push(next);
                }
            }
        }

        let mut target = vec![false; env.input.oil_count];

        for (i, oil) in env.input.oils.iter().enumerate() {
            for &p in oil.pos.iter() {
                if seen[p] {
                    target[i] = true;
                    break;
                }
            }
        }

        let mut oil_indices = (0..env.input.oil_count)
            .filter(|&i| target[i])
            .collect_vec();

        oil_indices.shuffle(rng);
        self.break_and_reconstruct(env, rng, &oil_indices)
    }

    fn break_and_reconstruct(
        mut self,
        env: &Env,
        rng: &mut impl Rng,
        oil_indices: &[usize],
    ) -> Self {
        // 同じ場所への配置を許可しない
        let last_shifts = self.shift.clone();
        let mut taboos = vec![false; env.input.oil_count];
        let taboo = oil_indices.choose(rng).copied().unwrap();
        if env.input.oils[taboo].height < env.input.map_size
            || env.input.oils[taboo].width < env.input.map_size
        {
            taboos[taboo] = true;
        }

        for &oil_i in oil_indices.iter() {
            let oil = &env.input.oils[oil_i];
            self.remove_oil(env, oil_i);

            // 消したままだと貪欲の判断基準がおかしくなるので、ランダムな適当な場所に置いておく
            let height = env.input.map_size - oil.height + 1;
            let width = env.input.map_size - oil.width + 1;
            let dr = rng.gen_range(0..height) as isize;
            let dc = rng.gen_range(0..width) as isize;
            let shift = CoordDiff::new(dr, dc);
            self.add_oil(env, oil_i, shift);
        }

        // ランダム性を入れた貪欲で場所を決めていく
        for &oil_i in oil_indices.iter() {
            let height = env.input.map_size - env.input.oils[oil_i].height + 1;
            let width = env.input.map_size - env.input.oils[oil_i].width + 1;
            let candidate_count = height * width;
            let mut shifts = Vec::with_capacity(candidate_count);
            let mut log_likelihoods = Vec::with_capacity(candidate_count);
            let mut max_log_likelihood = f64::NEG_INFINITY;

            // 消し直す
            self.remove_oil(env, oil_i);

            for row in 0..height {
                for col in 0..width {
                    let shift = CoordDiff::new(row as isize, col as isize);

                    // 同じ場所への配置を許可しない
                    if taboos[oil_i] && shift == last_shifts[oil_i] {
                        continue;
                    }

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

    pub(super) fn to_answer(&self, input: &Input) -> Vec<Coord> {
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

pub(super) fn mcmc(env: &Env, mut state: State, duration: f64, rng: &mut impl Rng) -> Vec<State> {
    let mut sampled_states = vec![state.clone()];

    let mut all_iter = 0;
    let mut valid_iter = 0;
    let mut accepted_count = 0;

    let duration_inv = 1.0 / duration;
    let since = std::time::Instant::now();

    let oil_count_dist = WeightedAliasIndex::new(vec![0, 10, 60, 10]).unwrap();

    loop {
        let time = (std::time::Instant::now() - since).as_secs_f64() * duration_inv;

        if time >= 1.0 {
            break;
        }

        all_iter += 1;

        // 変形
        let oil_count = oil_count_dist.sample(rng).min(env.input.oil_count);
        let new_state = if rng.gen_bool(0.9) {
            state.clone().neigh1(env, rng, oil_count)
        } else {
            state.clone().neigh2(env, rng)
        };

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
