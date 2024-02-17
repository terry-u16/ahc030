use crate::{
    common::ChangeMinMax as _,
    grid::{Coord, CoordDiff, Map2d},
    problem::Input,
};
use itertools::Itertools as _;
use ordered_float::OrderedFloat;
use rand::{
    seq::{IteratorRandom as _, SliceRandom as _},
    Rng,
};
use rand_distr::{Distribution as _, WeightedAliasIndex, WeightedIndex};
use rustc_hash::{FxHashMap, FxHashSet};
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
        let counts = vec![0; env.obs.observations.len()];

        for (obs, &count) in env.obs.observations.iter().zip(counts.iter()) {
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
        let cnt = &env.obs.relative_observation_cnt[oil_i][Coord::try_from(shift).unwrap()];
        self.hash ^= env.input.hashes[oil_i][Coord::try_from(shift).unwrap()];

        for (obs_i, &cnt) in cnt.iter().enumerate() {
            let observation = &env.obs.observations[obs_i];
            let count = &mut self.counts[obs_i];
            self.log_likelihood -= observation.log_likelihoods[*count];
            *count += cnt;
            self.log_likelihood += observation.log_likelihoods[*count];
        }
    }

    fn remove_oil(&mut self, env: &Env, oil_i: usize) {
        let shift = self.shift[oil_i];
        let cnt = &env.obs.relative_observation_cnt[oil_i][Coord::try_from(shift).unwrap()];
        self.hash ^= env.input.hashes[oil_i][Coord::try_from(shift).unwrap()];

        for (obs_i, &cnt) in cnt.iter().enumerate() {
            let observation = &env.obs.observations[obs_i];
            let count = &mut self.counts[obs_i];
            self.log_likelihood -= observation.log_likelihoods[*count];
            *count -= cnt;
            self.log_likelihood += observation.log_likelihoods[*count];
        }
    }

    #[target_feature(
        enable = "aes,avx,avx2,bmi1,bmi2,fma,fxsr,pclmulqdq,popcnt,rdrand,rdseed,sse,sse2,sse4.1,sse4.2,ssse3,xsave,xsavec,xsaveopt,xsaves"
    )]
    unsafe fn add_oil_whatif(&self, env: &Env, oil_i: usize, shift: CoordDiff) -> f64 {
        let cnt = &env.obs.relative_observation_cnt[oil_i][Coord::try_from(shift).unwrap()];
        let mut log_likelihood = 0.0;

        for (observation, (&count, &cnt)) in env
            .obs
            .obs_log_likelihoods
            .iter()
            .zip(self.counts.iter().zip(cnt.iter()))
        {
            log_likelihood += unsafe { observation.get_unchecked(count + cnt) };
        }

        log_likelihood
    }

    pub(super) fn add_last_observation(&mut self, env: &Env) {
        assert!(self.counts.len() + 1 == env.obs.observations.len());

        let mut count = 0;
        let overlaps = env.obs.overlaps.last().unwrap();
        let observation = env.obs.observations.last().unwrap();

        for (&shift, overlap) in self.shift.iter().zip(overlaps.iter()) {
            count += overlap[Coord::try_from(shift).unwrap()];
        }

        self.log_likelihood += observation.log_likelihoods[count];

        self.counts.push(count);
    }

    fn neigh(mut self, env: &Env, rng: &mut impl Rng, choose_cnt: usize) -> Self {
        let mut oil_indices = (0..env.input.oil_count).choose_multiple(rng, choose_cnt);
        oil_indices.shuffle(rng);

        let cand_lens = env
            .obs
            .shift_candidates
            .iter()
            .map(|c| {
                let len = c.len();
                let ratio = rng.gen_range(0.3..=1.0);
                let cand_ren = ((len as f64 * ratio * ratio).ceil() as usize)
                    .max(5)
                    .min(len);
                cand_ren
            })
            .collect_vec();

        // 同じ場所への配置を許可しない
        let last_shifts = self.shift.clone();
        let mut taboos = vec![false; env.input.oil_count];

        if rng.gen_bool(0.5) {
            let taboo = oil_indices.choose(rng).copied().unwrap();
            if env.obs.shift_candidates[taboo].len() > 1 {
                taboos[taboo] = true;
            }
        }

        for &oil_i in oil_indices.iter() {
            self.remove_oil(env, oil_i);

            // 消したままだと貪欲の判断基準がおかしくなるので、ランダムな適当な場所に置いておく
            let shift = env.obs.shift_candidates[oil_i][..cand_lens[oil_i]]
                .choose(rng)
                .copied()
                .unwrap();
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

            for &shift in env.obs.shift_candidates[oil_i][..cand_lens[oil_i]].iter() {
                // 同じ場所への配置を許可しない
                if taboos[oil_i] && shift == last_shifts[oil_i] {
                    continue;
                }

                let log_likelihood = unsafe { self.add_oil_whatif(env, oil_i, shift) };
                shifts.push(shift);
                log_likelihoods.push(log_likelihood);
                max_log_likelihood.change_max(log_likelihood);
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

pub(super) fn generate_states(
    env: &Env,
    mut states: Vec<State>,
    duration: f64,
    rng: &mut impl Rng,
) -> Vec<State> {
    let mut hashes = FxHashSet::default();

    for state in states.iter() {
        hashes.insert(state.hash);
    }

    let base_log_likelihood = states
        .iter()
        .map(|s| s.log_likelihood)
        .fold(f64::MIN, f64::max);

    let mut prefix_prob = vec![OrderedFloat(
        (states[0].log_likelihood - base_log_likelihood).exp(),
    )];

    for i in 1..states.len() {
        let p = OrderedFloat(
            (states[i].log_likelihood - base_log_likelihood).exp() + prefix_prob[i - 1].0,
        );
        prefix_prob.push(p);
    }

    let mut all_iter = 0;
    let mut valid_iter = 0;
    let mut accepted_count = 0;

    let duration_inv = 1.0 / duration;
    let since = std::time::Instant::now();

    let oil_count_dist = WeightedAliasIndex::new(vec![0, 0, 60, 20, 10, 5]).unwrap();

    loop {
        let time = env.input.duration_corrector.elapsed(since).as_secs_f64() * duration_inv;

        if time >= 1.0 {
            break;
        }

        all_iter += 1;

        // 変形
        let x = rng.gen_range(0.0..prefix_prob.last().unwrap().0);
        let index = prefix_prob
            .binary_search(&OrderedFloat(x))
            .unwrap_or_else(|x| x);
        let state = states[index].clone();
        let oil_count = oil_count_dist.sample(rng).min(env.input.oil_count);
        let new_state = state.neigh(env, rng, oil_count);

        if hashes.insert(new_state.hash) {
            // 凄く大きな値を引いてしまうとオーバーフローする可能性があるため注意
            // 適切にサンプリングできればよいので、重みは適当に上限を設ける
            let mut p = (new_state.log_likelihood - base_log_likelihood).exp()
                + prefix_prob.last().unwrap().0;
            p.change_min(1e200);
            prefix_prob.push(OrderedFloat(p));
            states.push(new_state);

            accepted_count += 1;
        }

        valid_iter += 1;
    }

    //eprintln!(
    //    "all_iter: {} valid_iter: {} accepted_count: {}",
    //    all_iter, valid_iter, accepted_count
    //);

    states
}
