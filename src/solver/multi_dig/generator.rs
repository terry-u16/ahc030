use crate::{
    common::ChangeMinMax as _,
    grid::{Coord, CoordDiff, Map2d, ADJACENTS},
    problem::Input,
};
use itertools::{izip, Itertools as _};
use ordered_float::OrderedFloat;
use rand::{
    seq::{IteratorRandom as _, SliceRandom as _},
    Rng,
};
use rand_distr::{Distribution as _, WeightedAliasIndex, WeightedIndex};
use rustc_hash::{FxHashMap, FxHashSet};
use std::{cmp::Reverse, hash::Hash};

use super::Env;

#[derive(Debug, Clone)]
pub(super) struct State {
    pub shift: Vec<CoordDiff>,
    pub log_likelihood: f64,
    counts: Vec<usize>,
    counts_u32: Vec<u32>,
    pub hash: u64,
}

impl State {
    pub(super) fn new(shift: Vec<CoordDiff>, env: &Env) -> Self {
        let mut log_likelihood = 0.0;
        let counts = vec![0; env.obs.observations.len()];
        let counts_i32 = vec![0; env.obs.observations.len()];

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
            counts_u32: counts_i32,
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
            self.counts_u32[obs_i] = *count as u32;
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
            self.counts_u32[obs_i] = *count as u32;
        }
    }

    #[target_feature(enable = "avx,avx2")]
    unsafe fn add_oil_whatif(&self, env: &Env, oil_i: usize, shift: CoordDiff) -> f64 {
        use std::arch::x86_64::*;

        // AVX2を使って高速化
        let log_likelihoods = &env.obs.obs_log_likelihoods;
        let mut offsets: &[u32] = &env.obs.obs_log_likelihoods_offsets;
        let mut cnts: &[u32] = &self.counts_u32;

        let mut cnts_added: &[u32] =
            &env.obs.relative_observation_cnt_u32[oil_i][Coord::try_from(shift).unwrap()];
        let mut log_likelihood = _mm256_broadcast_ss(&0.0);

        const ELEMENTS_PER_256BIT: usize = 8;

        while offsets.len() >= ELEMENTS_PER_256BIT {
            // j番目の観測が保存されている箇所の先頭のindex
            let offset = _mm256_loadu_si256(offsets.as_ptr() as *const __m256i);

            // j番目の観測について、現在の配置だと仮定したときの真の埋蔵量
            let cnt = _mm256_loadu_si256(cnts.as_ptr() as *const __m256i);

            // oil_iを追加したときの真の埋蔵量の増加量
            let cnt_added = _mm256_loadu_si256(cnts_added.as_ptr() as *const __m256i);

            // offset, cnt, cnt_addedを足す
            let offset = _mm256_add_epi32(offset, cnt);
            let offset = _mm256_add_epi32(offset, cnt_added);

            // log_likelihoodsからoffsetの位置にある値を取得
            let likelihoods = _mm256_i32gather_ps::<4>(log_likelihoods.as_ptr(), offset);

            // 取得した値をlog_likelihoodに足す
            log_likelihood = _mm256_add_ps(log_likelihood, likelihoods);

            // ポインタを進める（float型は256bitレジスタに8個入る）
            offsets = offsets.get_unchecked(ELEMENTS_PER_256BIT..);
            cnts = cnts.get_unchecked(ELEMENTS_PER_256BIT..);
            cnts_added = cnts_added.get_unchecked(ELEMENTS_PER_256BIT..);
        }

        // TODO: 水平加算を理解する
        let mut buffer = [0.0; ELEMENTS_PER_256BIT];
        _mm256_storeu_ps(buffer.as_mut_ptr(), log_likelihood);
        let mut log_likelihood = 0.0;

        for &v in buffer.iter() {
            log_likelihood += v;
        }

        for (&offset, &cnt, &cnt_added) in izip!(offsets, cnts, cnts_added) {
            let index = offset + cnt + cnt_added;
            log_likelihood += log_likelihoods.get_unchecked(index as usize);
        }

        log_likelihood as f64
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
        self.counts_u32.push(count as u32);
    }

    fn neigh(mut self, env: &Env, rng: &mut impl Rng, choose_cnt: usize) -> Self {
        let mut oil_indices = (0..env.input.oil_count).choose_multiple(rng, choose_cnt);
        oil_indices.shuffle(rng);

        // 同じ場所への配置を許可しない
        let last_shifts = self.shift.clone();
        let mut taboos = vec![false; env.input.oil_count];

        if rng.gen_bool(env.input.params.taboo_prob) {
            let taboo = oil_indices.choose(rng).copied().unwrap();
            if env.obs.shift_candidates[taboo].len() > 1 {
                taboos[taboo] = true;
            }
        }

        for &oil_i in oil_indices.iter() {
            self.remove_oil(env, oil_i);

            // 消したままだと貪欲の判断基準がおかしくなるので、ランダムな適当な場所に置いておく
            let shift = env.obs.shift_candidates[oil_i]
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

            for &shift in env.obs.shift_candidates[oil_i].iter() {
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
    if duration <= 0.0 {
        return states;
    }

    let mut hashes = FxHashSet::default();

    for state in states.iter() {
        hashes.insert(state.hash);
    }

    let mut state = states
        .iter()
        .max_by_key(|s| OrderedFloat(s.log_likelihood))
        .unwrap()
        .clone();
    let mut best_score = state.log_likelihood;

    let mut all_iter = 0;
    let mut valid_iter = 0;
    let mut accepted_count = 0;

    let duration_inv = 1.0 / duration;
    let since = std::time::Instant::now();

    let temp0 = 2.0;
    let temp1 = 1.0;
    let mut inv_temp = 1.0 / temp0;

    loop {
        if (all_iter & ((1 << 4) - 1)) == 0 {
            let time = (std::time::Instant::now() - since).as_secs_f64() * duration_inv;
            let temp = f64::powf(temp0, 1.0 - time) * f64::powf(temp1, time);
            inv_temp = 1.0 / temp;

            if time >= 1.0 {
                break;
            }
        }

        all_iter += 1;

        let prev_score = state.log_likelihood;

        // 変形
        let Some(neigh) = (match rng.gen_range(0..10) {
            0..=3 => ShiftNeigh::gen(env, &state, rng),
            4..=7 => MoveNeigh::gen(env, &state, rng),
            _ => SwapNeigh::gen(env, &state, rng),
        }) else {
            continue;
        };

        neigh.apply(env, &mut state);

        let score_diff = state.log_likelihood - prev_score;

        if state.log_likelihood - best_score >= -10.0 && hashes.insert(state.hash) {
            states.push(state.clone());
        }

        if score_diff >= 0.0 || rng.gen_bool(f64::exp(score_diff as f64 * inv_temp)) {
            accepted_count += 1;
            best_score.change_max(state.log_likelihood);
        } else {
            neigh.rollback(env, &mut state);
        }

        valid_iter += 1;
    }

    eprintln!(
        "[Generator] all_iter: {} valid_iter: {} accepted_count: {}",
        all_iter, valid_iter, accepted_count
    );

    states.sort_unstable_by_key(|s| Reverse(OrderedFloat(s.log_likelihood)));

    if states.len() > 5000 {
        states.truncate(5000);
    }

    states
}

trait Neigh {
    fn apply(&self, env: &Env, state: &mut State);

    fn rollback(&self, env: &Env, state: &mut State);
}

struct ShiftNeigh {
    oil_i: usize,
    old_shift: CoordDiff,
    new_shift: CoordDiff,
}

impl ShiftNeigh {
    fn gen(env: &Env, state: &State, rng: &mut impl Rng) -> Option<Box<dyn Neigh>> {
        let oil_i = rng.gen_range(0..env.input.oil_count);
        let shift = ADJACENTS.choose(rng).copied().unwrap();

        let next = state.shift[oil_i] + shift;

        if next.dr >= 0
            && next.dc >= 0
            && env.input.oils[oil_i].height.wrapping_add_signed(next.dr) <= env.input.map_size
            && env.input.oils[oil_i].width.wrapping_add_signed(next.dc) <= env.input.map_size
        {
            Some(Box::new(Self {
                oil_i,
                old_shift: state.shift[oil_i],
                new_shift: next,
            }))
        } else {
            None
        }
    }
}

impl Neigh for ShiftNeigh {
    fn apply(&self, env: &Env, state: &mut State) {
        state.remove_oil(env, self.oil_i);
        state.add_oil(env, self.oil_i, self.new_shift);
        state.normalize(&env.input);
    }

    fn rollback(&self, env: &Env, state: &mut State) {
        state.remove_oil(env, self.oil_i);
        state.add_oil(env, self.oil_i, self.old_shift);
        state.normalize(&env.input);
    }
}

struct MoveNeigh {
    oil_i: usize,
    old_shift: CoordDiff,
    new_shift: CoordDiff,
}

impl MoveNeigh {
    fn gen(env: &Env, state: &State, rng: &mut impl Rng) -> Option<Box<dyn Neigh>> {
        let oil_i = rng.gen_range(0..env.input.oil_count);
        let dr = rng.gen_range(0..=env.input.map_size - env.input.oils[oil_i].height);
        let dc = rng.gen_range(0..=env.input.map_size - env.input.oils[oil_i].width);

        Some(Box::new(Self {
            oil_i,
            old_shift: state.shift[oil_i],
            new_shift: CoordDiff::new(dr as isize, dc as isize),
        }))
    }
}

impl Neigh for MoveNeigh {
    fn apply(&self, env: &Env, state: &mut State) {
        state.remove_oil(env, self.oil_i);
        state.add_oil(env, self.oil_i, self.new_shift);
        state.normalize(&env.input);
    }

    fn rollback(&self, env: &Env, state: &mut State) {
        state.remove_oil(env, self.oil_i);
        state.add_oil(env, self.oil_i, self.old_shift);
        state.normalize(&env.input);
    }
}

struct SwapNeigh {
    oil0: usize,
    oil1: usize,
    old_shift0: CoordDiff,
    new_shift0: CoordDiff,
    old_shift1: CoordDiff,
    new_shift1: CoordDiff,
}

impl SwapNeigh {
    fn gen(env: &Env, state: &State, rng: &mut impl Rng) -> Option<Box<dyn Neigh>> {
        let oil0 = rng.gen_range(0..env.input.oil_count);
        let oil1 = rng.gen_range(0..env.input.oil_count);

        if oil0 == oil1 {
            return None;
        }

        let shift0 = state.shift[oil1]
            + env.swap_candidates[oil1][oil0]
                .choose(rng)
                .copied()
                .unwrap();
        let shift1 = state.shift[oil0]
            + env.swap_candidates[oil0][oil1]
                .choose(rng)
                .copied()
                .unwrap();

        if shift0.dr >= 0
            && shift0.dc >= 0
            && shift1.dr >= 0
            && shift1.dc >= 0
            && env.input.oils[oil0].height.wrapping_add_signed(shift0.dr) <= env.input.map_size
            && env.input.oils[oil0].width.wrapping_add_signed(shift0.dc) <= env.input.map_size
            && env.input.oils[oil1].height.wrapping_add_signed(shift1.dr) <= env.input.map_size
            && env.input.oils[oil1].width.wrapping_add_signed(shift1.dc) <= env.input.map_size
        {
            Some(Box::new(Self {
                oil0,
                oil1,
                old_shift0: state.shift[oil0],
                new_shift0: shift0,
                old_shift1: state.shift[oil1],
                new_shift1: shift1,
            }))
        } else {
            None
        }
    }
}

impl Neigh for SwapNeigh {
    fn apply(&self, env: &Env, state: &mut State) {
        state.remove_oil(env, self.oil0);
        state.remove_oil(env, self.oil1);
        state.add_oil(env, self.oil0, self.new_shift0);
        state.add_oil(env, self.oil1, self.new_shift1);
        state.normalize(&env.input);
    }

    fn rollback(&self, env: &Env, state: &mut State) {
        state.remove_oil(env, self.oil0);
        state.remove_oil(env, self.oil1);
        state.add_oil(env, self.oil0, self.old_shift0);
        state.add_oil(env, self.oil1, self.old_shift1);
        state.normalize(&env.input);
    }
}
