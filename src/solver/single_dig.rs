use itertools::Itertools as _;
use ordered_float::OrderedFloat;
use rand::{
    seq::{IteratorRandom, SliceRandom as _},
    Rng,
};
use rand_core::SeedableRng as _;
use rand_distr::{Distribution as _, WeightedAliasIndex};
use rand_pcg::Pcg64Mcg;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::{
    common::ChangeMinMax as _,
    grid::{Coord, CoordDiff, Map2d},
    problem::{Input, Judge},
};

use super::Solver;

pub struct SingleDigSolver<'a> {
    judge: Judge<'a>,
}

impl<'a> SingleDigSolver<'a> {
    pub fn new(judge: Judge<'a>) -> Self {
        Self { judge }
    }
}

impl<'a> Solver for SingleDigSolver<'a> {
    fn solve(&mut self, input: &crate::problem::Input) {
        let mut env = Env::new(input);
        let mut rng = Pcg64Mcg::from_entropy();

        let elapsed = input.duration_corrector.elapsed(input.since);
        let time_table = input.time_conductor.get_time_table(
            elapsed,
            Input::TIME_LIMIT,
            self.judge.max_query_count(),
        );

        let mut states = vec![State::new(
            vec![CoordDiff::new(0, 0); input.oil_count],
            env.map.clone(),
            &env,
        )];

        for turn in 0..input.map_size * input.map_size {
            let time_limit_turn = time_table[turn];
            let duration =
                time_limit_turn.saturating_sub(input.duration_corrector.elapsed(input.since));
            states = generate_states(&env, states, duration.as_secs_f64(), &mut rng);
            states.sort_unstable();
            states.dedup();

            let min_violation = states.iter().map(|s| s.violations).min().unwrap();
            states.retain(|s| s.violations == min_violation);
            eprintln!("turn: {}, states: {}", turn, states.len());

            color_map(&states, &mut rng, &env, &mut self.judge);

            if states.len() == 1 {
                let state = states.iter().next().unwrap();

                if state.calc_score() == 0 {
                    let answer = state.to_answer(input);

                    if self.judge.answer(&answer).is_ok() {
                        return;
                    }
                }
            }

            let next_pos = choose_next_pos(&states, &mut rng, input, &env, &mut self.judge);

            let observation = self.judge.query_single(next_pos);
            env.add_observation(next_pos, observation);

            for s in states.iter_mut() {
                s.add_observation(&env, next_pos, observation);
            }

            let min_violation = states.iter().map(|s| s.violations).min().unwrap();
            states.retain(|s| s.violations == min_violation);
        }

        let mut answer = vec![];

        for row in 0..input.map_size {
            for col in 0..input.map_size {
                let c = Coord::new(row, col);

                if env.map[c].unwrap() > 0 {
                    answer.push(c);
                }
            }
        }

        self.judge.answer(&answer).unwrap();
    }
}

fn choose_next_pos(
    solutions: &[State],
    rng: &mut rand_pcg::Mcg128Xsl64,
    input: &Input,
    env: &Env<'_>,
    judge: &mut Judge,
) -> Coord {
    let mut maps = vec![];
    let sample_count = 200.min(solutions.len());
    let mut states = solutions.choose_multiple(rng, sample_count);

    if states.len() == 1 {
        let mut map = Map2d::new_with(0, input.map_size);
        let state = states.next().unwrap();

        for (&shift, oil) in state.shift.iter().zip(input.oils.iter()) {
            for &p in oil.pos.iter() {
                let p = p + shift;

                map[p] += 1;
            }
        }

        let mut exist = vec![];
        let mut not_exist = vec![];

        for row in 0..input.map_size {
            for col in 0..input.map_size {
                let c = Coord::new(row, col);

                if env.map[c].is_some() {
                    continue;
                }

                if map[c] > 0 {
                    exist.push(c);
                } else {
                    not_exist.push(c);
                }
            }
        }

        // ブロックがある場所の面積の方が小さいので、そのような場所から選ぶことで早く間違いに気付かせる
        return exist
            .choose(rng)
            .copied()
            .unwrap_or_else(|| not_exist.choose(rng).copied().unwrap());
    }

    for state in states {
        let mut map = Map2d::new_with(0, input.map_size);

        for (&shift, oil) in state.shift.iter().zip(input.oils.iter()) {
            for &p in oil.pos.iter() {
                let p = p + shift;

                map[p] += 1;
            }
        }

        maps.push(map);
    }

    let max_v = maps
        .iter()
        .flat_map(|m| m.iter())
        .max()
        .copied()
        .unwrap_or(0) as usize;

    // log2のテーブルを作成
    // p log p はp=1のとき0になるため、適当な値を入れてよい
    let log_table = (0..=sample_count)
        .map(|v| (1.0 / v.max(1) as f64).log2())
        .collect::<Vec<_>>();

    let mut counts = Map2d::new_with(vec![0; max_v + 1], input.map_size);

    for map in maps.iter() {
        for (&m, cnt) in map.iter().zip(counts.iter_mut()) {
            cnt[m as usize] += 1;
        }
    }

    let mut candidates = vec![];
    let mut best_entropy = f64::MAX;

    for row in 0..input.map_size {
        for col in 0..input.map_size {
            let c = Coord::new(row, col);

            if env.map[c].is_some() {
                continue;
            }

            // エントロピーを計算し、最小となる箇所（相互情報量が最大となる箇所）を選ぶ
            let mut entropy = 0.0;

            for &c in counts[c].iter() {
                // H(配置|観測値) = Σ_i p(観測値=i)H(配置|観測値=i)
                //               = Σ_i p(観測値=i) log(n(配置|観測値=i))
                let p = c as f64 / sample_count as f64;
                entropy -= p * log_table[c];
            }

            if best_entropy.change_min(entropy) {
                candidates.clear();
            }

            if best_entropy == entropy {
                candidates.push(c);
            }
        }
    }

    let c = candidates.choose(rng).copied().unwrap();
    let mutual_info = -log_table[sample_count] - best_entropy;

    judge.comment(&format!(
        "next: {} with mutual information {:.3}",
        c, mutual_info
    ));

    c
}

fn color_map(states: &[State], rng: &mut rand_pcg::Mcg128Xsl64, env: &Env<'_>, judge: &mut Judge) {
    let state = states.choose(rng).unwrap().clone();
    let mut colors = Map2d::new_with(0.0, env.input.map_size);

    for &c in state.to_answer(env.input).iter() {
        colors[c] = 1.0 / states.len() as f64;
    }

    judge.comment(&format!(
        "score: {}, remaining: {}",
        state.calc_score(),
        states.len()
    ));
    judge.comment_colors(&colors);
}

#[derive(Debug, Clone)]
struct Env<'a> {
    input: &'a Input,
    map: Map2d<Option<i32>>,
    shift_candidates: Vec<Vec<CoordDiff>>,
}

impl<'a> Env<'a> {
    fn new(input: &'a Input) -> Self {
        let map = Map2d::new_with(None, input.map_size);
        let shift_candidates = input
            .oils
            .iter()
            .map(|oil| {
                let mut cands = vec![];

                for row in 0..=input.map_size - oil.height {
                    for col in 0..=input.map_size - oil.width {
                        cands.push(CoordDiff::new(row as isize, col as isize));
                    }
                }

                cands
            })
            .collect_vec();

        Self {
            input,
            map,
            shift_candidates,
        }
    }

    fn add_observation(&mut self, pos: Coord, value: i32) {
        self.map[pos] = Some(value);

        for (oil, cand) in self.input.oils.iter().zip(self.shift_candidates.iter_mut()) {
            if value == 0 {
                cand.retain(|&shift| {
                    oil.pos.iter().all(|&p| {
                        let p = p + shift;
                        p != pos
                    })
                });
            } else if value == self.input.oil_count as i32 {
                cand.retain(|&shift| {
                    oil.pos.iter().any(|&p| {
                        let p = p + shift;
                        p == pos
                    })
                });
            }
        }
    }
}

#[derive(Debug, Clone)]
struct State {
    shift: Vec<CoordDiff>,
    violations: i32,
    map: Map2d<Option<i32>>,
    hash: u64,
}

static mut BEST_SHIFT_BUF: Vec<CoordDiff> = vec![];

impl State {
    fn new(shift: Vec<CoordDiff>, mut map: Map2d<Option<i32>>, env: &Env) -> Self {
        for (oil, &shift) in env.input.oils.iter().zip(shift.iter()) {
            for &p in oil.pos.iter() {
                let pos = p + shift;

                let Some(v) = &mut map[pos] else {
                    continue;
                };

                *v -= 1;
            }
        }

        let mut violations = 0;

        for &v in map.iter().flatten() {
            violations += v.abs();
        }

        let mut hash = 0;

        for (&shift, h) in shift.iter().zip(env.input.hashes.iter()) {
            hash ^= h[Coord::try_from(shift).unwrap()];
        }

        let mut state = Self {
            shift,
            violations,
            map,
            hash,
        };

        state.normalize(&env);
        state
    }

    fn add_observation(&mut self, env: &Env, pos: Coord, value: i32) {
        self.map[pos] = Some(value);

        for (oil, &shift) in env.input.oils.iter().zip(self.shift.iter()) {
            for &p in oil.pos.iter() {
                let p = p + shift;

                if p != pos {
                    continue;
                }

                let v = self.map[p].as_mut().unwrap();

                *v -= 1;
            }
        }

        self.violations += self.map[pos].unwrap().abs();
    }

    fn neigh(mut self, env: &Env, rng: &mut impl Rng, choose_cnt: usize) -> Self {
        let mut oil_indices = (0..env.input.oil_count).choose_multiple(rng, choose_cnt);
        oil_indices.shuffle(rng);

        for &oil_i in &oil_indices {
            self.remove_oil(env, oil_i);

            let shift = env.shift_candidates[oil_i].choose(rng).copied().unwrap();
            self.add_oil(env, oil_i, shift);
        }

        for &oil_i in &oil_indices {
            let mut best_violation = i32::MAX;
            let best_shifts = unsafe { &mut BEST_SHIFT_BUF };
            best_shifts.clear();
            self.remove_oil(env, oil_i);

            for &shift in env.shift_candidates[oil_i].iter() {
                let violation = self.add_oil_whatif(env, oil_i, shift);

                if best_violation.change_min(violation) {
                    best_shifts.clear();
                }

                if violation == best_violation {
                    best_shifts.push(shift);
                }
            }

            let best_shift = *best_shifts.choose(rng).unwrap();

            self.add_oil(env, oil_i, best_shift);
        }

        self.normalize(&env);

        self
    }

    fn add_oil(&mut self, env: &Env, oil: usize, shift: CoordDiff) {
        for &p in env.input.oils[oil].pos.iter() {
            let p = p + shift;

            let Some(v) = &mut self.map[p] else {
                continue;
            };

            if *v > 0 {
                self.violations -= 1;
            } else {
                self.violations += 1;
            }

            *v -= 1;
        }

        self.shift[oil] = shift;
        self.hash ^= env.input.hashes[oil][Coord::try_from(shift).unwrap()];
    }

    fn remove_oil(&mut self, env: &Env, oil: usize) {
        let shift = self.shift[oil];

        for &p in env.input.oils[oil].pos.iter() {
            let p = p + shift;

            let Some(v) = &mut self.map[p] else {
                continue;
            };

            if *v >= 0 {
                self.violations += 1;
            } else {
                self.violations -= 1;
            }

            *v += 1;
        }

        self.hash ^= env.input.hashes[oil][Coord::try_from(shift).unwrap()];
    }

    fn add_oil_whatif(&self, env: &Env, oil: usize, shift: CoordDiff) -> i32 {
        let mut violation = 0;

        for &p in env.input.oils[oil].pos.iter() {
            let p = p + shift;

            let Some(v) = self.map[p] else {
                continue;
            };

            if v > 0 {
                violation -= 1;
            } else {
                violation += 1;
            }
        }

        violation
    }

    fn normalize(&mut self, env: &Env) {
        let mut groups = FxHashMap::default();

        for (i, (oil, shift)) in env
            .input
            .oils
            .iter()
            .zip(self.shift.iter().copied())
            .enumerate()
        {
            let entry = groups.entry(&oil.pos).or_insert_with(|| vec![]);
            entry.push((i, shift));
        }

        let mut new_shift = vec![CoordDiff::new(0, 0); env.input.oil_count];

        for group in groups.values() {
            for &(i, shift) in group {
                self.hash ^= env.input.hashes[i][Coord::try_from(shift).unwrap()];
            }

            let mut shifts = group.iter().map(|&(_, shift)| shift).collect_vec();

            shifts.sort_unstable();

            for (i, &shift) in group.iter().map(|&(i, _)| i).zip(shifts.iter()) {
                self.hash ^= env.input.hashes[i][Coord::try_from(shift).unwrap()];
                new_shift[i] = shift;
            }
        }

        self.shift = new_shift;
    }

    fn calc_score(&self) -> i32 {
        self.violations
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

impl PartialEq for State {
    fn eq(&self, other: &Self) -> bool {
        self.hash == other.hash
    }
}

impl Eq for State {}

impl std::hash::Hash for State {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.hash.hash(state);
    }
}

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

fn generate_states(
    env: &Env,
    mut states: Vec<State>,
    duration: f64,
    rng: &mut impl Rng,
) -> Vec<State> {
    let mut hashes = FxHashSet::default();

    for state in states.iter() {
        hashes.insert(state.hash);
    }

    let mut prefix_prob = vec![OrderedFloat(prob(&states[0]))];

    for i in 1..states.len() {
        let p = OrderedFloat(prob(&states[i]) + prefix_prob[i - 1].0);
        prefix_prob.push(p);
    }

    let mut all_iter = 0;

    let duration_inv = 1.0 / duration;
    let since = std::time::Instant::now();

    let oil_count_dist = WeightedAliasIndex::new(vec![0, 10, 60, 20, 10]).unwrap();

    loop {
        all_iter += 1;
        if (all_iter & ((1 << 4) - 1)) == 0 {
            let time = env.input.duration_corrector.elapsed(since).as_secs_f64() * duration_inv;

            if time >= 1.0 {
                break;
            }
        }

        // 変形
        let x = rng.gen_range(0.0..prefix_prob.last().unwrap().0);
        let index = prefix_prob
            .binary_search(&OrderedFloat(x))
            .unwrap_or_else(|x| x);
        let state = states[index].clone();

        let oil_count = (oil_count_dist.sample(rng)).min(env.input.oil_count);
        let new_state = state.neigh(env, rng, oil_count);

        if hashes.insert(new_state.hash) {
            let p = prob(&new_state) + prefix_prob.last().unwrap().0;
            prefix_prob.push(OrderedFloat(p));
            states.push(new_state);
        }
    }

    eprintln!(
        "all_iter: {}, violation: {}",
        all_iter,
        states.iter().map(|s| s.violations).min().unwrap()
    );

    states
}

fn prob(state: &State) -> f64 {
    2.0f64.powi(-state.violations)
}
