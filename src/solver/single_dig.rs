use itertools::Itertools as _;
use rand::{seq::SliceRandom as _, Rng};
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

        let each_duration = 2.0 / (input.map_size * input.map_size) as f64;
        let mut next_pos = None;

        for turn in 0..input.map_size * input.map_size {
            let state = State::new(
                vec![CoordDiff::new(0, 0); input.oil_count],
                env.map.clone(),
                input,
            );

            let state = state.neigh(&env, &mut rng, input.oil_count);

            let solutions = climbing(&env, state, each_duration);
            let candidates_len = solutions.len();
            eprintln!("turn: {}, solutions: {}", turn, candidates_len);

            let solutions = solutions.into_iter().collect_vec();
            color_map(
                &solutions,
                &mut rng,
                &env,
                input,
                candidates_len,
                &mut self.judge,
            );

            if candidates_len == 1 {
                let solution = solutions.iter().next().unwrap();
                let solution = State::new(solution.clone(), env.map.clone(), input);

                if solution.calc_score() == 0 {
                    let answer = solution.to_answer(input);

                    if self.judge.answer(&answer).is_ok() {
                        return;
                    }
                }
            } else if solutions.len() >= 2 {
                next_pos = choose_next_pos(solutions, &mut rng, input, &env, &mut self.judge);
            }

            let coord = next_pos.take().unwrap_or_else(|| loop {
                let row = rng.gen_range(0..input.map_size);
                let col = rng.gen_range(0..input.map_size);
                let c = Coord::new(row, col);

                if env.map[c].is_none() {
                    break c;
                }
            });

            let observation = self.judge.query_single(coord);
            env.add_observation(coord, observation);
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
    solutions: Vec<Vec<CoordDiff>>,
    rng: &mut rand_pcg::Mcg128Xsl64,
    input: &Input,
    env: &Env<'_>,
    judge: &mut Judge,
) -> Option<Coord> {
    let mut maps = vec![];
    let sample_count = 200.min(solutions.len());
    let solutions = solutions.choose_multiple(rng, sample_count);

    for shifts in solutions {
        let mut map = Map2d::new_with(0, input.map_size);

        for (&shift, oil) in shifts.iter().zip(input.oils.iter()) {
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

    let log_table = (0..=sample_count)
        .map(|v| (v.max(1) as f64 / sample_count as f64).log2())
        .collect::<Vec<_>>();

    let mut counts = Map2d::new_with(vec![0; max_v + 1], input.map_size);

    for map in maps.iter() {
        for (&m, cnt) in map.iter().zip(counts.iter_mut()) {
            cnt[m as usize] += 1;
        }
    }

    let mut candidates = vec![];
    let mut best_entropy = f64::MIN;

    for row in 0..input.map_size {
        for col in 0..input.map_size {
            let c = Coord::new(row, col);

            if env.map[c].is_some() {
                continue;
            }

            // エントロピーを計算し、最大となる箇所を選ぶ
            let mut entropy = 0.0;

            for &c in counts[c].iter() {
                let p = c as f64 / sample_count as f64;
                entropy -= p * log_table[c];
            }

            if best_entropy.change_max(entropy) {
                candidates.clear();
            }

            if best_entropy == entropy {
                candidates.push(c);
            }
        }
    }

    let c = candidates.choose(rng).copied();

    if let Some(c) = c {
        judge.comment(&format!("next: {} with entropy {:.3}", c, best_entropy));
    }

    c
}

fn color_map(
    solutions: &Vec<Vec<CoordDiff>>,
    rng: &mut rand_pcg::Mcg128Xsl64,
    env: &Env<'_>,
    input: &Input,
    candidates_len: usize,
    judge: &mut Judge,
) {
    let solution = solutions.choose(rng).unwrap().clone();
    let solution = State::new(solution.clone(), env.map.clone(), input);
    let mut colors = Map2d::new_with(0.0, input.map_size);

    for &c in solution.to_answer(input).iter() {
        colors[c] = 1.0 / candidates_len as f64;
    }

    judge.comment(&format!(
        "score: {}, remaining: {}",
        solution.calc_score(),
        candidates_len
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

impl State {
    fn new(shift: Vec<CoordDiff>, mut map: Map2d<Option<i32>>, input: &Input) -> Self {
        for (oil, &shift) in input.oils.iter().zip(shift.iter()) {
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

        for (&shift, h) in shift.iter().zip(input.hashes.iter()) {
            hash ^= h[Coord::try_from(shift).unwrap()];
        }

        let mut state = Self {
            shift,
            violations,
            map,
            hash,
        };

        state.normalize(input);
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

                let v = &mut self.map[p].unwrap();

                *v -= 1;
            }
        }

        self.violations += self.map[pos].unwrap().abs();
    }

    fn neigh(mut self, env: &Env, rng: &mut impl Rng, choose_cnt: usize) -> Self {
        let mut oil_indices = vec![];

        for _ in 0..choose_cnt {
            let oil = loop {
                let oil = rng.gen_range(0..env.input.oil_count);

                if !oil_indices.contains(&oil) {
                    break oil;
                }
            };

            oil_indices.push(oil);
        }

        for &oil_i in &oil_indices {
            self.remove_oil(env, oil_i);
        }

        for &oil_i in &oil_indices {
            let mut best_violation = i32::MAX;
            let mut best_shifts = vec![];

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

        self.normalize(&env.input);

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

    fn normalize(&mut self, input: &Input) {
        let mut groups = FxHashMap::default();

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

fn climbing(env: &Env, initial_solution: State, duration: f64) -> FxHashSet<Vec<CoordDiff>> {
    let mut solution = initial_solution;
    let mut best_solutions = FxHashSet::default();
    let mut current_score = solution.calc_score();
    let mut best_score = current_score;

    let mut all_iter = 0;
    let mut rng = rand_pcg::Pcg64Mcg::new(42);

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
        let oil_count = (oil_count_dist.sample(&mut rng)).min(env.input.oil_count);
        let new_solution = solution.clone().neigh(env, &mut rng, oil_count);

        // スコア計算
        let new_score = new_solution.calc_score();
        let score_diff = new_score - current_score;

        if score_diff <= 0 {
            // 解の更新
            current_score = new_score;
            solution = new_solution;

            if best_score.change_min(current_score) {
                best_solutions.clear();
            }

            best_solutions.insert(solution.shift.clone());
        }
    }

    best_solutions
}
