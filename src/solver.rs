use itertools::Itertools;
use rand::{seq::SliceRandom as _, Rng};
use rand_core::SeedableRng;
use rand_distr::{Distribution, WeightedAliasIndex};
use rand_pcg::Pcg64Mcg;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::{
    common::ChangeMinMax,
    grid::{Coord, CoordDiff, Map2d},
    problem::{Input, Judge},
};

pub fn solve(mut judge: Judge, input: &Input) {
    let mut env = Env::new(input);
    let mut rng = Pcg64Mcg::from_entropy();

    let each_duration = 2.0 / (input.map_size * input.map_size) as f64;
    let mut next_pos = None;

    for turn in 0..input.map_size * input.map_size {
        let coord = next_pos.take().unwrap_or_else(|| loop {
            let row = rng.gen_range(0..input.map_size);
            let col = rng.gen_range(0..input.map_size);
            let c = Coord::new(row, col);

            if env.map[c].is_none() {
                break c;
            }
        });

        env.map[coord] = Some(judge.query_single(coord));

        let state = State::new(
            vec![CoordDiff::new(0, 0); input.oil_count],
            env.map.clone(),
            input,
        );
        let state = state.neigh(input, &mut rng, input.oil_count);

        let solutions = climbing(&env, state, each_duration);
        let candidates_len = solutions.len();
        eprintln!("turn: {}, solutions: {}", turn, candidates_len);

        let solutions = solutions.into_iter().collect_vec();
        color_map(&solutions, &mut rng, &env, input, candidates_len, &judge);

        if candidates_len == 1 {
            let solution = solutions.iter().next().unwrap();
            let solution = State::new(solution.clone(), env.map.clone(), input);

            if solution.calc_score() != 0 {
                continue;
            }

            let answer = solution.to_answer(input);

            if judge.answer(&answer).is_ok() {
                return;
            }
        } else if solutions.len() >= 2 {
            next_pos = choose_next_pos(solutions, &mut rng, input, &env);
        }
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

    judge.answer(&answer).unwrap();
}

fn choose_next_pos(
    solutions: Vec<Vec<CoordDiff>>,
    rng: &mut rand_pcg::Mcg128Xsl64,
    input: &Input,
    env: &Env<'_>,
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
        .map(|v| (v.max(1) as f64 / sample_count as f64).ln())
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
                if c == 0 {
                    continue;
                }

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

    candidates.choose(rng).copied()
}

fn color_map(
    solutions: &Vec<Vec<CoordDiff>>,
    rng: &mut rand_pcg::Mcg128Xsl64,
    env: &Env<'_>,
    input: &Input,
    candidates_len: usize,
    judge: &Judge,
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
}

impl<'a> Env<'a> {
    fn new(input: &'a Input) -> Self {
        let map = Map2d::new_with(None, input.map_size);

        Self { input, map }
    }
}

#[derive(Debug, Clone)]
struct State {
    shift: Vec<CoordDiff>,
    violations: i32,
    map: Map2d<Option<i32>>,
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

        let mut state = Self {
            shift,
            violations,
            map,
        };

        state.normalize(input);
        state
    }

    fn neigh(&self, input: &Input, rng: &mut impl Rng, choose_cnt: usize) -> Self {
        let mut oil_indices = vec![];

        for _ in 0..choose_cnt {
            let oil = loop {
                let oil = rng.gen_range(0..input.oil_count);

                if !oil_indices.contains(&oil) {
                    break oil;
                }
            };

            oil_indices.push(oil);
        }

        let mut state = self.clone();

        for &oil in &oil_indices {
            let shift = state.shift[oil];

            for &p in input.oils[oil].pos.iter() {
                let p = p + shift;

                let Some(v) = &mut state.map[p] else {
                    continue;
                };

                if *v >= 0 {
                    state.violations += 1;
                } else {
                    state.violations -= 1;
                }

                *v += 1;
            }
        }

        for &oil_i in &oil_indices {
            let oil = &input.oils[oil_i];
            let mut best_violation = i32::MAX;
            let mut best_shifts = vec![];

            for row in 0..=input.map_size - oil.height {
                for col in 0..=input.map_size - oil.width {
                    let shift = CoordDiff::new(row as isize, col as isize);

                    let mut violation = 0;

                    for &p in oil.pos.iter() {
                        let p = p + shift;

                        let Some(v) = &mut state.map[p] else {
                            continue;
                        };

                        if *v > 0 {
                            violation -= 1;
                        } else {
                            violation += 1;
                        }
                    }

                    if best_violation.change_min(violation) {
                        best_shifts.clear();
                    }

                    if violation == best_violation {
                        best_shifts.push(shift);
                    }
                }
            }

            let best_shift = *best_shifts.choose(rng).unwrap();

            for &p in oil.pos.iter() {
                let p = p + best_shift;

                let Some(v) = &mut state.map[p] else {
                    continue;
                };

                *v -= 1;
            }

            state.shift[oil_i] = best_shift;
            state.violations += best_violation;
        }

        state.normalize(input);

        state
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

    let oil_count_dist = WeightedAliasIndex::new(vec![
        10, 60, 20, 10, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    ])
    .unwrap();

    loop {
        all_iter += 1;
        if (all_iter & ((1 << 4) - 1)) == 0 {
            let time = (std::time::Instant::now() - since).as_secs_f64() * duration_inv;

            if time >= 1.0 {
                break;
            }
        }

        // 変形
        let oil_count = (oil_count_dist.sample(&mut rng) + 1).min(env.input.oil_count);
        let new_solution = solution.neigh(&env.input, &mut rng, oil_count);

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
