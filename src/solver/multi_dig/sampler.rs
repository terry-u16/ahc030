use std::{cmp::Reverse, iter::Map};

use itertools::Itertools;
use ordered_float::OrderedFloat;
use rand::{seq::SliceRandom as _, Rng};

use crate::{
    common::ChangeMinMax as _,
    distributions::GaussianDistribution,
    grid::{Coord, Map2d},
    problem::Input,
};

use super::mcmc::{self};

pub(super) fn select_sample_points(
    input: &Input,
    prob_table: &mut ProbTable,
    mcmc_states: Vec<mcmc::State>,
    max_sample_count: usize,
    duration: f64,
    rng: &mut impl Rng,
) -> Vec<Coord> {
    assert!(max_sample_count > 0);
    assert!(input.map_size * input.map_size >= max_sample_count);

    let all_coords = (0..input.map_size)
        .flat_map(|row| (0..input.map_size).map(move |col| Coord::new(row, col)))
        .collect::<Vec<_>>();
    let sample_count = max_sample_count / 2;
    let sampled_coords = all_coords
        .choose_multiple(rng, sample_count)
        .copied()
        .collect_vec();

    let mut map = Map2d::new_with(false, input.map_size);

    for (&shift, oil) in mcmc_states[0].shift.iter().zip(input.oils.iter()) {
        for &p in oil.pos.iter() {
            map[p + shift] = true;
        }
    }

    let Ok(env) = Env::new(input, mcmc_states, max_sample_count) else {
        // エントロピーを計算できないのでランダムなものを返す
        return sampled_coords;
    };

    let count = map.iter().filter(|&&b| !b).count();
    let flip = count >= max_sample_count;

    let mut sampled_coords = vec![];

    for row in 0..input.map_size {
        for col in 0..input.map_size {
            let c = Coord::new(row, col);

            if !map[c] ^ flip {
                sampled_coords.push(c);
            }
        }
    }

    let state = State::new(&env, &sampled_coords);
    let state = annealing(&env, state, prob_table, duration, rng);

    let mut result = vec![];

    for row in 0..input.map_size {
        for col in 0..input.map_size {
            let c = Coord::new(row, col);

            if state.map[c] {
                result.push(c);
            }
        }
    }

    result
}

pub(super) struct ProbTable {
    prob: Vec<Vec<Vec<f64>>>,
    prob_log: Vec<Vec<Vec<f64>>>,
    targets: Vec<Vec<Vec<(f64, f64, usize)>>>,
}

impl ProbTable {
    const MIN_PROB: f64 = f64::MIN_POSITIVE;
    const MIN_PROB_LN: f64 = f64::MIN;

    pub(super) fn new(input: &Input) -> Self {
        let sq_cnt = input.map_size * input.map_size;
        let oil_cnt = input.oils.iter().map(|o| o.len).sum::<usize>();

        let prob = vec![vec![vec![]; oil_cnt + 1]; sq_cnt + 1];
        let prob_log = vec![vec![vec![]; oil_cnt + 1]; sq_cnt + 1];
        let targets = vec![vec![vec![]; oil_cnt + 1]; sq_cnt + 1];

        Self {
            prob,
            prob_log,
            targets,
        }
    }

    fn prob(&mut self, input: &Input, sq_cnt: usize, true_cnt: usize, observe_cnt: usize) -> f64 {
        assert!(sq_cnt > 0);

        if self.prob[sq_cnt][true_cnt].is_empty() {
            self.lazy_init(input, sq_cnt, true_cnt);
        }

        self.prob[sq_cnt][true_cnt]
            .get(observe_cnt)
            .copied()
            .unwrap_or(Self::MIN_PROB)
    }

    fn prob_log(
        &mut self,
        input: &Input,
        sq_cnt: usize,
        true_cnt: usize,
        observe_cnt: usize,
    ) -> f64 {
        assert!(sq_cnt > 0);

        if self.prob_log[sq_cnt][true_cnt].is_empty() {
            self.lazy_init(input, sq_cnt, true_cnt);
        }

        self.prob_log[sq_cnt][true_cnt]
            .get(observe_cnt)
            .copied()
            .unwrap_or(Self::MIN_PROB_LN)
    }

    fn targets(&mut self, input: &Input, sq_cnt: usize, true_cnt: usize) -> &[(f64, f64, usize)] {
        assert!(sq_cnt > 0);

        if self.targets[sq_cnt][true_cnt].is_empty() {
            self.lazy_init(input, sq_cnt, true_cnt);
        }

        &self.targets[sq_cnt][true_cnt]
    }

    fn lazy_init(&mut self, input: &Input, sq_cnt: usize, true_cnt: usize) {
        assert!(sq_cnt > 0);

        let mut prob = vec![Self::MIN_PROB; input.map_size * input.map_size];
        let mut prob_log = vec![Self::MIN_PROB_LN; input.map_size * input.map_size];
        let mut targets = vec![];

        if sq_cnt == 1 {
            prob[true_cnt] = 1.0;
            prob_log[true_cnt] = 0.0;
            targets.push((1.0, 0.0, true_cnt));

            self.prob[sq_cnt][true_cnt] = prob;
            self.prob_log[sq_cnt][true_cnt] = prob_log;
            self.targets[sq_cnt][true_cnt] = targets;
            return;
        }

        let mean =
            (sq_cnt as f64 - true_cnt as f64) * input.eps + true_cnt as f64 * (1.0 - input.eps);
        let varinace = sq_cnt as f64 * input.eps * (1.0 - input.eps);
        let std_dev = varinace.sqrt();
        let gauss = GaussianDistribution::new(mean, std_dev);

        let f = |v: usize| -> f64 {
            if v == 0 {
                gauss.calc_cumulative_dist(0.5)
            } else {
                gauss.calc_cumulative_dist(v as f64 + 0.5)
                    - gauss.calc_cumulative_dist(v as f64 - 0.5)
            }
            .max(Self::MIN_PROB)
        };

        let p = f(true_cnt);
        let p_log2 = p.log2();
        prob[true_cnt] = p;
        prob_log[true_cnt] = p_log2;
        targets.push((p, p_log2, true_cnt));

        let mut sum_p = p;

        let (mut left, mut left_p) = if true_cnt == 0 {
            (None, 0.0)
        } else {
            (Some(true_cnt - 1), f(true_cnt - 1))
        };
        let mut right = true_cnt + 1;
        let mut right_p = f(true_cnt + 1);

        // ±4σ分をカバーできる範囲を計算
        while sum_p < 0.99994 {
            if left_p >= right_p {
                let l = left.unwrap();
                let p_log2 = left_p.log2();
                prob[l] = left_p;
                prob_log[l] = p_log2;
                targets.push((left_p, p_log2, l));
                sum_p += left_p;

                if l == 0 {
                    left = None;
                    left_p = 0.0;
                } else {
                    left = Some(l - 1);
                    left_p = f(l - 1);
                }
            } else {
                let p_log2 = right_p.log2();
                prob[right] = right_p;
                prob_log[right] = p_log2;
                targets.push((right_p, p_log2, right));
                sum_p += right_p;

                right += 1;
                right_p = f(right);
            }
        }

        targets.sort_unstable_by_key(|&(_, _, i)| i);

        self.prob[sq_cnt][true_cnt] = prob;
        self.prob_log[sq_cnt][true_cnt] = prob_log;
        self.targets[sq_cnt][true_cnt] = targets;
    }
}

struct Env<'a> {
    input: &'a Input,
    states: Vec<mcmc::State>,
    state_maps: Vec<Map2d<usize>>,
    probs: Vec<f64>,
    probs_log2: Vec<f64>,
    base_entropy: f64,
    max_sample_count: usize,
}

impl<'a> Env<'a> {
    const MAX_STATE_COUNT: usize = 20;

    fn new(
        input: &'a Input,
        mut states: Vec<mcmc::State>,
        max_sample_count: usize,
    ) -> Result<Self, ()> {
        if states.len() <= 1 {
            return Err(());
        }

        states.sort_unstable_by_key(|s| Reverse(OrderedFloat(s.log_likelihood)));

        // 適当に尤度の大きい上位MAX_STATE_COUNT個を選ぶ
        if states.len() > Self::MAX_STATE_COUNT {
            states.truncate(Self::MAX_STATE_COUNT);
        }

        let mut state_maps = vec![];

        for state in states.iter() {
            let mut map = Map2d::new_with(0, input.map_size);

            for (oil, &shift) in input.oils.iter().zip(state.shift.iter()) {
                for &p in oil.pos.iter() {
                    map[p + shift] += 1;
                }
            }

            state_maps.push(map);
        }

        let max_log_likelihood = states[0].log_likelihood;
        let mut probs = states
            .iter()
            .map(|s| (s.log_likelihood - max_log_likelihood).exp())
            .collect_vec();
        let sum_probs = probs.iter().sum::<f64>();

        for p in probs.iter_mut() {
            *p /= sum_probs;
        }

        let probs_log2 = probs.iter().map(|&p| p.log2()).collect_vec();

        let mut base_entropy = 0.0;

        for &prob in probs.iter() {
            base_entropy -= prob * prob.log2();
        }

        Ok(Self {
            input,
            states,
            state_maps,
            probs,
            probs_log2,
            base_entropy,
            max_sample_count,
        })
    }
}

#[derive(Debug, Clone)]
struct State {
    map: Map2d<bool>,
    selected_count: usize,
    conditional_entropy: Option<f64>,
}

impl State {
    fn new(env: &Env, selected_coords: &[Coord]) -> Self {
        let mut map = Map2d::new_with(false, env.input.map_size);

        for &c in selected_coords.iter() {
            map[c] = true;
        }

        let selected_count = selected_coords.len();

        Self {
            map,
            selected_count,
            conditional_entropy: None,
        }
    }

    fn neigh_flip_single(mut self, env: &Env, rng: &mut impl Rng) -> Self {
        let coord = loop {
            let row = rng.gen_range(0..env.input.map_size);
            let col = rng.gen_range(0..env.input.map_size);
            let c = Coord::new(row, col);

            let n2 = env.input.map_size * env.input.map_size;

            if (self.selected_count <= 1 && self.map[c])
                || (self.selected_count >= env.max_sample_count && !self.map[c])
            {
                continue;
            }

            break c;
        };

        if self.map[coord] {
            self.selected_count -= 1;
        } else {
            self.selected_count += 1;
        }

        self.map[coord] ^= true;
        self.conditional_entropy = None;

        self
    }

    fn neigh_flip_double(mut self, env: &Env, rng: &mut impl Rng) -> Self {
        let c0 = loop {
            let row = rng.gen_range(0..env.input.map_size);
            let col = rng.gen_range(0..env.input.map_size);
            let c = Coord::new(row, col);
            if self.map[c] {
                break c;
            }
        };

        let c1 = loop {
            let row = rng.gen_range(0..env.input.map_size);
            let col = rng.gen_range(0..env.input.map_size);
            let c = Coord::new(row, col);
            if !self.map[c] {
                break c;
            }
        };

        self.map[c0] ^= true;
        self.map[c1] ^= true;
        self.conditional_entropy = None;

        self
    }

    /// 相互情報量を計算する
    fn calc_conditional_entropy(&mut self, env: &Env, prob_table: &mut ProbTable) -> f64 {
        if let Some(entropy) = self.conditional_entropy {
            return entropy;
        };

        let mut p_obs = vec![];
        let mut p_oil_obs = vec![];

        // 確率の表を作成する
        for (map, (&state_prob, state_prob_log2)) in env
            .state_maps
            .iter()
            .zip(env.probs.iter().zip(env.probs_log2.iter()))
        {
            let mut counts = 0;

            for (&v, &b) in map.iter().zip(self.map.iter()) {
                counts += v * b as usize;
            }

            let mut current_p_oil_obs = vec![];
            for &(prob, prob_log2, target_v) in
                prob_table.targets(&env.input, self.selected_count, counts)
            {
                if p_obs.len() <= target_v {
                    p_obs.resize(target_v + 1, f64::MIN_POSITIVE);
                }

                // probは配置が確定したときに v_obs が観測される確率 p(v_obs | 配置)
                // p(v_obs, 配置) = p(v_obs | 配置) * p(配置)
                let p = prob * state_prob;
                let p_log2 = prob_log2 + state_prob_log2;
                p_obs[target_v] += p;
                current_p_oil_obs.push((target_v, p, p_log2));
            }

            p_oil_obs.push(current_p_oil_obs);
        }

        let mut entropy = 0.0;

        for p in p_obs.iter_mut() {
            if *p == f64::MIN_POSITIVE {
                *p = f64::MIN;
            } else {
                *p = p.log2();
            }
        }

        let p_obs_log2 = p_obs;

        for p_oil_obs in p_oil_obs.iter() {
            for &(v, pp, p_log2) in p_oil_obs.iter() {
                // dH = -p(v_obs) * p(配置 | v_obs) * log(p(配置 | v_obs))
                //    = -p(配置, v_obs) * log(p(配置, v_obs) / p(v_obs))
                //    = -p(配置, v_obs) * (log(p(配置, v_obs)) - log(p(v_obs)))
                entropy -= pp * (p_log2 - p_obs_log2[v]);
            }
        }

        self.conditional_entropy = Some(entropy);
        entropy
    }

    /// スコア（= 相互情報量 / 調査コスト）を計算する
    ///
    /// スコアは大きいほどよい
    fn calc_score(&mut self, env: &Env, prob_table: &mut ProbTable) -> f64 {
        let mutual_information = env.base_entropy - self.calc_conditional_entropy(env, prob_table);
        let score = mutual_information * (self.selected_count as f64).sqrt();
        score
    }
}

fn annealing(
    env: &Env,
    initial_solution: State,
    prob_table: &mut ProbTable,
    duration: f64,
    rng: &mut impl Rng,
) -> State {
    let mut solution = initial_solution;
    let mut best_solution = solution.clone();
    let mut current_score = solution.calc_score(env, prob_table);
    let mut best_score = current_score;
    let init_score = current_score;

    let mut all_iter = 0;
    let mut valid_iter = 0;
    let mut accepted_count = 0;
    let mut update_count = 0;

    let duration_inv = 1.0 / duration;
    let since = std::time::Instant::now();

    let temp0 = 1e-2;
    let temp1 = 1e-4;
    let mut inv_temp = 1.0 / temp0;

    loop {
        all_iter += 1;
        if (all_iter & ((1 << 4) - 1)) == 0 {
            let time = (std::time::Instant::now() - since).as_secs_f64() * duration_inv;
            let temp = f64::powf(temp0, 1.0 - time) * f64::powf(temp1, time);
            inv_temp = 1.0 / temp;

            if time >= 1.0 {
                break;
            }
        }

        // 変形
        let mut new_state = if rng.gen_bool(0.5) {
            solution.clone().neigh_flip_single(env, rng)
        } else {
            solution.clone().neigh_flip_double(env, rng)
        };

        // スコア計算
        let new_score = new_state.calc_score(env, prob_table);
        let score_diff = new_score - current_score;

        // 速度が稼げていないので山登りにしている
        if score_diff >= 0.0 {
            // || rng.gen_bool(f64::exp(score_diff as f64 * inv_temp)) {
            // 解の更新
            current_score = new_score;
            accepted_count += 1;
            solution = new_state;

            if best_score.change_max(current_score) {
                best_solution = solution.clone();
                update_count += 1;
            }
        }

        valid_iter += 1;
    }

    eprintln!("===== annealing =====");
    eprintln!("init score : {}", init_score);
    eprintln!("score      : {}", best_score);
    eprintln!("all iter   : {}", all_iter);
    eprintln!("valid iter : {}", valid_iter);
    eprintln!("accepted   : {}", accepted_count);
    eprintln!("updated    : {}", update_count);
    eprintln!("");

    best_solution
}
