use nalgebra::{DMatrix, DVector};
use rand::{seq::IteratorRandom, Rng};
use rand_core::SeedableRng;
use rand_pcg::Pcg64Mcg;

use crate::{
    grid::{Coord, CoordDiff, Map2d},
    problem::{Input, Judge},
};

use super::Solver;

pub struct BayesianSolver {
    judge: Judge,
}

#[allow(dead_code)]
impl BayesianSolver {
    pub fn new(judge: Judge) -> Self {
        Self { judge }
    }
}

impl Solver for BayesianSolver {
    fn solve(&mut self, input: &crate::problem::Input) {
        let (mut mean, mut covariance) = sample_prior_dist(input);
        let vec_len = mean.len();
        let mut rng = Pcg64Mcg::from_entropy();

        for _ in 0..input.map_size * input.map_size * 2 - 1 {
            let mut map = Map2d::new_with(0.0, input.map_size);

            for i in 0..vec_len {
                let row = i / input.map_size;
                let col = i % input.map_size;
                let c = Coord::new(row, col);
                map[c] = mean[i] / 2.0;
            }

            self.judge.comment_colors(&map);

            let sample_amount = vec_len * 1 / 2;
            let candidates = (0..mean.len()).choose_multiple(&mut rng, sample_amount);
            let mut sample_vec = DVector::zeros(vec_len);
            let mut sample_targets = vec![];

            for &i in candidates.iter() {
                sample_vec[i] = 1.0;
                sample_targets.push(Coord::new(i / input.map_size, i % input.map_size));
            }

            let result = self.judge.query_multiple(&sample_targets) as f64;
            let variance = input.eps * (1.0 - input.eps) * sample_targets.len() as f64;
            let result_mat = DMatrix::from_vec(1, 1, vec![result]);
            let variance_mat = DMatrix::from_vec(1, 1, vec![variance]);

            // AHC003の2.926T解法+経緯
            // https://qiita.com/contramundum/items/b945400b81536df42d1a
            let syx = sample_vec.transpose() * &covariance;
            let sxy = syx.transpose();
            let syy = sample_vec.transpose() * &covariance * &sample_vec + variance_mat;
            let syy_inv = syy.try_inverse().unwrap();

            mean = &mean + &sxy * &syy_inv * (result_mat - sample_vec.transpose() * &mean);
            covariance = covariance - &sxy * &syy_inv * &syx;
        }

        let mut answer = vec![];

        for i in 0..vec_len {
            let row = i / input.map_size;
            let col = i % input.map_size;
            let c = Coord::new(row, col);

            if mean[i] >= 0.5 {
                answer.push(c);
            }
        }

        _ = self.judge.answer(&answer);
    }
}

fn sample_prior_dist(input: &Input) -> (DVector<f64>, DMatrix<f64>) {
    const SAMPLE_COUNT: usize = 10000;
    let mut samples: Map2d<Vec<u32>> =
        Map2d::new_with(Vec::with_capacity(SAMPLE_COUNT), input.map_size);
    let mut rng = Pcg64Mcg::from_entropy();
    let mut map = Map2d::new_with(0, input.map_size);

    for _ in 0..SAMPLE_COUNT {
        for v in map.iter_mut() {
            *v = 0;
        }

        for oil in input.oils.iter() {
            let dr = rng.gen_range(0..=input.map_size - oil.height);
            let dc = rng.gen_range(0..=input.map_size - oil.width);
            let shift = CoordDiff::new(dr as isize, dc as isize);

            for &p in oil.pos.iter() {
                map[p + shift] += 1;
            }
        }

        for row in 0..input.map_size {
            for col in 0..input.map_size {
                let c = Coord::new(row, col);
                samples[c].push(map[c]);
            }
        }
    }

    let len = input.map_size * input.map_size;
    let mut samples_vec = Map2d::new_with(DVector::zeros(0), input.map_size);

    for row in 0..input.map_size {
        for col in 0..input.map_size {
            let c = Coord::new(row, col);
            samples_vec[c] = DVector::from_vec(samples[c].clone());
        }
    }

    let mut mean = DVector::zeros(len);

    for i in 0..len {
        let row = i / input.map_size;
        let col = i % input.map_size;
        let c = Coord::new(row, col);
        let sum = samples_vec[c].sum();
        mean[i] = sum as f64 / SAMPLE_COUNT as f64;
    }

    let mut covariance = DMatrix::zeros(len, len);

    for i in 0..len {
        let c0 = Coord::new(i / input.map_size, i % input.map_size);

        for j in i..len {
            let c1 = Coord::new(j / input.map_size, j % input.map_size);

            // Cov(X, Y) = E[XY] - mx * my
            let dot = samples_vec[c0].dot(&samples_vec[c1]);
            let expected = dot as f64 / SAMPLE_COUNT as f64;
            let cov = expected - mean[i] * mean[j];
            covariance[(i, j)] = cov;
            covariance[(j, i)] = cov;
        }
    }

    (mean, covariance)
}
