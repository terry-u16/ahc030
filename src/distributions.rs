use std::time::Instant;

use nalgebra::{DMatrix, DVector};
use rand::Rng as _;
use rand_core::SeedableRng as _;
use rand_pcg::Pcg64Mcg;

use crate::{
    grid::{Coord, CoordDiff, Map2d},
    problem::Input,
};

#[derive(Debug, Clone, Copy)]
pub struct GaussianDistribution {
    mean: f64,
    std_dev: f64,
}

impl GaussianDistribution {
    pub fn new(mean: f64, std_dev: f64) -> Self {
        Self { mean, std_dev }
    }

    /// 累積分布関数F(x)を計算する
    pub fn calc_cumulative_dist(&self, x: f64) -> f64 {
        // ロジスティック近似する
        let x = (x - self.mean) / self.std_dev;
        1.0 / (1.0 + (-0.07056 * x * x * x - 1.5976 * x).exp())
    }
}

/// 石油埋蔵量が多変数正規分布に従うと大胆仮定して、ベイズ更新により事後分布を求めるやつ
#[derive(Debug, Clone)]
pub struct GaussianBayesianEstimator {
    mean: DVector<f64>,
    std_dev: DVector<f64>,
    covariance: DMatrix<f64>,
    vec_len: usize,
}

impl GaussianBayesianEstimator {
    pub fn new(input: &Input) -> Self {
        // 油田の位置を多数サンプリングして用いて事前分布を求める
        const SAMPLE_COUNT: usize = 10000;
        const TIME_LIMIT: f64 = 0.1;
        let vec_len = input.map_size * input.map_size;
        let mut samples: Map2d<Vec<u32>> =
            Map2d::new_with(Vec::with_capacity(SAMPLE_COUNT), input.map_size);
        let mut rng = Pcg64Mcg::from_entropy();
        let mut map = Map2d::new_with(0, input.map_size);
        let since = Instant::now();

        for _ in 0..SAMPLE_COUNT {
            if since.elapsed().as_secs_f64() > TIME_LIMIT {
                break;
            }

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

        let mut samples_vec = Map2d::new_with(DVector::zeros(0), input.map_size);

        for row in 0..input.map_size {
            for col in 0..input.map_size {
                let c = Coord::new(row, col);
                samples_vec[c] = DVector::from_vec(samples[c].clone());
            }
        }

        let mut mean = DVector::zeros(vec_len);

        for i in 0..vec_len {
            let row = i / input.map_size;
            let col = i % input.map_size;
            let c = Coord::new(row, col);
            let sum = samples_vec[c].sum();
            mean[i] = sum as f64 / SAMPLE_COUNT as f64;
        }

        let mut covariance = DMatrix::zeros(vec_len, vec_len);

        for i in 0..vec_len {
            let c0 = Coord::new(i / input.map_size, i % input.map_size);

            for j in i..vec_len {
                let c1 = Coord::new(j / input.map_size, j % input.map_size);

                // Cov(X, Y) = E[XY] - mx * my
                let dot = samples_vec[c0].dot(&samples_vec[c1]);
                let expected = dot as f64 / SAMPLE_COUNT as f64;
                let cov = expected - mean[i] * mean[j];
                covariance[(i, j)] = cov;
                covariance[(j, i)] = cov;
            }
        }

        let std_dev = covariance.diagonal().map(|v| v.sqrt()).into_owned();
        eprintln!("{:?}", since.elapsed());

        Self {
            mean,
            std_dev,
            covariance,
            vec_len,
        }
    }

    /// サンプリング結果を用いてベイズ更新を行う
    pub fn update(&mut self, input: &Input, sample_targets: &[Coord], sampled_value: i32) {
        let result = sampled_value as f64;
        let variance = input.eps * (1.0 - input.eps) * sample_targets.len() as f64;
        let result_mat = DMatrix::from_vec(1, 1, vec![result]);
        let variance_mat = DMatrix::from_vec(1, 1, vec![variance]);

        let mut sample_vec = DVector::zeros(self.vec_len);

        for &c in sample_targets {
            let i = c.row * input.map_size + c.col;
            sample_vec[i] = 1.0;
        }

        // AHC003の2.926T解法+経緯
        // https://qiita.com/contramundum/items/b945400b81536df42d1a
        let syx = sample_vec.transpose() * &self.covariance;
        let sxy = syx.transpose();
        let syy = sample_vec.transpose() * &self.covariance * &sample_vec + variance_mat;
        let syy_inv = syy.try_inverse().unwrap();

        self.mean += &sxy * &syy_inv * (result_mat - sample_vec.transpose() * &self.mean);
        self.covariance -= &sxy * &syy_inv * &syx;
        self.std_dev = self.covariance.diagonal().map(|v| v.sqrt()).into_owned();
    }

    /// 与えられた座標の埋蔵量の周辺確率分布を取得する
    pub fn get_marginal_distribution(&self, input: &Input, coord: Coord) -> GaussianDistribution {
        let i = coord.row * input.map_size + coord.col;
        let mean = self.mean[i].max(0.0);
        let std_dev = self.std_dev[i];

        GaussianDistribution::new(mean, std_dev)
    }
}

#[cfg(test)]
mod test {
    use super::GaussianDistribution;

    #[test]
    fn calc_cumulative_dist() {
        const TOLERANCE: f64 = 1e-3;
        let cases = [(0.0, 1.0, 0.0, 0.5), (5.0, 10.0, 0.11, 0.312420848)];

        for &(average, std_dev, x, expected) in cases.iter() {
            let gauss = GaussianDistribution::new(average, std_dev);
            let actual = gauss.calc_cumulative_dist(x);
            assert!((actual - expected).abs() < TOLERANCE);
        }
    }
}
