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
        // 正規分布の確率密度関数と累積分布関数の計算
        // https://marui.hatenablog.com/entry/2023/01/23/194507
        let x = (x - self.mean) / self.std_dev;

        let result = if x == 0.0 {
            0.5
        } else if x > 0.0 {
            Self::calc_cumulative_dist_inner(x)
        } else {
            1.0 - Self::calc_cumulative_dist_inner(-x)
        };

        result
    }

    fn calc_cumulative_dist_inner(x: f64) -> f64 {
        const P: f64 = 0.2316419;
        let t1 = 1.0 / (1.0 + P * x);
        let t2 = t1 * t1;
        let t3 = t2 * t1;
        let t4 = t3 * t1;
        let t5 = t4 * t1;
        1.0 - Self::dnorm(x)
            * (0.319381530 * t1 - 0.356563782 * t2 + 1.781477937 * t3 - 1.821255978 * t4
                + 1.330274429 * t5)
    }

    fn dnorm(x: f64) -> f64 {
        (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt()
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
