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
        0.5 * (1.0 + libm::erf((x - self.mean) / (self.std_dev * 1.41421356237)))
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
