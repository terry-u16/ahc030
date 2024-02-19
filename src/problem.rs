pub mod params;

use crate::{
    common::DurationCorrector,
    grid::{Coord, CoordDiff, Map2d},
};
use bitset_fixed::BitSet;
use im_rc::HashMap;
use itertools::Itertools;
use proconio::{input, source::line::LineSource};
use rand::Rng as _;
use rand_pcg::Pcg64Mcg;
use rustc_hash::FxHashSet;
use std::{
    cmp::Reverse,
    fmt::Display,
    io::{self, BufReader, BufWriter, StdoutLock, Write},
    time::{Duration, Instant},
};

use self::params::ParamSuggester;

#[derive(Debug, Clone)]
pub struct Input {
    pub map_size: usize,
    pub oil_count: usize,
    pub eps: f64,
    pub oils: Vec<Oils>,
    pub avg_oil_size: f64,
    pub total_oil_tiles: usize,
    pub hashes: Vec<Map2d<u64>>,
    pub duration_corrector: DurationCorrector,
    pub time_conductor: TimeConductor,
    pub since: Instant,
    pub params: Params,
}

impl Input {
    pub const TIME_LIMIT: Duration = Duration::new(2, 950_000_000);

    fn new(map_size: usize, oil_count: usize, eps: f64, oils: Vec<Oils>) -> Self {
        let mut counts = HashMap::new();

        for oil in &oils {
            let entry = counts.entry(oil.pos.clone()).or_insert(0);
            *entry += 1;
        }

        let mut rng = Pcg64Mcg::new(42);
        let mut hashes = vec![];

        for _ in 0..oil_count {
            let mut map = Map2d::new_with(0, map_size);

            for row in 0..map_size {
                for col in 0..map_size {
                    map[Coord::new(row, col)] = rng.gen();
                }
            }

            hashes.push(map);
        }

        let total_oil_tiles = oils.iter().map(|o| o.len).sum();
        let avg_oil_size = total_oil_tiles as f64 / oil_count as f64;
        let duration_corrector = DurationCorrector::from_env();
        let time_conductor = TimeConductor::new(map_size, oil_count, eps, avg_oil_size);
        let params = Params::new(map_size, oil_count, eps, avg_oil_size);
        let since = Instant::now();

        Self {
            map_size,
            oil_count,
            eps,
            oils,
            total_oil_tiles,
            avg_oil_size,
            hashes,
            duration_corrector,
            time_conductor,
            since,
            params,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Oils {
    pub pos: Vec<Coord>,
    pub width: usize,
    pub height: usize,
    pub len: usize,
    pub bitset: BitSet,
}

impl Oils {
    fn new(mut pos: Vec<Coord>, map_size: usize) -> Self {
        pos.sort();
        let height = pos.iter().map(|c| c.row).max().unwrap() + 1;
        let width = pos.iter().map(|c| c.col).max().unwrap() + 1;
        let len = pos.len();

        let mut bitset = BitSet::new(map_size * map_size);

        for p in &pos {
            bitset.set(p.row * map_size + p.col, true);
        }

        Self {
            pos,
            width,
            height,
            len,
            bitset,
        }
    }

    pub fn get_shifted_bitset(&self, input: &Input, shift: CoordDiff) -> BitSet {
        let result =
            self.bitset.clone() << (shift.dr as usize * input.map_size + shift.dc as usize);
        result
    }
}

impl Display for Oils {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut map = vec![vec![false; self.width]; self.height];

        for p in &self.pos {
            map[p.row][p.col] = true;
        }

        for row in 0..self.height {
            for col in 0..self.width {
                write!(f, "{}", if map[row][col] { '#' } else { '.' })?;
            }

            writeln!(f)?;
        }

        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct Params {
    pub use_multi_dig_solver: f64,
    pub taboo_prob: f64,
    pub max_entropy_len: usize,
}

impl Params {
    pub fn new(map_size: usize, oil_count: usize, eps: f64, avg: f64) -> Self {
        /*
        let use_multi_dig_solver = match std::env::args().nth(1) {
            Some(s) => {
                if s == "1" {
                    1.0
                } else {
                    0.0
                }
            }
            None => ParamSuggester::gen_multi_pred().suggest(map_size, oil_count, eps, avg),
        };
        */

        let use_multi_dig_solver =
            ParamSuggester::gen_multi_pred().suggest(map_size, oil_count, eps, avg);

        let taboo_prob = std::env::args()
            .nth(1)
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.5);
        let max_entropy_len = std::env::args()
            .nth(2)
            .and_then(|s| s.parse().ok())
            .unwrap_or(50);

        Self {
            use_multi_dig_solver,
            taboo_prob,
            max_entropy_len,
        }
    }

    pub fn use_multi_dig_solver(&self) -> bool {
        self.use_multi_dig_solver >= 0.5
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TimeConductor {
    k: f64,
    b: f64,
    ratio: f64,
}

impl TimeConductor {
    fn new(map_size: usize, oil_count: usize, eps: f64, avg: f64) -> Self {
        // ターンごとの実行時間は (1-x)^k + bx とする
        /*
        let k = std::env::args()
            .nth(2)
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| ParamSuggester::gen_k_pred().suggest(map_size, oil_count, eps, avg));

        let b = std::env::args()
            .nth(3)
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| ParamSuggester::gen_b_pred().suggest(map_size, oil_count, eps, avg));

        let ratio = std::env::args()
            .nth(4)
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| ParamSuggester::gen_r_pred().suggest(map_size, oil_count, eps, avg));
        */

        let k = ParamSuggester::gen_k_pred().suggest(map_size, oil_count, eps, avg);
        let b = ParamSuggester::gen_b_pred().suggest(map_size, oil_count, eps, avg);
        let ratio = ParamSuggester::gen_r_pred().suggest(map_size, oil_count, eps, avg);

        Self { k, b, ratio }
    }

    pub fn get_time_table(
        &self,
        elapsed: Duration,
        time_limit: Duration,
        max_turn: usize,
    ) -> Vec<Duration> {
        let mut times_f64 = vec![];

        for t in 0..max_turn {
            let x = t as f64 / max_turn as f64;

            let t = (1.0 - x).powf(self.k) + self.b * x;
            times_f64.push(t);
        }

        for i in 1..max_turn {
            times_f64[i] += times_f64[i - 1];
        }

        let total_time = times_f64.last().unwrap();

        let mut durations = vec![];

        let remaining = time_limit.saturating_sub(elapsed);

        for &t in times_f64.iter() {
            let t = elapsed + remaining.mul_f64(t / total_time);
            durations.push(t);
        }

        durations
    }

    pub fn k(&self) -> f64 {
        self.k
    }

    pub fn b(&self) -> f64 {
        self.b
    }

    pub fn phase_ratio(&self) -> f64 {
        self.ratio
    }
}

pub struct Judge<'a> {
    source: LineSource<BufReader<io::Stdin>>,
    writer: BufWriter<StdoutLock<'a>>,
    wa: FxHashSet<Vec<Coord>>,
    comment_buf: Vec<String>,
    color_buf: Vec<Map2d<f64>>,
    query_limit: usize,
    query_count: usize,
    show_comment: bool,
}

#[allow(dead_code)]
impl<'a> Judge<'a> {
    pub fn new() -> Self {
        let source = LineSource::new(BufReader::new(io::stdin()));
        let writer = BufWriter::new(io::stdout().lock());
        let show_comment = std::env::var("AHC030_SHOW_COMMENT").is_ok_and(|s| s == "1");

        Self {
            source,
            writer,
            wa: FxHashSet::default(),
            comment_buf: vec![],
            color_buf: vec![],
            query_limit: 0,
            query_count: 0,
            show_comment,
        }
    }

    pub fn read_input(&mut self) -> Input {
        input! {
            from &mut self.source,
            n: usize,
            m: usize,
            eps: f64,
        }

        self.query_limit = n * n * 2;

        let mut oils = vec![];

        for _ in 0..m {
            input! {
                from &mut self.source,
                d: usize,
            }

            let mut pos = vec![];

            for _ in 0..d {
                input! {
                    from &mut self.source,
                    row: usize,
                    col: usize,
                }

                pos.push(Coord::new(row, col));
            }

            oils.push(Oils::new(pos, n));
        }

        oils.sort_by_key(|o| Reverse(o.len));

        Input::new(n, m, eps, oils)
    }

    pub fn query_count(&self) -> usize {
        self.query_count
    }

    pub fn max_query_count(&self) -> usize {
        self.query_limit
    }

    pub fn can_query(&self) -> bool {
        self.query_count < self.query_limit
    }

    pub fn remaining_query_count(&self) -> usize {
        self.query_limit - self.query_count
    }

    pub fn query_single(&mut self, coord: Coord) -> i32 {
        self.query_multiple(&[coord])
    }

    pub fn query_multiple(&mut self, coords: &[Coord]) -> i32 {
        assert!(coords.len() > 0);

        if self.query_count >= self.query_limit {
            // do nothing
            return 0;
        }

        self.query_count += 1;
        self.flush_comments();
        let len = coords.len();
        let coords = coords
            .iter()
            .map(|c| format!("{} {}", c.row, c.col))
            .join(" ");

        writeln!(self.writer, "q {} {}", len, coords).unwrap();
        self.writer.flush().unwrap();

        input! {
            from &mut self.source,
            value: i32
        }

        value
    }

    pub fn answer(&mut self, coords: &[Coord]) -> Result<(), ()> {
        if self.query_count >= self.query_limit {
            // do nothing
            return Ok(());
        }

        let len = coords.len();
        let coords_vec = coords.iter().copied().collect_vec();

        if self.wa.contains(&coords_vec) {
            return Err(());
        }

        self.query_count += 1;
        self.flush_comments();

        let coords_str = coords
            .iter()
            .map(|c| format!("{} {}", c.row, c.col))
            .join(" ");

        writeln!(self.writer, "a {} {}", len, coords_str).unwrap();
        self.writer.flush().unwrap();

        input! {
            from &mut self.source,
            value: i32
        }

        if value == 1 {
            Ok(())
        } else {
            self.wa.insert(coords_vec);
            Err(())
        }
    }

    pub fn comment(&mut self, message: &str) {
        self.comment_buf.push(message.to_string());
    }

    pub fn comment_colors(&mut self, colors: &Map2d<f64>) {
        self.color_buf.push(colors.clone());
    }

    fn flush_comments(&mut self) {
        if !self.show_comment {
            return;
        }

        for comment in &self.comment_buf {
            writeln!(self.writer, "# {}", comment).unwrap();
        }

        self.comment_buf.clear();

        for colors in self.color_buf.iter() {
            for row in 0..colors.size {
                for col in 0..colors.size {
                    let c = Coord::new(row, col);
                    let v = ((1.0 - colors[c].clamp(0.0, 1.0)) * 255.0).round() as u8;
                    writeln!(self.writer, "#c {row} {col} #{:02x}{:02x}{:02x}", v, v, 255).unwrap();
                }
            }
        }

        self.writer.flush().unwrap();
        self.color_buf.clear();
    }
}
