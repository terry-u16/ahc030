use crate::{
    common::DurationCorrector,
    grid::{Coord, Map2d},
};
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
};

#[derive(Debug, Clone)]
pub struct Input {
    pub map_size: usize,
    pub oil_count: usize,
    pub eps: f64,
    pub oils: Vec<Oils>,
    pub total_oil_tiles: usize,
    pub hashes: Vec<Map2d<u64>>,
    pub duration_corrector: DurationCorrector,
}

impl Input {
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
        let duration_corrector = DurationCorrector::from_env();

        Self {
            map_size,
            oil_count,
            eps,
            oils,
            total_oil_tiles,
            hashes,
            duration_corrector,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Oils {
    pub pos: Vec<Coord>,
    pub width: usize,
    pub height: usize,
    pub len: usize,
}

impl Oils {
    fn new(mut pos: Vec<Coord>) -> Self {
        pos.sort();
        let height = pos.iter().map(|c| c.row).max().unwrap() + 1;
        let width = pos.iter().map(|c| c.col).max().unwrap() + 1;
        let len = pos.len();

        Self {
            pos,
            width,
            height,
            len,
        }
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

            oils.push(Oils::new(pos));
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
