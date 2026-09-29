//! Shows what each sampling setting does to a distribution and to generated
//! text, and what sampling costs at a real vocabulary size.
//!
//! Run with: cargo run --release -p ch15-sampling

use ch15_sampling::{Rng, Sampler, SamplingParams, top_logprobs};
use std::collections::HashMap;
use std::time::Instant;

fn main() {
    one_distribution();
    frequencies();
    cost();
    text();
}

const WORDS: [&str; 6] = [" Paris", " Lyon", " France", " the", " a", " banana"];
const LOGITS: [f32; 6] = [4.0, 2.5, 2.0, 1.5, 0.5, -2.0];

/// Part 1: one distribution under different settings.
fn one_distribution() {
    println!("== 1. \"The capital of France is\" -> next-token probabilities");
    print!("   {:<22}", "setting");
    for w in WORDS {
        print!("{w:>9}");
    }
    println!();
    let settings: [(&str, SamplingParams); 7] = [
        ("temperature 1", SamplingParams::default()),
        ("temperature 0.5", with(|p| p.temperature = 0.5)),
        ("temperature 2", with(|p| p.temperature = 2.0)),
        ("greedy (temperature 0)", SamplingParams::greedy()),
        ("top-k 4", with(|p| p.top_k = 4)),
        ("top-p 0.8", with(|p| p.top_p = 0.8)),
        ("min-p 0.1", with(|p| p.min_p = 0.1)),
    ];
    for (name, params) in settings {
        let mut sampler = Sampler::new(params);
        let mut p = [0.0f32; 6];
        for c in sampler.distribution(&LOGITS) {
            p[c.token as usize] = c.p;
        }
        print!("   {name:<22}");
        for v in p {
            if v == 0.0 {
                print!("{:>9}", "-");
            } else {
                print!("{v:>9.3}");
            }
        }
        println!();
    }
    println!();
}

/// Part 2: draws follow the distribution.
fn frequencies() {
    let n = 100_000;
    let mut sampler = Sampler::new(SamplingParams::default());
    let expected: Vec<f32> = sampler.distribution(&LOGITS).iter().map(|c| c.p).collect();
    let mut counts = [0u32; 6];
    for _ in 0..n {
        counts[sampler.sample(&LOGITS) as usize] += 1;
    }
    println!("== 2. {n} draws at temperature 1");
    for ((w, count), p) in WORDS.iter().zip(counts).zip(expected) {
        println!(
            "   {w:<8} drawn {:>6.3} of the time, probability {p:.3}",
            f64::from(count) / f64::from(n)
        );
    }
    println!();
}

/// Part 3: the cost of each setting with SmolLM2's vocabulary.
fn cost() {
    let vocab = 49_152;
    let mut rng = Rng::new(1);
    // Normally distributed logits (Box-Muller), standard deviation 2.
    let logits: Vec<f32> = (0..vocab)
        .map(|_| {
            let (u1, u2) = (rng.next_f64().max(1e-300), rng.next_f64());
            (2.0 * (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()) as f32
        })
        .collect();
    println!("== 3. time per sampled token, {vocab} logits");
    let settings: [(&str, SamplingParams); 7] = [
        ("greedy", SamplingParams::greedy()),
        ("temperature 1, no filter", SamplingParams::default()),
        ("top-k 50", with(|p| p.top_k = 50)),
        ("top-p 0.9", with(|p| p.top_p = 0.9)),
        (
            "top-k 50 + top-p 0.9",
            with(|p| (p.top_k, p.top_p) = (50, 0.9)),
        ),
        ("min-p 0.05", with(|p| p.min_p = 0.05)),
        ("frequency penalty 0.5", with(|p| p.frequency_penalty = 0.5)),
    ];
    for (name, params) in settings {
        let mut sampler = Sampler::new(params);
        let t = time_per_call(200, || {
            sampler.sample(&logits);
        });
        println!("   {name:<26} {t:>8.1} µs");
    }
    let t = time_per_call(200, || {
        std::hint::black_box(top_logprobs(&logits, 5));
    });
    println!("   {:<26} {t:>8.1} µs", "top 5 logprobs");
    println!();
}

/// Part 4: generating text from a word-pair (bigram) model.
fn text() {
    let corpus = include_str!("../data/pride-and-prejudice-1-6.txt");
    let model = Bigram::new(corpus);
    println!(
        "== 4. a bigram model of Pride and Prejudice, chapters 1-6 ({} words, {} distinct)",
        corpus.split_whitespace().count(),
        model.words.len()
    );
    let settings: [(&str, SamplingParams); 5] = [
        ("greedy", SamplingParams::greedy()),
        (
            "greedy, repetition penalty 1.5",
            with(|p| (p.temperature, p.repetition_penalty) = (0.0, 1.5)),
        ),
        ("temperature 1", with(|p| p.seed = 1)),
        (
            "temperature 1, top-p 0.5",
            with(|p| (p.top_p, p.seed) = (0.5, 1)),
        ),
        (
            "temperature 0.7, top-p 0.9",
            with(|p| (p.temperature, p.top_p, p.seed) = (0.7, 0.9, 1)),
        ),
    ];
    let mut logits = vec![0.0; model.words.len()];
    for (name, params) in settings {
        let mut sampler = Sampler::new(params);
        let mut word = model.index["It"];
        sampler.start(&[word]);
        let mut out = vec!["It"];
        for _ in 0..24 {
            if !model.logits(word, &mut logits) {
                break; // the last word of the text: nothing ever followed it
            }
            word = sampler.sample(&logits);
            out.push(&model.words[word as usize]);
        }
        println!("   {name}:");
        println!("     {}", out.join(" "));
    }
}

/// Counts of which word follows which. Its "logits" for the next word are
/// log-probabilities: ln(count(prev, next) / count(prev)).
struct Bigram {
    words: Vec<String>,
    index: HashMap<String, u32>,
    next: Vec<HashMap<u32, u32>>,
}

impl Bigram {
    fn new(text: &str) -> Self {
        let mut model = Self {
            words: Vec::new(),
            index: HashMap::new(),
            next: Vec::new(),
        };
        let ids: Vec<u32> = text.split_whitespace().map(|w| model.id(w)).collect();
        for pair in ids.windows(2) {
            *model.next[pair[0] as usize].entry(pair[1]).or_insert(0) += 1;
        }
        model
    }

    fn id(&mut self, word: &str) -> u32 {
        if let Some(&id) = self.index.get(word) {
            return id;
        }
        let id = self.words.len() as u32;
        self.words.push(word.to_owned());
        self.index.insert(word.to_owned(), id);
        self.next.push(HashMap::new());
        id
    }

    /// Fills `out`; false if no word ever followed `prev`.
    fn logits(&self, prev: u32, out: &mut [f32]) -> bool {
        out.fill(f32::NEG_INFINITY);
        let followers = &self.next[prev as usize];
        let total: u32 = followers.values().sum();
        for (&w, &count) in followers {
            out[w as usize] = (f64::from(count) / f64::from(total)).ln() as f32;
        }
        total > 0
    }
}

fn with(change: impl FnOnce(&mut SamplingParams)) -> SamplingParams {
    let mut p = SamplingParams::default();
    change(&mut p);
    p
}

/// Mean microseconds per call over `n` calls, best of three rounds.
fn time_per_call(n: u32, mut f: impl FnMut()) -> f64 {
    (0..3)
        .map(|_| {
            let start = Instant::now();
            for _ in 0..n {
                f();
            }
            start.elapsed().as_secs_f64() * 1e6 / f64::from(n)
        })
        .fold(f64::INFINITY, f64::min)
}
