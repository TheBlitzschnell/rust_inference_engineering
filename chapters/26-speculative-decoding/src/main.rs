//! Speculative decoding on SmolLM2: SmolLM2-360M-Instruct is the target,
//! SmolLM2-135M-Instruct (in bf16, int8 or 4 bits) or n-gram lookup drafts.
//!
//! Run with: cargo run --release -p ch26-speculative-decoding [cost|ceiling|draft|ngram|sampling]
//! (needs both models: ./tools/download_model.sh and ./tools/download_model.sh 360M)

use ch07_threads::SpinPool;
use ch14_kv_cache::{KvCache, Matrix, Model, Scratch};
use ch15_sampling::Rng;
use ch16_real_model::{DenseBf16, Message, Placement, Tokenizer, chat_prompt, load_bf16};
use ch17_profiling::{map_matrices, measure};
use ch18_int8::{Activations, Granularity, Q8Matrix};
use ch19_4bit::{Q4Matrix, Scheme};
use ch23_continuous_batching::PackedBf16;
use ch26_speculative_decoding::{
    Draft, Drafter, Generation, Mode, ModelDrafter, NgramDrafter, Proposal, SpecStats, Target,
};
use std::path::Path;
use std::time::{Duration, Instant};

const CHAT: [&str; 3] = [
    "Explain why the sky is blue in a short paragraph.",
    "Write a short story about a robot who learns to paint.",
    "What are three tips for staying healthy? Explain each.",
];

const EDIT: [&str; 3] = [
    "Correct the spelling in this text and write it out in full: Teh quick brown fox jumpd over the lazy dog. It was a sunny day and the fox was very hapy. The dog did not care at al, it just kept sleeping under the old oak tree.",
    "Change every name in this paragraph to Alex and write the whole paragraph again: Maria went to the market with John. Maria bought apples, and John bought bread. Later, Maria and John cooked dinner together for their friends.",
    "Add a comment above each line of this Python function and return the whole function:\n\ndef average(values):\n    total = sum(values)\n    count = len(values)\n    return total / count",
];

fn main() {
    let small = ch09_safetensors::smollm2_dir("135m");
    let big = ch09_safetensors::smollm2_dir("360m");
    if !small.join("model.safetensors").exists() || !big.join("model.safetensors").exists() {
        eprintln!(
            "models not found in {} and {}: run ./tools/download_model.sh and ./tools/download_model.sh 360M",
            small.display(),
            big.display()
        );
        return;
    }
    let tok = Tokenizer::from_file(&small.join("tokenizer.json")).expect("tokenizer");
    let target = packed(&big);
    let mut pool = SpinPool::with_all_cores();
    let part = std::env::args().nth(1).unwrap_or_default();
    let run = |name: &str| part.is_empty() || part == name;
    let mut bench = Bench {
        target: Target::new(&target, 1024),
        stop: vec![tok.token_id("<|im_end|>").expect("end-of-turn token")],
        tok: &tok,
        plain_chat: None,
    };
    // Warm up: the first generation pays for page faults and cold caches.
    let mut none = NgramDrafter { min_n: 1, max_n: 1 };
    bench.generate(&mut pool, &mut none, CHAT[0], 0, Mode::Greedy, 0);
    if run("cost") {
        cost(&small, &target, &tok, &mut pool);
    }
    if run("ceiling") {
        ceiling(&mut bench, &mut pool);
    }
    if run("draft") {
        draft(&small, &mut bench, &mut pool);
    }
    if run("ngram") {
        ngram(&mut bench, &mut pool);
    }
    if run("sampling") {
        sampling(&small, &mut bench, &mut pool);
    }
}

fn bf16(dir: &Path) -> Model<DenseBf16> {
    load_bf16(dir, Placement::Mapped).expect("model").0
}

fn to_f32(m: &DenseBf16) -> Vec<f32> {
    m.values().iter().map(|v| v.to_f32()).collect()
}

/// Chapter 23's packed `bf16` weights.
fn packed(dir: &Path) -> Model<PackedBf16> {
    map_matrices(bf16(dir), |w| {
        PackedBf16::new(w.values(), w.rows(), w.cols())
    })
}

/// Chapter 18's int8 blocks, with int8 activations.
fn int8(dir: &Path) -> Model<Q8Matrix> {
    map_matrices(bf16(dir), |m| {
        let v = to_f32(&m);
        Q8Matrix::new(
            &v,
            m.rows(),
            m.cols(),
            Granularity::PerBlock,
            Activations::Int8PerBlock,
        )
    })
}

/// Chapter 19's 4-bit blocks (min-max searched, groups of 64, W4A8).
fn int4(dir: &Path) -> Model<Q4Matrix> {
    map_matrices(bf16(dir), |m| {
        let v = to_f32(&m);
        Q4Matrix::new(&v, m.rows(), m.cols(), 64, Scheme::MinMaxSearch, true)
    })
}

fn ask(tok: &Tokenizer, question: &str) -> Vec<u32> {
    tok.encode(&chat_prompt(&[Message {
        role: "user",
        content: question,
    }]))
}

/// The time to process `n` tokens after a chat prompt, in milliseconds.
fn pass_ms<W: Matrix>(model: &Model<W>, tok: &Tokenizer, pool: &mut SpinPool, n: usize) -> f64 {
    let mut cache = KvCache::new(&model.config, 512);
    let mut scratch = Scratch::new(&model.config, 64, 512);
    model.forward_last(pool, &ask(tok, CHAT[0]), &mut cache, &mut scratch);
    let len = cache.len();
    let tokens: Vec<u32> = (0..n as u32).map(|i| 1000 + i).collect();
    let stats = measure(3, 20, || {
        cache.truncate(len);
        model.forward_all(pool, &tokens, &mut cache, &mut scratch);
    });
    stats.median.as_secs_f64() * 1e3
}

/// Part 1: what one forward pass costs, by the number of tokens in it.
fn cost(small: &Path, target: &Model<PackedBf16>, tok: &Tokenizer, pool: &mut SpinPool) {
    let ns = [1, 2, 3, 4, 5, 6, 8, 16];
    let ms: Vec<f64> = ns.iter().map(|&n| pass_ms(target, tok, pool, n)).collect();
    println!("== 1. one forward pass of SmolLM2-360M over n new tokens (after a 35-token prompt)");
    let cells = |f: &dyn Fn(usize) -> String| -> String { (0..ns.len()).map(f).collect() };
    println!(
        "   n                    {}",
        cells(&|i| format!("{:>7}", ns[i]))
    );
    println!(
        "   time (ms)            {}",
        cells(&|i| format!("{:>7.1}", ms[i]))
    );
    println!(
        "   relative to n = 1    {}",
        cells(&|i| format!("{:>6.2}x", ms[i] / ms[0]))
    );
    println!("   one token of the draft models:");
    let drafts = [
        ("135M, packed bf16", pass_ms(&packed(small), tok, pool, 1)),
        ("135M, int8", pass_ms(&int8(small), tok, pool, 1)),
        ("135M, 4-bit", pass_ms(&int4(small), tok, pool, 1)),
    ];
    for (name, d) in drafts {
        println!(
            "      {name:<18} {d:>5.1} ms, {:.2} of a target token",
            d / ms[0]
        );
    }
    println!();
}

/// The target, its stop token, and a way to run a set of prompts.
struct Bench<'m> {
    target: Target<'m, PackedBf16>,
    stop: Vec<u32>,
    tok: &'m Tokenizer,
    /// Plain greedy decoding of the chat questions, once computed.
    plain_chat: Option<Run>,
}

/// The totals of a set of generations.
#[derive(Clone, Default)]
struct Run {
    outputs: Vec<Vec<u32>>,
    stats: SpecStats,
    time: Duration,
}

impl Run {
    fn tokens(&self) -> usize {
        self.outputs.iter().map(Vec::len).sum()
    }

    fn ms_per_token(&self) -> f64 {
        self.time.as_secs_f64() * 1e3 / self.tokens() as f64
    }
}

impl Bench<'_> {
    fn generate(
        &mut self,
        pool: &mut SpinPool,
        drafter: &mut dyn Drafter,
        question: &str,
        k: usize,
        mode: Mode,
        seed: u64,
    ) -> (Vec<u32>, SpecStats) {
        let g = Generation {
            max_new: 200,
            k,
            mode,
            seed,
            stop: self.stop.clone(),
        };
        self.target
            .generate(pool, drafter, &ask(self.tok, question), &g)
    }

    fn run(
        &mut self,
        pool: &mut SpinPool,
        drafter: &mut dyn Drafter,
        questions: &[&str],
        k: usize,
        mode: Mode,
    ) -> Run {
        let mut run = Run::default();
        let start = Instant::now();
        for (i, q) in questions.iter().enumerate() {
            let (out, stats) = self.generate(pool, drafter, q, k, mode, i as u64);
            run.outputs.push(out);
            run.stats.add(&stats);
        }
        run.time = start.elapsed();
        run
    }

    /// Plain greedy decoding (no drafts) of `questions`; the chat
    /// questions' run is kept for the parts that compare with it.
    fn plain(&mut self, pool: &mut SpinPool, questions: &[&str]) -> Run {
        let chat = questions == CHAT;
        if chat && let Some(run) = &self.plain_chat {
            return run.clone();
        }
        let mut none = NgramDrafter { min_n: 1, max_n: 1 };
        let run = self.run(pool, &mut none, questions, 0, Mode::Greedy);
        if chat {
            self.plain_chat = Some(run.clone());
        }
        run
    }
}

fn header(base: &Run) {
    println!(
        "   plain decoding: {} tokens, {:.1} ms per token",
        base.tokens(),
        base.ms_per_token()
    );
    println!(
        "   drafter             k   accepted   tokens/pass   per token   speedup   same answers"
    );
}

/// Prints a row comparing `run` with plain decoding (`base`).
fn row(name: &str, k: usize, run: &Run, base: &Run) {
    let same = run
        .outputs
        .iter()
        .zip(&base.outputs)
        .filter(|(a, b)| a == b)
        .count();
    println!(
        "   {name:<18} {k:>2}   {:>7.0}%   {:>11.2}   {:>6.1} ms   {:>6.2}x   {same} of {}",
        100.0 * run.stats.acceptance(),
        run.stats.tokens_per_pass(),
        run.ms_per_token(),
        base.ms_per_token() / run.ms_per_token(),
        run.outputs.len()
    );
}

/// A drafter that knows the answer: it replays the target's own greedy
/// output. Always right and free, it shows the best speculation can do.
struct Oracle {
    prompt_len: usize,
    answer: Vec<u32>,
}

impl Drafter for Oracle {
    fn propose(
        &mut self,
        _pool: &mut SpinPool,
        context: &[u32],
        k: usize,
        _mode: Mode,
        _rng: &mut Rng,
    ) -> Draft {
        let done = context.len() - self.prompt_len;
        let tokens: Vec<u32> = self.answer.iter().skip(done).take(k).copied().collect();
        Draft {
            proposals: vec![Proposal::Exact; tokens.len()],
            tokens,
        }
    }
}

/// Part 2: the best case, a drafter that is free and always right.
fn ceiling(bench: &mut Bench, pool: &mut SpinPool) {
    println!("== 2. the ceiling: a drafter that knows the answer (three chat answers)");
    let base = bench.plain(pool, &CHAT);
    header(&base);
    for k in [1, 2, 4, 8, 16] {
        let mut run = Run::default();
        let start = Instant::now();
        for (q, answer) in CHAT.iter().zip(&base.outputs) {
            let mut oracle = Oracle {
                prompt_len: ask(bench.tok, q).len(),
                answer: answer.clone(),
            };
            let (out, stats) = bench.generate(pool, &mut oracle, q, k, Mode::Greedy, 0);
            run.outputs.push(out);
            run.stats.add(&stats);
        }
        run.time = start.elapsed();
        row("oracle", k, &run, &base);
    }
    println!();
}

/// Part 3: SmolLM2-135M drafting for SmolLM2-360M, in three precisions.
fn draft(small: &Path, bench: &mut Bench, pool: &mut SpinPool) {
    println!("== 3. SmolLM2-135M drafts for SmolLM2-360M, greedy (three chat answers)");
    let base = bench.plain(pool, &CHAT);
    header(&base);
    let mut d = ModelDrafter::new(packed(small), 1024);
    for k in [1, 2, 3, 4] {
        let run = bench.run(pool, &mut d, &CHAT, k, Mode::Greedy);
        row("135M bf16", k, &run, &base);
    }
    let mut d = ModelDrafter::new(int8(small), 1024);
    for k in [2, 3] {
        let run = bench.run(pool, &mut d, &CHAT, k, Mode::Greedy);
        row("135M int8", k, &run, &base);
    }
    let mut d = ModelDrafter::new(int4(small), 1024);
    for k in [2, 3] {
        let run = bench.run(pool, &mut d, &CHAT, k, Mode::Greedy);
        row("135M 4-bit", k, &run, &base);
    }
    println!();
}

/// Part 4: no draft model, only n-gram lookup, on answers that copy from
/// their prompt and on answers that do not.
fn ngram(bench: &mut Bench, pool: &mut SpinPool) {
    for (name, questions) in [("editing tasks", EDIT), ("chat answers", CHAT)] {
        println!("== 4. n-gram lookup, greedy: three {name}");
        let base = bench.plain(pool, &questions);
        header(&base);
        for (min_n, k) in [(1, 4), (2, 4), (2, 8)] {
            let mut d = NgramDrafter { min_n, max_n: 4 };
            let run = bench.run(pool, &mut d, &questions, k, Mode::Greedy);
            row(&format!("n-gram, {min_n} to 4"), k, &run, &base);
        }
        println!();
    }
    let mut d = NgramDrafter { min_n: 2, max_n: 4 };
    let (out, _) = bench.generate(pool, &mut d, EDIT[1], 8, Mode::Greedy, 0);
    println!("   the second editing answer: {:?}", bench.tok.decode(&out));
    println!();
}

/// Part 5: sampling at a temperature, with rejection sampling, against
/// greedy decoding with the same drafter and k.
fn sampling(small: &Path, bench: &mut Bench, pool: &mut SpinPool) {
    println!("== 5. 135M bf16 drafts, k = 2, greedy and sampled (three chat answers)");
    println!("   decoding           accepted   tokens/pass   plain   speculative   speedup");
    let mut d = ModelDrafter::new(packed(small), 1024);
    for (name, mode) in [
        ("greedy", Mode::Greedy),
        ("temperature 0.5", Mode::Sample { temperature: 0.5 }),
        ("temperature 1.0", Mode::Sample { temperature: 1.0 }),
    ] {
        let base = if mode == Mode::Greedy {
            bench.plain(pool, &CHAT)
        } else {
            let mut none = NgramDrafter { min_n: 1, max_n: 1 };
            bench.run(pool, &mut none, &CHAT, 0, mode)
        };
        let run = bench.run(pool, &mut d, &CHAT, 2, mode);
        println!(
            "   {name:<16} {:>9.0}%   {:>11.2}   {:>2.1} ms   {:>8.1} ms   {:>6.2}x",
            100.0 * run.stats.acceptance(),
            run.stats.tokens_per_pass(),
            base.ms_per_token(),
            run.ms_per_token(),
            base.ms_per_token() / run.ms_per_token()
        );
    }
    println!();
}
