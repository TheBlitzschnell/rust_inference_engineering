//! Structured output on SmolLM2: extracting fields as JSON, with and
//! without constraints, and what a token mask costs.
//!
//! Run with: cargo run --release -p ch27-structured-output [extract|mask|budget]
//! (needs the models: ./tools/download_model.sh and, for the 360M rows,
//! ./tools/download_model.sh 360M)

use ch07_threads::SpinPool;
use ch14_kv_cache::{KvCache, Matrix, Model, Scratch};
use ch15_sampling::{Sampler, SamplingParams};
use ch16_real_model::{DenseBf16, Message, Placement, Tokenizer, chat_prompt, load_bf16};
use ch17_profiling::{map_matrices, measure};
use ch23_continuous_batching::PackedBf16;
use ch27_structured_output::{Matcher, Schema, Vocab, allowed_naive, generate_json};
use serde_json::Value;
use std::path::Path;
use std::time::{Duration, Instant};

/// Short texts and the fields they contain: (text, name, age, city).
const PEOPLE: [(&str, &str, i64, &str); 10] = [
    (
        "Maria Lopez is a 34-year-old nurse who lives in Madrid.",
        "Maria Lopez",
        34,
        "Madrid",
    ),
    (
        "Kenji Watanabe, 52, teaches history in Osaka.",
        "Kenji Watanabe",
        52,
        "Osaka",
    ),
    (
        "Born 27 years ago, Amara Okafor now works as a software engineer in Lagos.",
        "Amara Okafor",
        27,
        "Lagos",
    ),
    (
        "Lukas Becker (41) runs a bakery in Hamburg.",
        "Lukas Becker",
        41,
        "Hamburg",
    ),
    (
        "The painter Sofia Rossi, aged 29, has a studio in Florence.",
        "Sofia Rossi",
        29,
        "Florence",
    ),
    (
        "At 63, retired pilot John Carter spends his days in Denver.",
        "John Carter",
        63,
        "Denver",
    ),
    (
        "Priya Sharma is 38 and works as a doctor in Mumbai.",
        "Priya Sharma",
        38,
        "Mumbai",
    ),
    (
        "Elena Petrova, a 45-year-old architect, lives in Sofia.",
        "Elena Petrova",
        45,
        "Sofia",
    ),
    (
        "Tom Nguyen just turned 19; he studies music in Toronto.",
        "Tom Nguyen",
        19,
        "Toronto",
    ),
    (
        "Chen Wei, 56, is a fisherman from Qingdao.",
        "Chen Wei",
        56,
        "Qingdao",
    ),
];

fn person() -> Schema {
    Schema::Object(vec![
        ("name".into(), Schema::String { max_len: 40 }),
        ("age".into(), Schema::Integer),
        ("city".into(), Schema::String { max_len: 30 }),
    ])
}

fn main() {
    let small = ch09_safetensors::smollm2_dir("135m");
    if !small.join("model.safetensors").exists() {
        eprintln!(
            "model not found in {}: run ./tools/download_model.sh",
            small.display()
        );
        return;
    }
    let tok = Tokenizer::from_file(&small.join("tokenizer.json")).expect("tokenizer");
    let part = std::env::args().nth(1).unwrap_or_default();
    let run = |name: &str| part.is_empty() || part == name;
    if run("extract") {
        extract(&tok);
    }
    if run("mask") {
        mask_cost(&tok);
    }
    if run("budget") {
        budget(&small, &tok);
    }
}

fn packed(dir: &Path) -> Model<PackedBf16> {
    map_matrices(
        load_bf16(dir, Placement::Mapped).expect("model").0,
        |w: DenseBf16| PackedBf16::new(w.values(), w.rows(), w.cols()),
    )
}

/// The request, asking for JSON (`ask_json`) or simply asking the question.
fn prompt(tok: &Tokenizer, text: &str, ask_json: bool) -> Vec<u32> {
    let question = if ask_json {
        "Extract the person's name, age and city from the text. Reply with only a JSON object with the keys \"name\", \"age\" and \"city\"."
    } else {
        "Who is the person in this text, how old are they, and where do they live?"
    };
    tok.encode(&chat_prompt(&[Message {
        role: "user",
        content: &format!("{question}\n\nText: {text}"),
    }]))
}

/// The tokenizer's vocabulary for the constrained generator.
fn vocab(tok: &Tokenizer) -> Vocab<'_> {
    let eos = tok.token_id("<|im_end|>").expect("end-of-turn token");
    let bytes = (0..tok.vocab_size() as u32)
        .map(|id| tok.token_bytes(id))
        .collect();
    Vocab::new(bytes, |id| tok.is_special(id), eos)
}

/// The first `{...}` in `text` that parses as JSON, if any (as a lenient
/// client would cut it out of a Markdown code block).
fn find_object(text: &str) -> Option<Value> {
    let start = text.find('{')?;
    let mut depth = 0;
    for (i, c) in text[start..].char_indices() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return serde_json::from_str(&text[start..=start + i]).ok();
                }
            }
            _ => {}
        }
    }
    None
}

/// Whether `value` has exactly the three fields, with the right types (in
/// any order: an object's keys have no order once parsed).
fn conforms(value: &Value) -> bool {
    value.as_object().is_some_and(|o| {
        o.len() == 3 && o["name"].is_string() && o["age"].is_i64() && o["city"].is_string()
    })
}

/// How many of the three fields `value` got right.
fn correct(value: &Value, (_, name, age, city): (&str, &str, i64, &str)) -> usize {
    usize::from(value["name"].as_str() == Some(name))
        + usize::from(value["age"].as_i64() == Some(age))
        + usize::from(value["city"].as_str() == Some(city))
}

#[derive(Default)]
struct Tally {
    whole: usize,
    found: usize,
    conform: usize,
    fields: usize,
    tokens: usize,
    time: Duration,
    mask: Duration,
}

impl Tally {
    fn add(&mut self, text: &str, person: (&str, &str, i64, &str)) {
        if serde_json::from_str::<Value>(text.trim()).is_ok() {
            self.whole += 1;
        }
        if let Some(v) = find_object(text) {
            self.found += 1;
            if conforms(&v) {
                self.conform += 1;
                self.fields += correct(&v, person);
            }
        }
    }

    fn print(&self, name: &str) {
        println!(
            "   {name:<37} {:>5} {:>8} {:>9} {:>6}/30 {:>7.1} {:>9.1} ms {:>8.2} ms",
            self.whole,
            self.found,
            self.conform,
            self.fields,
            self.tokens as f64 / PEOPLE.len() as f64,
            self.time.as_secs_f64() * 1e3 / self.tokens as f64,
            self.mask.as_secs_f64() * 1e3 / self.tokens as f64,
        );
    }
}

/// Part 1: ten extraction requests, unconstrained and constrained, with
/// SmolLM2-135M and SmolLM2-360M (greedy).
fn extract(tok: &Tokenizer) {
    println!("== 1. extract {{name, age, city}} as JSON from ten short texts (greedy)");
    println!(
        "                                        valid JSON:          fields  tokens   per decoded token:"
    );
    println!(
        "                                        whole  contains  conforming right   each      total    masking"
    );
    let vocab = vocab(tok);
    let schema = person();
    let stop = vocab.eos;
    let mut pool = SpinPool::with_all_cores();
    let mut example = None;
    for size in ["135m", "360m"] {
        let dir = ch09_safetensors::smollm2_dir(size);
        if !dir.join("model.safetensors").exists() {
            println!("   (SmolLM2-{size} not downloaded: skipped)");
            continue;
        }
        let model = packed(&dir);
        let mut cache = KvCache::new(&model.config, 512);
        let mut scratch = Scratch::new(&model.config, 256, 512);
        let (mut free, mut constrained) = (Tally::default(), Tally::default());
        let mut unasked = Tally::default();
        let mut wrong_example = None;
        let mut compact = Tally::default();
        for (i, &person) in PEOPLE.iter().enumerate() {
            let p = prompt(tok, person.0, true);
            // Unconstrained: plain greedy decoding up to the end of the turn
            // (timed after the prompt, like the constrained decode below).
            cache.clear();
            let mut logits = model
                .forward_last(&mut pool, &p, &mut cache, &mut scratch)
                .to_vec();
            let start = Instant::now();
            let mut out = Vec::new();
            for _ in 0..96 {
                let t = argmax(&logits);
                if t == stop {
                    break;
                }
                out.push(t);
                logits.copy_from_slice(model.forward_last(
                    &mut pool,
                    &[t],
                    &mut cache,
                    &mut scratch,
                ));
            }
            free.time += start.elapsed();
            free.tokens += out.len().max(1);
            let text = tok.decode(&out);
            free.add(&text, person);
            if i == 0 && size == "135m" {
                example = Some(text);
            }
            // Constrained.
            let mut sampler = Sampler::new(SamplingParams::greedy());
            let json = generate_json(
                &model,
                &mut pool,
                &mut cache,
                &mut scratch,
                &vocab,
                Matcher::new(&schema),
                &mut sampler,
                &p,
                96,
            );
            constrained.time += json.mask_time + json.model_time;
            constrained.mask += json.mask_time;
            constrained.tokens += json.tokens.len().max(1);
            constrained.add(&json.text, person);
            // Constrained, with a prompt that does not mention JSON.
            let mut sampler = Sampler::new(SamplingParams::greedy());
            let plain = generate_json(
                &model,
                &mut pool,
                &mut cache,
                &mut scratch,
                &vocab,
                Matcher::new(&schema),
                &mut sampler,
                &prompt(tok, person.0, false),
                96,
            );
            unasked.time += plain.mask_time + plain.model_time;
            unasked.mask += plain.mask_time;
            unasked.tokens += plain.tokens.len().max(1);
            unasked.add(&plain.text, person);
            let wrong = find_object(&plain.text).is_none_or(|v| correct(&v, person) < 3);
            if wrong && size == "135m" && wrong_example.is_none() {
                wrong_example = Some((person.0, plain.text.clone()));
            }
            // The same, with no whitespace allowed outside strings.
            let mut sampler = Sampler::new(SamplingParams::greedy());
            let tight = generate_json(
                &model,
                &mut pool,
                &mut cache,
                &mut scratch,
                &vocab,
                Matcher::with_max_whitespace(&schema, 0),
                &mut sampler,
                &prompt(tok, person.0, false),
                96,
            );
            compact.time += tight.mask_time + tight.model_time;
            compact.mask += tight.mask_time;
            compact.tokens += tight.tokens.len().max(1);
            compact.add(&tight.text, person);
            if i == 0 && size == "135m" {
                println!("   SmolLM2-135M on {:?}", person.0);
                println!(
                    "      unconstrained:          {:?}",
                    example.take().unwrap_or_default()
                );
                println!("      constrained:            {:?}", json.text);
                println!("      constrained, not asked: {:?}", plain.text);
            }
        }
        let name = format!("SmolLM2-{}", size.to_uppercase());
        free.print(&format!("{name}, free"));
        constrained.print(&format!("{name}, constrained"));
        unasked.print(&format!("{name}, constrained, not asked"));
        if let Some((text, answer)) = wrong_example {
            println!("      e.g. not asked, on {text:?}:");
            println!("      {answer:?}");
        }
        compact.print(&format!("{name}, not asked, compact"));
    }
    println!();
}

/// The index of the largest logit.
fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0;
    for (i, &v) in logits.iter().enumerate() {
        if v > logits[best] {
            best = i;
        }
    }
    best as u32
}

/// Part 2: the cost of one token mask in four matcher states: testing
/// every token, walking the trie, and the trie with the string shortcut.
fn mask_cost(tok: &Tokenizer) {
    let vocab = vocab(tok);
    let list: Vec<(u32, &[u8])> = (0..tok.vocab_size() as u32)
        .filter(|&id| !tok.is_special(id) && !tok.token_bytes(id).is_empty())
        .map(|id| (id, tok.token_bytes(id)))
        .collect();
    println!(
        "== 2. one token mask over {} tokens (trie of {} nodes), single thread",
        list.len(),
        vocab.trie.nodes()
    );
    println!(
        "   after                                  allowed   each token     trie walk     with shortcut"
    );
    let schema = person();
    let ms = |s: ch17_profiling::Stats| s.median.as_secs_f64() * 1e3;
    for text in [
        "",
        "{\"name\": \"",
        "{\"name\": \"Maria Lopez\", \"age\": ",
        "{\"name\": \"Maria Lopez\", \"age\": 34",
    ] {
        let mut m = Matcher::new(&schema);
        assert!(m.push_all(text.as_bytes()));
        let mut mask = vec![false; tok.vocab_size()];
        let naive = measure(2, 10, || {
            mask.fill(false);
            allowed_naive(&list, &m, &mut mask);
        });
        let walk = measure(2, 10, || {
            mask.fill(false);
            vocab.trie.walk(&m, &mut mask);
        });
        let shortcut = measure(2, 20, || {
            mask.fill(false);
            vocab.trie.allowed(&m, &mut mask);
        });
        let allowed = mask.iter().filter(|&&a| a).count();
        println!(
            "   {:<38} {allowed:>7} {:>9.2} ms {:>10.2} ms {:>13.2} ms",
            format!("{text:?}"),
            ms(naive),
            ms(walk),
            ms(shortcut),
        );
    }
    println!();
}

/// Part 3: a budget too small for the answer.
fn budget(dir: &Path, tok: &Tokenizer) {
    println!("== 3. the same request with a budget of 8 tokens");
    let model = packed(dir);
    let vocab = vocab(tok);
    let schema = person();
    let mut pool = SpinPool::with_all_cores();
    let mut cache = KvCache::new(&model.config, 512);
    let mut scratch = Scratch::new(&model.config, 256, 512);
    let mut sampler = Sampler::new(SamplingParams::greedy());
    let p = prompt(tok, PEOPLE[0].0, true);
    let json = generate_json(
        &model,
        &mut pool,
        &mut cache,
        &mut scratch,
        &vocab,
        Matcher::new(&schema),
        &mut sampler,
        &p,
        8,
    );
    let generated = json.text.len() - json.completed;
    println!(
        "   generated {:?}, completed with {:?}",
        &json.text[..generated],
        &json.text[generated..]
    );
    println!();
}
