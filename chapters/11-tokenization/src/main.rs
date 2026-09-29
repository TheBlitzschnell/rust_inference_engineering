//! Trains BPE tokenizers on this course's own lessons and looks at what
//! they do.
//!
//! Run with: cargo run --release -p ch11-tokenization

use std::collections::HashMap;
use std::hint::black_box;
use std::path::Path;
use std::time::Instant;

use ch11_tokenization::{Bpe, StreamDecoder};

/// The text of chapters 1-10's lessons: about 400 KB of English with code.
fn corpus() -> String {
    let chapters = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let mut dirs: Vec<_> = std::fs::read_dir(&chapters)
        .expect("chapters directory")
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.as_bytes()[0].is_ascii_digit() && n < "11")
        })
        .collect();
    dirs.sort();
    dirs.iter()
        .filter_map(|d| std::fs::read_to_string(d.join("README.md")).ok())
        .collect::<Vec<_>>()
        .join("\n")
}

fn main() {
    let text = corpus();
    println!(
        "corpus: lessons of chapters 1-10, {} bytes, {} chars\n",
        text.len(),
        text.chars().count()
    );

    println!("== 1. vocabulary size vs compression");
    println!("   vocab | training time | tokens | bytes per token");
    let mut trained = Vec::new();
    for vocab in [256, 512, 1024, 2048] {
        let start = Instant::now();
        let bpe = Bpe::train(&text, vocab);
        let t = start.elapsed();
        let n = bpe.encode(&text).len();
        println!(
            "   {vocab:>5} | {:>13} | {n:>6} | {:>6.2}",
            format!("{t:.2?}"),
            text.len() as f64 / n as f64
        );
        trained.push(bpe);
    }
    let bpe = trained.pop().expect("trained tokenizers");
    println!();

    println!("== 2. the first and last merges learned (vocab 2048)");
    let show = |id: u32| format!("{:?}", String::from_utf8_lossy(bpe.token_bytes(id)));
    let first: Vec<String> = (256..266).map(show).collect();
    let last: Vec<String> = (2038..2048).map(show).collect();
    println!("   first: {}", first.join(" "));
    println!("   last:  {}", last.join(" "));
    println!();

    println!("== 3. how some sentences split (vocab 2048)");
    for s in [
        "The KV cache stores keys and values for every token.",
        "let sum: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();",
        "Tokenization in Portuguese costs more tokens.",
        "推理引擎的内存带宽",
        "3.14159 * 2.71828 = 8.5397",
    ] {
        let ids = bpe.encode(s);
        let pieces: Vec<String> = ids
            .iter()
            .map(|&id| String::from_utf8_lossy(bpe.token_bytes(id)).into_owned())
            .collect();
        println!(
            "   {:>2} tokens for {:>2} bytes: {}",
            ids.len(),
            s.len(),
            pieces.join("|")
        );
    }
    println!();

    println!("== 4. encoding speed on the whole corpus (vocab 2048)");
    let start = Instant::now();
    let plain = black_box(bpe.encode(&text));
    let t_plain = start.elapsed();
    let mut cache = HashMap::new();
    let start = Instant::now();
    let cached = black_box(bpe.encode_cached(&text, &mut cache));
    let t_cold = start.elapsed();
    let start = Instant::now();
    let cached_again = black_box(bpe.encode_cached(&text, &mut cache));
    let t_warm = start.elapsed();
    assert!(plain == cached && cached == cached_again);
    let mb = text.len() as f64 / 1e6;
    println!(
        "   no cache:            {t_plain:>9.2?}  ({:.1} MB/s)",
        mb / t_plain.as_secs_f64()
    );
    println!(
        "   cache, first pass:   {t_cold:>9.2?}  ({:.1} MB/s, {} distinct chunks)",
        mb / t_cold.as_secs_f64(),
        cache.len()
    );
    println!(
        "   cache, second pass:  {t_warm:>9.2?}  ({:.1} MB/s)",
        mb / t_warm.as_secs_f64()
    );
    println!();

    println!("== 5. streaming decode, one token at a time");
    let sentence = "Café 👋🏽 déjà vu";
    let ids = bpe.encode(sentence);
    let mut dec = StreamDecoder::new();
    for id in &ids {
        let bytes = bpe.token_bytes(*id);
        let out = dec.push(bytes);
        println!("   token {id:>4} bytes {bytes:02x?} -> emits {out:?}");
    }
    println!("   end of stream -> {:?}", dec.finish());
}
