//! Reading `tokenizer.json`: Hugging Face's byte-level BPE, exactly.
//!
//! The algorithm is chapter 11's. What this file adds is the format and the
//! details that must match the model's training token for token: GPT-2's
//! byte-to-character table, its pre-tokenization pattern (including a
//! look-ahead the `regex` crate does not support), SmolLM2's digit
//! splitting, and special tokens.

use crate::Error;
use regex::Regex;
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

#[derive(Deserialize)]
struct TokenizerJson {
    added_tokens: Vec<AddedToken>,
    normalizer: Option<serde_json::Value>,
    pre_tokenizer: serde_json::Value,
    model: BpeJson,
}

#[derive(Deserialize)]
struct AddedToken {
    id: u32,
    content: String,
}

#[derive(Deserialize)]
struct BpeJson {
    #[serde(rename = "type")]
    kind: String,
    vocab: HashMap<String, u32>,
    merges: Vec<MergeJson>,
    #[serde(default)]
    ignore_merges: bool,
    #[serde(default)]
    byte_fallback: bool,
}

/// Merges are written `"Ġ t"` in older files and `["Ġ", "t"]` in newer ones.
#[derive(Deserialize)]
#[serde(untagged)]
enum MergeJson {
    Joined(String),
    Pair(String, String),
}

/// GPT-2's pre-tokenization pattern, minus one alternative: the original has
/// `\s+(?!\S)` ("whitespace not followed by a non-space") before the final
/// `\s+`. The `regex` crate has no look-ahead, so [`Tokenizer::split_words`]
/// reproduces its effect by hand.
const PATTERN: &str = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+";

/// A byte-level BPE tokenizer loaded from a Hugging Face `tokenizer.json`.
pub struct Tokenizer {
    /// Token id → the bytes it stands for.
    token_bytes: Vec<Vec<u8>>,
    /// Byte → id of the token for that single byte. `None` for bytes the
    /// vocabulary has no token for (SmolLM2 lacks 21 of them).
    byte_tokens: [Option<u32>; 256],
    /// Pair of token ids → (rank, id of the merged token).
    merges: HashMap<(u32, u32), (u32, u32)>,
    /// Bytes → token id, for `ignore_merges` and lookups.
    vocab: HashMap<Vec<u8>, u32>,
    /// Special tokens, matched literally before anything else.
    specials: Vec<(String, u32)>,
    split_digits: bool,
    ignore_merges: bool,
    pattern: Regex,
}

/// GPT-2's table that gives every byte a printable character, so a token's
/// bytes can be written as a JSON string: printable ASCII and most of
/// Latin-1 stand for themselves, the other 68 bytes (space, control
/// characters...) get the characters from U+0100 on. A space becomes `Ġ`
/// (U+0120), a newline `Ċ` (U+010A).
pub fn byte_to_char() -> [char; 256] {
    let mut table = ['\0'; 256];
    let mut next = 256;
    for b in 0..=255u8 {
        let printable = matches!(b, b'!'..=b'~' | 0xA1..=0xAC | 0xAE..=0xFF);
        table[b as usize] = if printable {
            char::from(b)
        } else {
            next += 1;
            char::from_u32(next - 1).expect("U+0100..U+0143 are valid characters")
        };
    }
    table
}

impl Tokenizer {
    pub fn from_file(path: &Path) -> Result<Self, Error> {
        Self::from_json(&crate::read_to_string(path)?)
    }

    pub fn from_json(json: &str) -> Result<Self, Error> {
        let bad = |what: String| Error::Tokenizer(what);
        let file: TokenizerJson =
            serde_json::from_str(json).map_err(|e| bad(format!("invalid JSON: {e}")))?;
        if file.model.kind != "BPE" || file.model.byte_fallback {
            return Err(bad(format!(
                "model type {:?} is not byte-level BPE",
                file.model.kind
            )));
        }
        if file.normalizer.as_ref().is_some_and(|n| !n.is_null()) {
            return Err(bad("normalizers are not supported".into()));
        }
        let split_digits = check_pre_tokenizer(&file.pre_tokenizer).map_err(bad)?;

        // Vocabulary: decode each token's characters back into bytes.
        let char_to_byte: HashMap<char, u8> = byte_to_char()
            .iter()
            .enumerate()
            .map(|(b, &c)| (c, b as u8))
            .collect();
        let size = file
            .model
            .vocab
            .values()
            .max()
            .map_or(0, |&m| m as usize + 1);
        let mut token_bytes = vec![Vec::new(); size];
        for (text, &id) in &file.model.vocab {
            token_bytes[id as usize] = text
                .chars()
                .map(|c| char_to_byte.get(&c).copied())
                .collect::<Option<Vec<u8>>>()
                .ok_or_else(|| bad(format!("token {text:?} is not byte-level")))?;
        }
        // Special tokens stand for their literal text.
        for t in &file.added_tokens {
            if t.id as usize >= token_bytes.len() {
                token_bytes.resize(t.id as usize + 1, Vec::new());
            }
            token_bytes[t.id as usize] = t.content.as_bytes().to_vec();
        }
        let vocab: HashMap<Vec<u8>, u32> = token_bytes
            .iter()
            .enumerate()
            .map(|(id, bytes)| (bytes.clone(), id as u32))
            .collect();

        let mut byte_tokens = [None; 256];
        for (b, slot) in byte_tokens.iter_mut().enumerate() {
            *slot = vocab.get(&vec![b as u8]).copied();
        }

        // Merges: rank = position in the list.
        let id_of = |s: &str| {
            file.model
                .vocab
                .get(s)
                .copied()
                .ok_or_else(|| bad(format!("merge uses unknown token {s:?}")))
        };
        let mut merges = HashMap::with_capacity(file.model.merges.len());
        for (rank, merge) in file.model.merges.iter().enumerate() {
            let (a, b) = match merge {
                MergeJson::Joined(s) => s
                    .split_once(' ')
                    .ok_or_else(|| bad(format!("bad merge {s:?}")))?,
                MergeJson::Pair(a, b) => (a.as_str(), b.as_str()),
            };
            let merged = id_of(&format!("{a}{b}"))?;
            merges.insert((id_of(a)?, id_of(b)?), (rank as u32, merged));
        }

        Ok(Self {
            token_bytes,
            byte_tokens,
            merges,
            vocab,
            specials: file
                .added_tokens
                .iter()
                .map(|t| (t.content.clone(), t.id))
                .collect(),
            split_digits,
            ignore_merges: file.model.ignore_merges,
            pattern: Regex::new(PATTERN).expect("the pattern is valid"),
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.token_bytes.len()
    }

    /// The bytes a token stands for (possibly part of a UTF-8 character).
    pub fn token_bytes(&self, id: u32) -> &[u8] {
        &self.token_bytes[id as usize]
    }

    /// Bytes that have no token of their own. Hugging Face's tokenizer
    /// silently drops them when encoding (with no `unk_token` to fall back
    /// on), and so does this one.
    pub fn missing_bytes(&self) -> Vec<u8> {
        (0..=255u8)
            .filter(|&b| self.byte_tokens[b as usize].is_none())
            .collect()
    }

    /// The id of the token for exactly this text, if there is one.
    pub fn token_id(&self, text: &str) -> Option<u32> {
        self.vocab.get(text.as_bytes()).copied()
    }

    /// Text → token ids. Special tokens in the text become their ids.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        let mut rest = text;
        while !rest.is_empty() {
            // The earliest special token; the longest if two start together.
            let next = self
                .specials
                .iter()
                .filter_map(|(s, id)| rest.find(s.as_str()).map(|at| (at, s.len(), *id)))
                .min_by_key(|&(at, len, _)| (at, std::cmp::Reverse(len)));
            let plain_end = next.map_or(rest.len(), |(at, _, _)| at);
            self.encode_plain(&rest[..plain_end], &mut out);
            match next {
                Some((at, len, id)) => {
                    out.push(id);
                    rest = &rest[at + len..];
                }
                None => rest = "",
            }
        }
        out
    }

    /// Token ids → text. Invalid UTF-8 (a character cut in half) becomes
    /// U+FFFD; use chapter 11's `StreamDecoder` when streaming.
    pub fn decode(&self, ids: &[u32]) -> String {
        let bytes: Vec<u8> = ids
            .iter()
            .flat_map(|&id| self.token_bytes(id).iter().copied())
            .collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }

    fn encode_plain(&self, text: &str, out: &mut Vec<u32>) {
        let mut words = Vec::new();
        if self.split_digits {
            // Every digit is a piece of its own; the pattern then runs on
            // the text between digits.
            let mut start = 0;
            for (i, c) in text.char_indices() {
                if c.is_numeric() {
                    self.split_words(&text[start..i], &mut words);
                    words.push(&text[i..i + c.len_utf8()]);
                    start = i + c.len_utf8();
                }
            }
            self.split_words(&text[start..], &mut words);
        } else {
            self.split_words(text, &mut words);
        }
        for word in words {
            self.encode_word(word, out);
        }
    }

    /// Splits text with GPT-2's pattern. Where the original pattern's
    /// `\s+(?!\S)` would give a whitespace run followed by a word back its
    /// last character, so that `"a   b"` becomes `"a"`, `"  "`, `" b"`, we
    /// shorten the run by one character ourselves.
    fn split_words<'t>(&self, text: &'t str, words: &mut Vec<&'t str>) {
        let mut pos = 0;
        while pos < text.len() {
            let m = self
                .pattern
                .find_at(text, pos)
                .expect("every character matches some alternative");
            let mut end = m.end();
            let run = &text[pos..end];
            if end < text.len()
                && run.chars().all(char::is_whitespace)
                && let Some((last, _)) = run.char_indices().last()
                && last > 0
            {
                end = pos + last;
            }
            words.push(&text[pos..end]);
            pos = end;
        }
    }

    /// Chapter 11's merge loop: repeatedly apply the lowest-ranked merge
    /// among adjacent pairs.
    fn encode_word(&self, word: &str, out: &mut Vec<u32>) {
        if self.ignore_merges
            && let Some(&id) = self.vocab.get(word.as_bytes())
        {
            out.push(id);
            return;
        }
        // Bytes without a token are dropped, as the reference does.
        let mut ids: Vec<u32> = word
            .bytes()
            .filter_map(|b| self.byte_tokens[b as usize])
            .collect();
        while ids.len() >= 2 {
            let best = ids
                .windows(2)
                .filter_map(|p| {
                    self.merges
                        .get(&(p[0], p[1]))
                        .map(|&(rank, new)| (rank, p[0], p[1], new))
                })
                .min();
            let Some((_, a, b, new)) = best else { break };
            // Replace every (a, b), left to right.
            let mut merged = Vec::with_capacity(ids.len());
            let mut i = 0;
            while i < ids.len() {
                if i + 1 < ids.len() && ids[i] == a && ids[i + 1] == b {
                    merged.push(new);
                    i += 2;
                } else {
                    merged.push(ids[i]);
                    i += 1;
                }
            }
            ids = merged;
        }
        out.extend_from_slice(&ids);
    }
}

/// Accepts `ByteLevel`, optionally preceded by `Digits` with
/// `individual_digits`, as SmolLM2 and GPT-2 style models use. Returns
/// whether digits are split.
fn check_pre_tokenizer(p: &serde_json::Value) -> Result<bool, String> {
    let steps: Vec<&serde_json::Value> = match p["type"].as_str() {
        Some("Sequence") => p["pretokenizers"]
            .as_array()
            .map_or_else(Vec::new, |a| a.iter().collect()),
        _ => vec![p],
    };
    let mut split_digits = false;
    let mut byte_level = false;
    for step in steps {
        match step["type"].as_str() {
            Some("Digits") if step["individual_digits"] == true => split_digits = true,
            Some("ByteLevel") if step["add_prefix_space"] == false && step["use_regex"] == true => {
                byte_level = true;
            }
            _ => return Err(format!("unsupported pre-tokenizer step {step}")),
        }
    }
    if byte_level {
        Ok(split_digits)
    } else {
        Err("no ByteLevel pre-tokenizer".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_byte_table_matches_gpt2() {
        let t = byte_to_char();
        assert_eq!(t[b' ' as usize], 'Ġ');
        assert_eq!(t[b'\n' as usize], 'Ċ');
        assert_eq!(t[b'a' as usize], 'a');
        assert_eq!(t[0], 'Ā'); // U+0100
        let mut all: Vec<char> = t.to_vec();
        all.sort_unstable();
        all.dedup();
        assert_eq!(all.len(), 256, "every byte gets its own character");
    }
}
