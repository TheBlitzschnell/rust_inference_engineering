//! Chapter 11: turning text into token IDs and back.
//!
//! A language model never sees characters. It sees a sequence of integers,
//! each naming an entry in a fixed vocabulary of text fragments. This crate
//! implements byte-level Byte Pair Encoding (BPE), the scheme used by GPT-2,
//! Llama 3, SmolLM2 and most current models:
//!
//! - start from the 256 possible bytes, so any text can be encoded;
//! - learn "merges" from a corpus: repeatedly glue the most frequent pair of
//!   adjacent tokens into a new token;
//! - to encode, split text into chunks (roughly words) and apply the learned
//!   merges in the order they were learned.
//!
//! It also implements [`StreamDecoder`], which turns a stream of tokens back
//! into text without ever emitting half of a UTF-8 character: exactly the
//! problem a streaming server has to solve.

use std::collections::HashMap;

/// A token ID.
pub type Token = u32;

/// A trained byte-level BPE tokenizer.
#[derive(Debug, Clone)]
pub struct Bpe {
    /// `merges[r] = (a, b)`: the r-th learned merge glues token `a` and
    /// token `b` into token `256 + r`.
    merges: Vec<(Token, Token)>,
    /// Pair → rank (position in `merges`). Lower rank = learned earlier =
    /// applied first.
    ranks: HashMap<(Token, Token), u32>,
    /// The bytes each token stands for.
    vocab: Vec<Vec<u8>>,
    /// Special tokens like `<|endoftext|>`: matched literally, never merged.
    specials: Vec<(String, Token)>,
}

/// The character classes pre-tokenization cares about.
#[derive(PartialEq, Clone, Copy)]
enum Kind {
    Letter,
    Digit,
    Space,
    Other,
}

fn kind(c: char) -> Kind {
    if c.is_alphabetic() {
        Kind::Letter
    } else if c.is_numeric() {
        Kind::Digit
    } else if c.is_whitespace() {
        Kind::Space
    } else {
        Kind::Other
    }
}

/// Splits text into chunks that merges may not cross: a run of letters, of
/// digits, of other symbols, or of whitespace, where a single space directly
/// in front of a non-space run belongs to that run (" the", " 42").
///
/// This is a simplified version of the regular expression GPT-2 uses (which
/// also special-cases contractions like "'s" and hands the last space of a
/// whitespace run to the following word). It keeps " the" as one unit and
/// stops merges from gluing the end of one word to the start of the next.
pub fn pre_tokenize(text: &str) -> Vec<&str> {
    let mut chunks = Vec::new();
    let mut chars = text.char_indices().peekable();
    while let Some((start, c)) = chars.next() {
        let mut end = start + c.len_utf8();
        let mut chunk_kind = kind(c);
        // A single space in front of a word, number or symbol joins it.
        if c == ' '
            && let Some(&(j, next)) = chars.peek()
            && kind(next) != Kind::Space
        {
            chars.next();
            end = j + next.len_utf8();
            chunk_kind = kind(next);
        }
        // Extend the chunk while the characters keep the same kind.
        while let Some(&(j, next)) = chars.peek() {
            if kind(next) != chunk_kind {
                break;
            }
            chars.next();
            end = j + next.len_utf8();
        }
        chunks.push(&text[start..end]);
    }
    chunks
}

/// Replaces every occurrence of the pair `(a, b)` in `ids` with `new`.
fn merge_pair(ids: &mut Vec<Token>, a: Token, b: Token, new: Token) {
    let mut out = 0;
    let mut i = 0;
    while i < ids.len() {
        if i + 1 < ids.len() && ids[i] == a && ids[i + 1] == b {
            ids[out] = new;
            i += 2;
        } else {
            ids[out] = ids[i];
            i += 1;
        }
        out += 1;
    }
    ids.truncate(out);
}

impl Bpe {
    /// A tokenizer with no merges: one token per byte.
    pub fn bytes_only() -> Self {
        Self {
            merges: Vec::new(),
            ranks: HashMap::new(),
            vocab: (0..=255u8).map(|b| vec![b]).collect(),
            specials: Vec::new(),
        }
    }

    /// Learns `vocab_size - 256` merges from `corpus`.
    ///
    /// Each round counts every adjacent pair of tokens in the corpus (words
    /// are counted once and weighted by how often they occur), merges the
    /// most frequent pair everywhere, and records the merge. Ties go to the
    /// numerically smallest pair, so training is deterministic.
    pub fn train(corpus: &str, vocab_size: usize) -> Self {
        assert!(vocab_size >= 256, "the 256 byte tokens are always included");
        let mut tok = Self::bytes_only();

        // Count each distinct chunk once; this shrinks the work enormously.
        let mut words: HashMap<Vec<Token>, u64> = HashMap::new();
        for chunk in pre_tokenize(corpus) {
            let ids: Vec<Token> = chunk.bytes().map(Token::from).collect();
            *words.entry(ids).or_default() += 1;
        }
        let mut words: Vec<(Vec<Token>, u64)> = words.into_iter().collect();

        while tok.vocab.len() < vocab_size {
            let mut counts: HashMap<(Token, Token), u64> = HashMap::new();
            for (ids, n) in &words {
                for pair in ids.windows(2) {
                    *counts.entry((pair[0], pair[1])).or_default() += n;
                }
            }
            // Most frequent pair; ties broken by the smallest pair.
            let Some((&(a, b), _)) = counts
                .iter()
                .max_by(|x, y| x.1.cmp(y.1).then_with(|| y.0.cmp(x.0)))
            else {
                break; // nothing left to merge
            };
            let new = tok.vocab.len() as Token;
            for (ids, _) in &mut words {
                merge_pair(ids, a, b, new);
            }
            tok.add_merge(a, b);
        }
        tok
    }

    fn add_merge(&mut self, a: Token, b: Token) {
        let new_bytes = [self.vocab[a as usize].as_slice(), &self.vocab[b as usize]].concat();
        self.ranks.insert((a, b), self.merges.len() as u32);
        self.merges.push((a, b));
        self.vocab.push(new_bytes);
    }

    /// Registers a special token (e.g. `<|endoftext|>`) with the next free ID.
    pub fn add_special(&mut self, text: &str) -> Token {
        let id = self.vocab.len() as Token;
        self.vocab.push(text.as_bytes().to_vec());
        self.specials.push((text.to_owned(), id));
        id
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    pub fn merges(&self) -> &[(Token, Token)] {
        &self.merges
    }

    /// The bytes a token stands for.
    pub fn token_bytes(&self, id: Token) -> &[u8] {
        &self.vocab[id as usize]
    }

    /// Encodes one pre-tokenized chunk by applying merges in rank order.
    ///
    /// Repeatedly find the adjacent pair with the lowest rank (the merge
    /// learned earliest) and merge it, until no adjacent pair has a merge.
    pub fn encode_chunk(&self, chunk: &str, out: &mut Vec<Token>) {
        let mut ids: Vec<Token> = chunk.bytes().map(Token::from).collect();
        while ids.len() >= 2 {
            let best = ids
                .windows(2)
                .filter_map(|p| self.ranks.get(&(p[0], p[1])).map(|&r| (r, p[0], p[1])))
                .min();
            let Some((rank, a, b)) = best else { break };
            merge_pair(&mut ids, a, b, 256 + rank);
        }
        out.extend_from_slice(&ids);
    }

    /// Encodes text into tokens. Special tokens are matched first and are
    /// never split or merged; everything between them goes through
    /// pre-tokenization and BPE.
    pub fn encode(&self, text: &str) -> Vec<Token> {
        let mut out = Vec::new();
        let mut rest = text;
        while !rest.is_empty() {
            // The earliest special token in the remaining text, if any.
            let next_special = self
                .specials
                .iter()
                .filter_map(|(s, id)| rest.find(s.as_str()).map(|pos| (pos, s.len(), *id)))
                .min_by_key(|&(pos, len, _)| (pos, std::cmp::Reverse(len)));
            let (plain, special) = match next_special {
                Some((pos, len, id)) => (&rest[..pos], Some((len, id))),
                None => (rest, None),
            };
            for chunk in pre_tokenize(plain) {
                self.encode_chunk(chunk, &mut out);
            }
            match special {
                Some((len, id)) => {
                    out.push(id);
                    rest = &rest[plain.len() + len..];
                }
                None => rest = "",
            }
        }
        out
    }

    /// Encodes with a cache of already-encoded chunks. Text has many
    /// repeated words, so most chunks are found in the cache and the merge
    /// loop is skipped entirely.
    pub fn encode_cached(&self, text: &str, cache: &mut HashMap<String, Vec<Token>>) -> Vec<Token> {
        assert!(
            self.specials.is_empty(),
            "the cached path does not handle specials"
        );
        let mut out = Vec::new();
        for chunk in pre_tokenize(text) {
            if let Some(ids) = cache.get(chunk) {
                out.extend_from_slice(ids);
            } else {
                let mut ids = Vec::new();
                self.encode_chunk(chunk, &mut ids);
                out.extend_from_slice(&ids);
                cache.insert(chunk.to_owned(), ids);
            }
        }
        out
    }

    /// The bytes of a token sequence, concatenated. Not necessarily valid
    /// UTF-8: a token may end in the middle of a character.
    pub fn decode_bytes(&self, ids: &[Token]) -> Vec<u8> {
        ids.iter()
            .flat_map(|&id| self.token_bytes(id).iter().copied())
            .collect()
    }

    /// Decodes to a string, replacing invalid UTF-8 with U+FFFD.
    pub fn decode(&self, ids: &[Token]) -> String {
        String::from_utf8_lossy(&self.decode_bytes(ids)).into_owned()
    }
}

/// Turns a stream of token bytes into text, one token at a time, without
/// ever splitting a UTF-8 character.
///
/// A character like "é" is 2 bytes and "👋" is 4. A byte-level tokenizer can
/// put the first bytes of a character at the end of one token and the rest
/// in the next. A server that streams text to a user must hold those bytes
/// back until the character is complete.
#[derive(Debug, Default)]
pub struct StreamDecoder {
    pending: Vec<u8>,
}

impl StreamDecoder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds one token's bytes and returns whatever text is now complete
    /// (possibly empty).
    pub fn push(&mut self, bytes: &[u8]) -> String {
        self.pending.extend_from_slice(bytes);
        let mut out = String::new();
        loop {
            match std::str::from_utf8(&self.pending) {
                Ok(s) => {
                    out.push_str(s);
                    self.pending.clear();
                    return out;
                }
                Err(e) => {
                    let valid = e.valid_up_to();
                    // The prefix up to `valid` is complete, valid UTF-8.
                    out.push_str(
                        std::str::from_utf8(&self.pending[..valid])
                            .expect("checked by valid_up_to"),
                    );
                    match e.error_len() {
                        // The remaining bytes are the *start* of a character
                        // that has not finished arriving: keep them.
                        None => {
                            self.pending.drain(..valid);
                            return out;
                        }
                        // The bytes are invalid and can never become valid:
                        // replace them and keep going.
                        Some(bad) => {
                            out.push(char::REPLACEMENT_CHARACTER);
                            self.pending.drain(..valid + bad);
                        }
                    }
                }
            }
        }
    }

    /// Flushes whatever is left at the end of the stream. Incomplete
    /// characters become U+FFFD.
    pub fn finish(self) -> String {
        String::from_utf8_lossy(&self.pending).into_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CORPUS: &str = "the cat sat on the mat. the cat ate the rat. \
                          then the cat sat on the hat, and the rat ran.";

    #[test]
    fn pre_tokenize_keeps_leading_spaces_with_words() {
        assert_eq!(
            pre_tokenize("the cat, 42!"),
            ["the", " cat", ",", " 42", "!"]
        );
        assert_eq!(pre_tokenize("héllo wörld"), ["héllo", " wörld"]);
        assert_eq!(pre_tokenize("a\n\nb"), ["a", "\n\n", "b"]);
        assert_eq!(pre_tokenize(""), Vec::<&str>::new());
        // Chunks always cover the input exactly.
        let s = "  x  y\t\tz 12ab!?";
        assert_eq!(pre_tokenize(s).concat(), s);
    }

    #[test]
    fn training_learns_frequent_pairs_first() {
        let bpe = Bpe::train(CORPUS, 256 + 10);
        assert_eq!(bpe.vocab_size(), 266);
        // " the" and "at" are the most repeated fragments in the corpus.
        let learned: Vec<String> = (256..266)
            .map(|id| String::from_utf8_lossy(bpe.token_bytes(id)).into_owned())
            .collect();
        assert!(learned.contains(&"at".to_string()), "{learned:?}");
        assert!(learned.iter().any(|t| t.contains("th")), "{learned:?}");
        // Training is deterministic.
        assert_eq!(Bpe::train(CORPUS, 266).merges(), bpe.merges());
    }

    #[test]
    fn encode_decode_round_trips_any_text() {
        let bpe = Bpe::train(CORPUS, 300);
        for text in [
            "the cat sat",
            "",
            "héllo wörld",
            "日本語のテキスト",
            "emoji 👋🏽 and flags 🇫🇷",
            "tabs\tand\nnewlines\r\n",
            "a\u{0301} combining mark",
        ] {
            let ids = bpe.encode(text);
            assert_eq!(bpe.decode(&ids), text);
        }
    }

    #[test]
    fn merges_shorten_the_encoding() {
        let bytes = Bpe::bytes_only();
        let trained = Bpe::train(CORPUS, 320);
        let text = "the cat sat on the mat";
        assert_eq!(bytes.encode(text).len(), text.len());
        assert!(trained.encode(text).len() < text.len() / 2);
    }

    #[test]
    fn cached_encoding_gives_the_same_tokens() {
        let bpe = Bpe::train(CORPUS, 300);
        let mut cache = HashMap::new();
        let text = "the cat sat on the mat and the cat sat again";
        assert_eq!(bpe.encode_cached(text, &mut cache), bpe.encode(text));
        assert!(cache.contains_key(" cat"));
    }

    #[test]
    fn special_tokens_are_atomic() {
        let mut bpe = Bpe::train(CORPUS, 300);
        let eot = bpe.add_special("<|endoftext|>");
        let ids = bpe.encode("the cat<|endoftext|>the mat");
        assert_eq!(ids.iter().filter(|&&t| t == eot).count(), 1);
        assert_eq!(bpe.decode(&ids), "the cat<|endoftext|>the mat");
        // A special token's text inside a normal word is still recognized.
        assert!(bpe.encode("x<|endoftext|>").contains(&eot));
    }

    #[test]
    fn stream_decoder_never_splits_a_character() {
        let text = "héllo 👋🏽 wörld";
        let mut dec = StreamDecoder::new();
        let mut pieces = Vec::new();
        // Worst case: one byte per token.
        for b in text.as_bytes() {
            let piece = dec.push(&[*b]);
            // Every emitted piece is complete, valid text on its own.
            assert!(!piece.contains(char::REPLACEMENT_CHARACTER));
            pieces.push(piece);
        }
        assert_eq!(pieces.concat() + &dec.finish(), text);
        // The 4 bytes of 👋 produce nothing, nothing, nothing, then "👋".
        let wave_start = text.find('👋').unwrap();
        assert_eq!(pieces[wave_start], "");
        assert_eq!(pieces[wave_start + 3], "👋");
    }

    #[test]
    fn stream_decoder_replaces_invalid_bytes() {
        let mut dec = StreamDecoder::new();
        assert_eq!(dec.push(&[b'a', 0xFF, b'b']), "a\u{FFFD}b");
        assert_eq!(dec.push(&[0xE2, 0x82]), ""); // first 2 bytes of "€"
        assert_eq!(dec.finish(), "\u{FFFD}"); // stream ended mid-character
    }
}
