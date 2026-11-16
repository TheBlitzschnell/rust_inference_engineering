//! Which tokens may come next: the token mask.
//!
//! A token is allowed if the matcher accepts all of its bytes after the
//! text so far. Testing every token separately ([`allowed_naive`]) feeds
//! the matcher about 300,000 bytes per step for SmolLM2's 49,152 tokens.
//! Most tokens share their first bytes with others (" the", " then",
//! " there"...), so a trie of the vocabulary lets one walk test each shared
//! prefix once, and skip a whole subtree as soon as its prefix is
//! rejected ([`TokenTrie::walk`]). Inside a string the walk would visit
//! nearly the whole trie, so there [`TokenTrie::allowed`] uses a shortcut
//! computed once: tokens that are plain text are allowed if they fit.

use crate::json::Matcher;

/// The vocabulary's byte strings, as a tree: each edge a byte, each node
/// the tokens whose bytes end there. Also each token's class for the
/// shortcut inside strings.
pub struct TokenTrie {
    /// Node `i`'s children: `(byte, child)`, sorted by byte.
    children: Vec<Vec<(u8, u32)>>,
    /// The tokens whose bytes end at node `i`.
    tokens: Vec<Vec<u32>>,
    /// Tokens that are plain string content (whole UTF-8 characters, no
    /// quote, backslash or control character), with their lengths: inside
    /// a string, one is allowed exactly when it fits.
    plain: Vec<(u32, usize)>,
    /// Every other token: inside a string, each is tested on its own.
    other: Vec<(u32, Vec<u8>)>,
}

/// Whether `bytes` can sit inside a JSON string as they are.
fn plain_text(bytes: &[u8]) -> bool {
    std::str::from_utf8(bytes).is_ok()
        && bytes.iter().all(|&b| b >= 0x20 && b != b'"' && b != b'\\')
}

impl TokenTrie {
    /// Builds the trie from `(token id, bytes)` pairs.
    pub fn new<'a>(vocab: impl IntoIterator<Item = (u32, &'a [u8])>) -> Self {
        let mut trie = Self {
            children: vec![Vec::new()],
            tokens: vec![Vec::new()],
            plain: Vec::new(),
            other: Vec::new(),
        };
        for (id, bytes) in vocab {
            if plain_text(bytes) {
                trie.plain.push((id, bytes.len()));
            } else {
                trie.other.push((id, bytes.to_vec()));
            }
            let mut node = 0;
            for &b in bytes {
                node = match trie.children[node].binary_search_by_key(&b, |&(c, _)| c) {
                    Ok(i) => trie.children[node][i].1 as usize,
                    Err(i) => {
                        let child = trie.children.len();
                        trie.children.push(Vec::new());
                        trie.tokens.push(Vec::new());
                        trie.children[node].insert(i, (b, child as u32));
                        child
                    }
                };
            }
            trie.tokens[node].push(id);
        }
        trie
    }

    pub fn nodes(&self) -> usize {
        self.children.len()
    }

    /// Sets `mask[t]` for every token `t` whose bytes the matcher accepts
    /// next (and leaves the others as they are). Returns the number of
    /// matcher copies made, a measure of the work.
    ///
    /// Inside a string, almost every token is allowed and walking the trie
    /// would visit nearly all of it; there, plain tokens are allowed by
    /// their length alone, and only the others go through the matcher.
    pub fn allowed(&self, matcher: &Matcher<'_>, mask: &mut [bool]) -> usize {
        let Some(room) = matcher.string_room() else {
            return self.walk(matcher, mask);
        };
        for &(t, len) in &self.plain {
            if len <= room {
                mask[t as usize] = true;
            }
        }
        for (t, bytes) in &self.other {
            let mut m = *matcher;
            if m.push_all(bytes) {
                mask[*t as usize] = true;
            }
        }
        self.other.len()
    }

    /// Like [`TokenTrie::allowed`], always by walking the trie.
    pub fn walk(&self, matcher: &Matcher<'_>, mask: &mut [bool]) -> usize {
        let mut tried = 0;
        self.walk_from(0, matcher, mask, &mut tried);
        tried
    }

    fn walk_from(&self, node: usize, matcher: &Matcher<'_>, mask: &mut [bool], tried: &mut usize) {
        for &(b, child) in &self.children[node] {
            let mut m = *matcher;
            *tried += 1;
            if m.push(b) {
                for &t in &self.tokens[child as usize] {
                    mask[t as usize] = true;
                }
                self.walk_from(child as usize, &m, mask, tried);
            }
        }
    }
}

/// Sets `mask[t]` for every allowed token by testing each one on its own.
pub fn allowed_naive(vocab: &[(u32, &[u8])], matcher: &Matcher<'_>, mask: &mut [bool]) {
    for &(id, bytes) in vocab {
        let mut m = *matcher;
        if m.push_all(bytes) {
            mask[id as usize] = true;
        }
    }
}
