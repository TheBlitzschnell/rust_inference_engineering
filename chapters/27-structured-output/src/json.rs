//! A JSON matcher: reads bytes one at a time and says whether what it has
//! read so far can still be the beginning of a JSON value that follows a
//! schema.
//!
//! It is a pushdown automaton: a stack of frames, one per value being read
//! (an object inside an array inside an object...), each frame a small
//! state machine. The whole matcher is `Copy`: a fixed array of frames
//! that hold references to the schema, so trying a byte on a copy costs a
//! memory copy, not an allocation. Masking tokens (chapter's
//! [`crate::mask`]) tries thousands of byte sequences per step.

/// What the value must look like: a small subset of JSON Schema.
#[derive(Clone, Debug, PartialEq)]
pub enum Schema {
    /// Any JSON value.
    Any,
    /// An object with exactly these properties, in this order.
    Object(Vec<(String, Schema)>),
    /// An array of values of one schema.
    Array(Box<Schema>),
    /// A string of at most `max_len` bytes (between the quotes, escapes
    /// counted as written).
    String {
        max_len: usize,
    },
    Integer,
    Number,
    Boolean,
    Null,
    /// One of these strings (plain text: no quotes, backslashes or
    /// control characters).
    Enum(Vec<String>),
}

impl Schema {
    /// Panics if the schema uses something the matcher cannot follow.
    fn check(&self) {
        match self {
            Self::Object(props) => {
                for (name, schema) in props {
                    assert!(plain(name), "key {name:?} needs escaping");
                    schema.check();
                }
            }
            Self::Array(item) => item.check(),
            Self::Enum(options) => {
                assert!(
                    !options.is_empty() && options.len() <= 64,
                    "an enum needs 1 to 64 options"
                );
                assert!(
                    options.iter().all(|o| plain(o)),
                    "enum options need escaping"
                );
            }
            _ => {}
        }
    }
}

/// Text that can appear between JSON quotes as it is.
fn plain(text: &str) -> bool {
    text.bytes().all(|b| b >= 0x20 && b != b'"' && b != b'\\')
}

/// The deepest nesting the matcher follows. Deeper input is rejected.
pub const MAX_DEPTH: usize = 24;
/// Whitespace allowed in a row by default (enough for pretty-printed JSON
/// a few levels deep). [`Matcher::with_max_whitespace`] changes it.
pub const MAX_WHITESPACE: u8 = 12;
/// Digits allowed in the integer and in the fraction part of a number.
pub const MAX_DIGITS: u8 = 15;
/// Digits allowed in an exponent: with 2, every number fits an `f64`
/// (`1e999` is valid JSON, and most parsers reject it).
pub const MAX_EXPONENT_DIGITS: u8 = 2;

/// What the next byte inside a string must be.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Expect {
    /// The start of a character, a backslash, or the closing quote.
    Char,
    /// The character after a backslash.
    Escape,
    /// `left` more hex digits of a `\u` escape (`d`: the first was `d`,
    /// so the second must not make it a UTF-16 surrogate, D800-DFFF).
    Hex { left: u8, d: bool },
    /// `left` more continuation bytes of a UTF-8 character, the next one
    /// in `lo..=hi` (UTF-8 forbids some combinations, such as overlong
    /// encodings and surrogates, through the second byte's range).
    Utf8 { left: u8, lo: u8, hi: u8 },
}

/// Where an object or array is, between its brackets.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stage {
    /// After `{` or `[`.
    Open,
    /// After a key (objects only): `:` comes next.
    Key,
    /// After a value: `,` or the closing bracket comes next.
    Value,
    /// After `,`.
    Comma,
}

/// Where a number is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Num {
    Minus,
    Zero,
    Int,
    Dot,
    Frac,
    Exp,
    ExpSign,
    ExpDigits,
}

impl Num {
    /// Whether a number may end here.
    fn complete(self) -> bool {
        matches!(self, Self::Zero | Self::Int | Self::Frac | Self::ExpDigits)
    }
}

#[derive(Clone, Copy, Debug)]
enum Frame<'s> {
    /// A value of this schema comes next (after optional whitespace).
    Value(&'s Schema),
    /// Inside an object with fixed properties; `next` is the index of the
    /// property whose key comes next (or whose value is being read).
    Object {
        props: &'s [(String, Schema)],
        next: usize,
        stage: Stage,
    },
    /// Inside an object with any keys (for `Schema::Any`).
    AnyObject {
        stage: Stage,
    },
    Array {
        item: &'s Schema,
        stage: Stage,
    },
    /// Inside a string: bytes so far, the most allowed, and what the next
    /// byte must be.
    Str {
        len: usize,
        max: usize,
        expect: Expect,
    },
    /// Inside a string that must be one of `options` followed by `"`: the
    /// options still possible (a bit mask) and the bytes matched so far.
    Fixed {
        options: &'s [String],
        alive: u64,
        pos: usize,
    },
    Number {
        state: Num,
        integer: bool,
        digits: u8,
    },
    /// The rest of `true`, `false` or `null`.
    Literal {
        text: &'static [u8],
        pos: usize,
    },
}

/// Reads JSON a byte at a time against a schema.
#[derive(Clone, Copy, Debug)]
pub struct Matcher<'s> {
    stack: [Option<Frame<'s>>; MAX_DEPTH],
    depth: usize,
    /// A fixed key being matched (its text and the bytes matched so far),
    /// on top of the stack's object frame.
    key: Option<(&'s str, usize)>,
    whitespace: u8,
    max_whitespace: u8,
    /// The root value has been read completely.
    done: bool,
}

/// Whether `b` is JSON whitespace.
fn is_ws(b: u8) -> bool {
    matches!(b, b' ' | b'\n' | b'\t' | b'\r')
}

/// What a frame did with a byte.
enum Step<'s> {
    /// Consumed; the frame stays (possibly changed).
    Stay(Frame<'s>),
    /// Consumed; the frame is finished.
    Pop,
    /// Consumed; the frame stays (changed) and a child frame starts on top.
    Push(Frame<'s>, Frame<'s>),
    /// Consumed; the frame is replaced by another.
    Replace(Frame<'s>),
    /// Not consumed: the frame is finished and the byte belongs to the
    /// frame below (a number ends at the first byte that is not part of it).
    PopAndRetry,
    /// Consumed as whitespace.
    Whitespace,
    Reject,
}

impl<'s> Matcher<'s> {
    pub fn new(schema: &'s Schema) -> Self {
        Self::with_max_whitespace(schema, MAX_WHITESPACE)
    }

    /// A matcher that allows at most `max` whitespace bytes in a row; 0
    /// allows only compact JSON, with no whitespace outside strings.
    pub fn with_max_whitespace(schema: &'s Schema, max: u8) -> Self {
        schema.check();
        let mut stack = [None; MAX_DEPTH];
        stack[0] = Some(Frame::Value(schema));
        Self {
            stack,
            depth: 1,
            key: None,
            whitespace: 0,
            max_whitespace: max,
            done: false,
        }
    }

    /// Whether the input so far is a complete value (more whitespace may
    /// follow, nothing else).
    pub fn is_complete(&self) -> bool {
        self.done
            || (self.depth == 1
                && matches!(self.stack[0], Some(Frame::Number { state, .. }) if state.complete()))
    }

    /// Whether the value is complete and closed: nothing but whitespace
    /// can follow. (A complete number at the root could still grow.)
    pub fn is_finished(&self) -> bool {
        self.done
    }

    /// How many more bytes of plain text the string being read can take,
    /// if the matcher is in a string between characters (not inside an
    /// escape, a multi-byte character, a fixed key or an enum value).
    /// [`crate::mask`] uses it to skip the matcher for most tokens.
    pub fn string_room(&self) -> Option<usize> {
        if self.key.is_some() || self.done || self.depth == 0 {
            return None;
        }
        match self.stack[self.depth - 1] {
            Some(Frame::Str {
                len,
                max,
                expect: Expect::Char,
            }) => Some(max - len),
            _ => None,
        }
    }

    /// Feeds every byte of `bytes`; false (leaving the matcher in an
    /// unspecified state) if one of them cannot continue the input.
    pub fn push_all(&mut self, bytes: &[u8]) -> bool {
        bytes.iter().all(|&b| self.push(b))
    }

    /// Feeds one byte; false if it cannot continue the input. After false
    /// the matcher's state is unspecified: try bytes on a copy.
    pub fn push(&mut self, b: u8) -> bool {
        if let Some((name, pos)) = self.key {
            return self.push_key(name, pos, b);
        }
        loop {
            if self.done || self.depth == 0 {
                return is_ws(b) && self.count_whitespace();
            }
            let top = self.stack[self.depth - 1].expect("frames below depth exist");
            let step = self.step(top, b);
            if !matches!(step, Step::Whitespace) {
                self.whitespace = 0;
            }
            match step {
                Step::Stay(f) | Step::Replace(f) => self.stack[self.depth - 1] = Some(f),
                Step::Push(f, child) => {
                    if self.depth == MAX_DEPTH {
                        return false;
                    }
                    self.stack[self.depth - 1] = Some(f);
                    self.stack[self.depth] = Some(child);
                    self.depth += 1;
                }
                Step::Pop => self.pop(),
                Step::PopAndRetry => {
                    self.pop();
                    continue;
                }
                Step::Whitespace => return self.count_whitespace(),
                Step::Reject => return false,
            }
            return true;
        }
    }

    fn count_whitespace(&mut self) -> bool {
        self.whitespace += 1;
        self.whitespace <= self.max_whitespace
    }

    fn pop(&mut self) {
        self.depth -= 1;
        self.stack[self.depth] = None;
        if self.depth == 0 {
            self.done = true;
        }
    }

    /// The next byte of a fixed key: the key's text, then `"`.
    fn push_key(&mut self, name: &'s str, pos: usize, b: u8) -> bool {
        let text = name.as_bytes();
        let want = if pos < text.len() { text[pos] } else { b'"' };
        if b != want {
            return false;
        }
        self.whitespace = 0;
        self.key = if pos < text.len() {
            Some((name, pos + 1))
        } else {
            None // the closing quote: the object frame is at its `Key` stage
        };
        true
    }

    fn step(&mut self, frame: Frame<'s>, b: u8) -> Step<'s> {
        match frame {
            Frame::Value(schema) => start_value(schema, b),
            Frame::Object { props, next, stage } => {
                if is_ws(b) {
                    return Step::Whitespace;
                }
                match (stage, b) {
                    (Stage::Open | Stage::Comma, b'"') if next < props.len() => {
                        self.key = Some((&props[next].0, 0));
                        Step::Stay(Frame::Object {
                            props,
                            next,
                            stage: Stage::Key,
                        })
                    }
                    (Stage::Open, b'}') if props.is_empty() => Step::Pop,
                    (Stage::Key, b':') => Step::Push(
                        Frame::Object {
                            props,
                            next: next + 1,
                            stage: Stage::Value,
                        },
                        Frame::Value(&props[next].1),
                    ),
                    (Stage::Value, b',') if next < props.len() => Step::Stay(Frame::Object {
                        props,
                        next,
                        stage: Stage::Comma,
                    }),
                    (Stage::Value, b'}') if next == props.len() => Step::Pop,
                    _ => Step::Reject,
                }
            }
            Frame::AnyObject { stage } => {
                if is_ws(b) {
                    return Step::Whitespace;
                }
                match (stage, b) {
                    (Stage::Open | Stage::Comma, b'"') => Step::Push(
                        Frame::AnyObject { stage: Stage::Key },
                        Frame::Str {
                            len: 0,
                            max: usize::MAX,
                            expect: Expect::Char,
                        },
                    ),
                    (Stage::Open | Stage::Value, b'}') => Step::Pop,
                    (Stage::Key, b':') => Step::Push(
                        Frame::AnyObject {
                            stage: Stage::Value,
                        },
                        Frame::Value(&Schema::Any),
                    ),
                    (Stage::Value, b',') => Step::Stay(Frame::AnyObject {
                        stage: Stage::Comma,
                    }),
                    _ => Step::Reject,
                }
            }
            Frame::Array { item, stage } => {
                if is_ws(b) {
                    return Step::Whitespace;
                }
                match (stage, b) {
                    (Stage::Open | Stage::Value, b']') => Step::Pop,
                    (Stage::Value, b',') => Step::Stay(Frame::Array {
                        item,
                        stage: Stage::Comma,
                    }),
                    (Stage::Open | Stage::Comma, _) => {
                        // The byte starts the item: hand it to a new value
                        // frame.
                        match start_value(item, b) {
                            Step::Replace(child) => Step::Push(
                                Frame::Array {
                                    item,
                                    stage: Stage::Value,
                                },
                                child,
                            ),
                            _ => Step::Reject,
                        }
                    }
                    _ => Step::Reject,
                }
            }
            Frame::Str { len, max, expect } => string_byte(len, max, expect, b),
            Frame::Fixed {
                options,
                alive,
                pos,
            } => {
                let mut still = 0u64;
                let mut finished = false;
                for (i, option) in options.iter().enumerate() {
                    if alive & (1 << i) == 0 {
                        continue;
                    }
                    let text = option.as_bytes();
                    let want = if pos < text.len() { text[pos] } else { b'"' };
                    if b == want {
                        still |= 1 << i;
                        finished |= pos == text.len();
                    }
                }
                if still == 0 {
                    Step::Reject
                } else if finished {
                    Step::Pop
                } else {
                    Step::Stay(Frame::Fixed {
                        options,
                        alive: still,
                        pos: pos + 1,
                    })
                }
            }
            Frame::Number {
                state,
                integer,
                digits,
            } => number_byte(state, integer, digits, b),
            Frame::Literal { text, pos } => {
                if b != text[pos] {
                    Step::Reject
                } else if pos + 1 == text.len() {
                    Step::Pop
                } else {
                    Step::Stay(Frame::Literal { text, pos: pos + 1 })
                }
            }
        }
    }

    /// The shortest bytes that complete the input into a valid value: what
    /// to append when the token budget runs out in the middle.
    pub fn completion(&self) -> Vec<u8> {
        let mut out = Vec::new();
        if let Some((name, pos)) = self.key {
            out.extend_from_slice(&name.as_bytes()[pos..]);
            out.push(b'"');
        }
        for frame in self.stack[..self.depth].iter().rev().flatten() {
            match *frame {
                Frame::Value(schema) => minimal(schema, &mut out),
                Frame::Object { props, next, stage } => {
                    let mut i = next;
                    match stage {
                        Stage::Key => {
                            out.push(b':');
                            minimal(&props[i].1, &mut out);
                            i += 1;
                        }
                        Stage::Open | Stage::Comma => {
                            if i < props.len() {
                                property(&props[i], &mut out);
                                i += 1;
                            }
                        }
                        Stage::Value => {}
                    }
                    for p in &props[i..] {
                        out.push(b',');
                        property(p, &mut out);
                    }
                    out.push(b'}');
                }
                Frame::AnyObject { stage } => match stage {
                    Stage::Key => out.extend_from_slice(b":0}"),
                    Stage::Comma => out.extend_from_slice(b"\"\":0}"),
                    Stage::Open | Stage::Value => out.push(b'}'),
                },
                Frame::Array { item, stage } => {
                    if stage == Stage::Comma {
                        minimal(item, &mut out);
                    }
                    out.push(b']');
                }
                Frame::Str { expect, .. } => {
                    match expect {
                        Expect::Char => {}
                        Expect::Escape => out.push(b'n'),
                        Expect::Hex { left, .. } => {
                            out.extend(std::iter::repeat_n(b'0', usize::from(left)));
                        }
                        // The smallest valid end of the character.
                        Expect::Utf8 { left, lo, .. } => {
                            out.push(lo);
                            out.extend(std::iter::repeat_n(0x80, usize::from(left - 1)));
                        }
                    }
                    out.push(b'"');
                }
                Frame::Fixed {
                    options,
                    alive,
                    pos,
                } => {
                    let best = options
                        .iter()
                        .enumerate()
                        .filter(|&(i, _)| alive & (1 << i) != 0)
                        .map(|(_, o)| o)
                        .min_by_key(|o| o.len())
                        .expect("a live option");
                    out.extend_from_slice(&best.as_bytes()[pos.min(best.len())..]);
                    out.push(b'"');
                }
                Frame::Number { state, .. } => {
                    if !state.complete() {
                        out.push(b'0');
                    }
                }
                Frame::Literal { text, pos } => out.extend_from_slice(&text[pos..]),
            }
        }
        out
    }
}

/// The first byte of a value of `schema` (or whitespace before it).
fn start_value(schema: &Schema, b: u8) -> Step<'_> {
    if is_ws(b) {
        return Step::Whitespace;
    }
    let number = |integer| match b {
        b'-' => Step::Replace(Frame::Number {
            state: Num::Minus,
            integer,
            digits: 0,
        }),
        b'0' => Step::Replace(Frame::Number {
            state: Num::Zero,
            integer,
            digits: 1,
        }),
        b'1'..=b'9' => Step::Replace(Frame::Number {
            state: Num::Int,
            integer,
            digits: 1,
        }),
        _ => Step::Reject,
    };
    let literal = |text: &'static [u8]| {
        if text.len() == 1 {
            Step::Pop
        } else {
            Step::Replace(Frame::Literal { text, pos: 1 })
        }
    };
    match (schema, b) {
        (Schema::Object(props), b'{') => Step::Replace(Frame::Object {
            props,
            next: 0,
            stage: Stage::Open,
        }),
        (Schema::Any, b'{') => Step::Replace(Frame::AnyObject { stage: Stage::Open }),
        (Schema::Array(item), b'[') => Step::Replace(Frame::Array {
            item,
            stage: Stage::Open,
        }),
        (Schema::Any, b'[') => Step::Replace(Frame::Array {
            item: &Schema::Any,
            stage: Stage::Open,
        }),
        (Schema::String { max_len }, b'"') => Step::Replace(Frame::Str {
            len: 0,
            max: *max_len,
            expect: Expect::Char,
        }),
        (Schema::Any, b'"') => Step::Replace(Frame::Str {
            len: 0,
            max: usize::MAX,
            expect: Expect::Char,
        }),
        (Schema::Enum(options), b'"') => Step::Replace(Frame::Fixed {
            options,
            alive: u64::MAX >> (64 - options.len()),
            pos: 0,
        }),
        (Schema::Integer, _) => number(true),
        (Schema::Number | Schema::Any, b'-' | b'0'..=b'9') => number(false),
        (Schema::Boolean | Schema::Any, b't') => literal(b"true"),
        (Schema::Boolean | Schema::Any, b'f') => literal(b"false"),
        (Schema::Null | Schema::Any, b'n') => literal(b"null"),
        _ => Step::Reject,
    }
}

/// One byte inside a string. An escape or a multi-byte character is only
/// started if all of it fits within `max`, so a string can always be
/// closed; and every string is valid UTF-8.
fn string_byte<'s>(len: usize, max: usize, expect: Expect, b: u8) -> Step<'s> {
    // `room`: bytes, this one included, that must still fit.
    let next = |expect, room: usize| {
        if len + room > max {
            Step::Reject
        } else {
            Step::Stay(Frame::Str {
                len: len + 1,
                max,
                expect,
            })
        }
    };
    let utf8 = |left, lo, hi| next(Expect::Utf8 { left, lo, hi }, 1 + usize::from(left));
    match expect {
        Expect::Char => match b {
            b'"' => Step::Pop,
            b'\\' => next(Expect::Escape, 2),
            0x20..=0x7f => next(Expect::Char, 1),
            0xc2..=0xdf => utf8(1, 0x80, 0xbf),
            0xe0 => utf8(2, 0xa0, 0xbf),
            0xed => utf8(2, 0x80, 0x9f),
            0xe1..=0xef => utf8(2, 0x80, 0xbf),
            0xf0 => utf8(3, 0x90, 0xbf),
            0xf1..=0xf3 => utf8(3, 0x80, 0xbf),
            0xf4 => utf8(3, 0x80, 0x8f),
            // Control characters (they must be escaped), a continuation
            // byte without a start, or a byte never valid in UTF-8.
            _ => Step::Reject,
        },
        Expect::Escape => match b {
            b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' => next(Expect::Char, 1),
            b'u' => next(Expect::Hex { left: 4, d: false }, 5),
            _ => Step::Reject,
        },
        Expect::Hex { left, d } => {
            let surrogate = d && left == 3 && matches!(b, b'8'..=b'9' | b'a'..=b'f' | b'A'..=b'F');
            if !b.is_ascii_hexdigit() || surrogate {
                Step::Reject
            } else if left == 1 {
                next(Expect::Char, 1)
            } else {
                let d = left == 4 && matches!(b, b'd' | b'D');
                next(Expect::Hex { left: left - 1, d }, 1)
            }
        }
        Expect::Utf8 { left, lo, hi } => {
            if !(lo..=hi).contains(&b) {
                Step::Reject
            } else if left == 1 {
                next(Expect::Char, 1)
            } else {
                next(
                    Expect::Utf8 {
                        left: left - 1,
                        lo: 0x80,
                        hi: 0xbf,
                    },
                    1,
                )
            }
        }
    }
}

/// One byte inside a number.
fn number_byte<'s>(state: Num, integer: bool, digits: u8, b: u8) -> Step<'s> {
    let to = |state, digits| {
        let limit = if matches!(state, Num::ExpDigits) {
            MAX_EXPONENT_DIGITS
        } else {
            MAX_DIGITS
        };
        if digits > limit {
            Step::Reject
        } else {
            Step::Stay(Frame::Number {
                state,
                integer,
                digits,
            })
        }
    };
    match (state, b) {
        (Num::Minus, b'0') => to(Num::Zero, 1),
        (Num::Minus, b'1'..=b'9') => to(Num::Int, 1),
        (Num::Int, b'0'..=b'9') => to(Num::Int, digits + 1),
        (Num::Zero | Num::Int, b'.') if !integer => to(Num::Dot, 0),
        (Num::Dot, b'0'..=b'9') => to(Num::Frac, 1),
        (Num::Frac, b'0'..=b'9') => to(Num::Frac, digits + 1),
        (Num::Zero | Num::Int | Num::Frac, b'e' | b'E') if !integer => to(Num::Exp, 0),
        (Num::Exp, b'+' | b'-') => to(Num::ExpSign, 0),
        (Num::Exp | Num::ExpSign, b'0'..=b'9') => to(Num::ExpDigits, 1),
        (Num::ExpDigits, b'0'..=b'9') => to(Num::ExpDigits, digits + 1),
        (s, _) if s.complete() => Step::PopAndRetry,
        _ => Step::Reject,
    }
}

/// The shortest value of `schema`.
fn minimal(schema: &Schema, out: &mut Vec<u8>) {
    match schema {
        Schema::Any | Schema::Integer | Schema::Number => out.push(b'0'),
        Schema::Object(props) => {
            out.push(b'{');
            for (i, p) in props.iter().enumerate() {
                if i > 0 {
                    out.push(b',');
                }
                property(p, out);
            }
            out.push(b'}');
        }
        Schema::Array(_) => out.extend_from_slice(b"[]"),
        Schema::String { .. } => out.extend_from_slice(b"\"\""),
        Schema::Boolean => out.extend_from_slice(b"true"),
        Schema::Null => out.extend_from_slice(b"null"),
        Schema::Enum(options) => {
            let best = options.iter().min_by_key(|o| o.len()).expect("an option");
            out.push(b'"');
            out.extend_from_slice(best.as_bytes());
            out.push(b'"');
        }
    }
}

/// `"name":` and the shortest value of a property.
fn property((name, schema): &(String, Schema), out: &mut Vec<u8>) {
    out.push(b'"');
    out.extend_from_slice(name.as_bytes());
    out.extend_from_slice(b"\":");
    minimal(schema, out);
}

#[cfg(test)]
mod tests {
    use super::*;
    use ch15_sampling::Rng;

    /// Whether the matcher accepts every byte of `text` and ends complete.
    fn accepts(schema: &Schema, text: &str) -> bool {
        let mut m = Matcher::new(schema);
        m.push_all(text.as_bytes()) && m.is_complete()
    }

    fn parses(text: &str) -> bool {
        serde_json::from_str::<serde_json::Value>(text).is_ok()
    }

    const VALID: [&str; 10] = [
        "{}",
        "[]",
        r#"{"a":1,"b":[true,false,null],"c":{"d":"e\né"}}"#,
        "-0.5e+10",
        r#""x""#,
        " [1, 2 , 3] ",
        r#"{"nested":[[[]]]}"#,
        "\"h\u{e9}llo, \u{4e16}\u{754c}\"",
        "12",
        "{\n  \"pretty\": [\n    1,\n    2\n  ]\n}",
    ];

    #[test]
    fn valid_json_is_accepted_byte_by_byte() {
        for text in VALID {
            assert!(parses(text), "{text}");
            let mut m = Matcher::new(&Schema::Any);
            for (i, &b) in text.as_bytes().iter().enumerate() {
                assert!(m.push(b), "{text:?} rejected at byte {i}");
            }
            assert!(m.is_complete(), "{text:?} not complete");
        }
    }

    #[test]
    fn invalid_json_is_rejected() {
        for text in [
            "{,}",
            "[1,]",
            r#"{"a" 1}"#,
            "01",
            "1.",
            "tru",
            r#""\x""#,
            r#"{"a":1}}"#,
            "[1 2]",
            "nul",
            "{",
            r#"{"a"}"#,
            "+1",
            "[,1]",
            "\"\u{1}\"",
            "1e",
            "--1",
        ] {
            assert!(!parses(text), "{text:?} is valid JSON after all");
            assert!(!accepts(&Schema::Any, text), "{text:?} accepted");
        }
    }

    /// Whether `text` has an exponent with more digits than the matcher
    /// allows (`e` or `E`, an optional sign, then 3 digits or more).
    fn long_exponent(text: &str) -> bool {
        let b = text.as_bytes();
        (0..b.len()).any(|i| {
            if !matches!(b[i], b'e' | b'E') || i == 0 || !b[i - 1].is_ascii_digit() {
                return false;
            }
            let mut j = i + 1;
            if j < b.len() && matches!(b[j], b'+' | b'-') {
                j += 1;
            }
            b[j..].iter().take_while(|c| c.is_ascii_digit()).count()
                > usize::from(MAX_EXPONENT_DIGITS)
        })
    }

    #[test]
    fn the_matcher_agrees_with_serde_json_on_random_strings() {
        // Short strings over JSON's punctuation, digits and letters: most
        // are invalid, some valid, many are tricky near-misses.
        let alphabet = br#"{}[]",:0123-.eEtrufalsn "#;
        let mut rng = Rng::new(27);
        let mut valid = 0;
        for _ in 0..200_000 {
            let len = 1 + (rng.next_u64() % 10) as usize;
            let text: String = (0..len)
                .map(|_| char::from(alphabet[(rng.next_u64() % alphabet.len() as u64) as usize]))
                .collect();
            if long_exponent(&text) {
                // Valid JSON perhaps, but beyond MAX_EXPONENT_DIGITS.
                assert!(!accepts(&Schema::Any, &text), "{text:?}");
                continue;
            }
            let want = parses(&text);
            assert_eq!(accepts(&Schema::Any, &text), want, "{text:?}");
            valid += usize::from(want);
        }
        assert!(
            valid > 1000,
            "only {valid} valid strings: the test is too easy"
        );
    }

    fn person() -> Schema {
        Schema::Object(vec![
            ("name".into(), Schema::String { max_len: 20 }),
            ("age".into(), Schema::Integer),
            (
                "role".into(),
                Schema::Enum(vec!["engineer".into(), "teacher".into(), "artist".into()]),
            ),
            (
                "tags".into(),
                Schema::Array(Box::new(Schema::String { max_len: 8 })),
            ),
        ])
    }

    #[test]
    fn a_schema_fixes_keys_order_and_types() {
        let s = person();
        assert!(accepts(
            &s,
            r#"{"name": "Ann", "age": 34, "role": "teacher", "tags": ["a", "b"]}"#
        ));
        assert!(accepts(
            &s,
            r#"{"name":"","age":-1,"role":"artist","tags":[]}"#
        ));
        for wrong in [
            r#"{"age": 34, "name": "Ann", "role": "teacher", "tags": []}"#, // order
            r#"{"name": "Ann", "age": 34.5, "role": "teacher", "tags": []}"#, // integer
            r#"{"name": "Ann", "age": 34, "role": "pilot", "tags": []}"#,   // enum
            r#"{"name": "Ann", "age": 34, "role": "teacher"}"#,             // missing
            r#"{"name": "Ann", "age": 34, "role": "teacher", "tags": [], "x": 1}"#, // extra
            r#"{"name": "a very long name indeed", "age": 1, "role": "artist", "tags": []}"#,
            r#"{"name": "Ann", "age": 34, "role": "teacher", "tags": ["too long!"]}"#,
        ] {
            assert!(parses(wrong));
            assert!(!accepts(&s, wrong), "{wrong}");
        }
    }

    #[test]
    fn every_prefix_has_a_valid_completion() {
        let s = person();
        let docs = [
            (
                &s,
                r#"{"name": "Aé\n", "age": -120, "role": "engineer", "tags": ["x", "\\"]}"#,
            ),
            (&Schema::Any, VALID[2]),
            (&Schema::Any, VALID[3]),
            (&Schema::Any, VALID[9]),
        ];
        for (schema, doc) in docs {
            let bytes = doc.as_bytes();
            for cut in 0..bytes.len() {
                let mut m = Matcher::new(schema);
                assert!(m.push_all(&bytes[..cut]));
                let mut whole = bytes[..cut].to_vec();
                whole.extend(m.completion());
                let text = String::from_utf8(whole).expect("ASCII completion");
                assert!(parses(&text), "{text}");
                assert!(accepts(schema, &text), "{text}");
            }
        }
    }

    #[test]
    fn strings_are_valid_utf8() {
        let accepts_bytes = |text: &[u8]| {
            let mut m = Matcher::new(&Schema::Any);
            m.push_all(text) && m.is_complete()
        };
        for good in [
            &b"\"\xc3\xa9\""[..],
            b"\"\xe4\xb8\x96\"",
            b"\"\xf0\x9f\x98\x80\"",
            br#""\u00e9""#,
        ] {
            assert!(accepts_bytes(good), "{good:?}");
            assert!(serde_json::from_slice::<serde_json::Value>(good).is_ok());
        }
        for bad in [
            &b"\"\x80\""[..],    // a continuation byte alone
            b"\"\xc3\"",         // a quote in the middle of a character
            b"\"\xc0\xaf\"",     // an overlong encoding of '/'
            b"\"\xed\xa0\x80\"", // a UTF-16 surrogate, encoded in UTF-8
            br#""\ud800""#,      // a lone surrogate escape
        ] {
            assert!(!accepts_bytes(bad), "{bad:?}");
            assert!(serde_json::from_slice::<serde_json::Value>(bad).is_err());
        }
    }

    #[test]
    fn strings_never_start_an_escape_that_cannot_fit() {
        let s = Schema::String { max_len: 3 };
        assert!(accepts(&s, r#""abc""#));
        assert!(!accepts(&s, r#""abcd""#));
        let mut m = Matcher::new(&s);
        assert!(m.push_all(br#""ab"#));
        assert!(!m.clone().push(b'\\'), "a 2-byte escape does not fit");
        let mut m = Matcher::new(&s);
        assert!(m.push_all(br#""\"#));
        assert!(!m.clone().push(b'u'), "a 6-byte escape does not fit");
        assert_eq!(m.completion(), b"n\"");
    }
}
