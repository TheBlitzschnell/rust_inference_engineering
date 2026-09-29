//! Maintenance tasks for the course repository.
//!
//! `cargo xtask check` runs two checks over every Markdown file:
//!
//! 1. **Snippets.** A lesson quotes source code like this:
//!
//!    ````text
//!    <!-- file: src/lib.rs -->
//!    ```rust
//!    pub fn dot(a: &[f32], b: &[f32]) -> f32 {
//!    ```
//!    ````
//!
//!    The marker names a file relative to the Markdown file. The fenced block
//!    after it must appear in that file, line for line (indentation may be
//!    shifted as a whole). A line that is exactly `// ...` splits the block
//!    into pieces; each piece must appear, in order.
//!
//! 2. **Links.** Every relative Markdown link `[text](path)` must point to a
//!    file or directory that exists.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

fn main() -> ExitCode {
    let task = std::env::args().nth(1).unwrap_or_default();
    let root = repo_root();
    let markdown = markdown_files(&root);
    let mut errors = Vec::new();
    let mut snippets = 0;
    match task.as_str() {
        "check" | "check-snippets" | "check-links" => {
            for md in &markdown {
                let text = fs::read_to_string(md).expect("read markdown");
                if task != "check-links" {
                    snippets += check_snippets(md, &text, &mut errors);
                }
                if task != "check-snippets" {
                    check_links(md, &text, &mut errors);
                }
            }
        }
        _ => {
            eprintln!("usage: cargo xtask <check | check-snippets | check-links>");
            return ExitCode::FAILURE;
        }
    }
    for e in &errors {
        eprintln!("error: {e}");
    }
    println!(
        "checked {} markdown files, {snippets} code snippets: {} problem(s)",
        markdown.len(),
        errors.len()
    );
    if errors.is_empty() {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

/// The workspace root is the parent of this crate's manifest directory.
fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask lives one level below the root")
        .to_path_buf()
}

/// Every `.md` file outside `target/` and hidden directories.
fn markdown_files(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in fs::read_dir(&dir).expect("read dir") {
            let path = entry.expect("dir entry").path();
            let name = path.file_name().unwrap_or_default().to_string_lossy();
            if name.starts_with('.') || name == "target" || name == "models" {
                continue;
            }
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|e| e == "md") {
                out.push(path);
            }
        }
    }
    out.sort();
    out
}

fn check_snippets(md: &Path, text: &str, errors: &mut Vec<String>) -> usize {
    let lines: Vec<&str> = text.lines().collect();
    let mut count = 0;
    let mut i = 0;
    while i < lines.len() {
        let Some(target) = parse_marker(lines[i]) else {
            i += 1;
            continue;
        };
        let marker_line = i + 1;
        // Skip blank lines between the marker and the fence.
        i += 1;
        while i < lines.len() && lines[i].trim().is_empty() {
            i += 1;
        }
        if i >= lines.len() || !lines[i].trim_start().starts_with("```") {
            errors.push(format!(
                "{}:{marker_line}: marker not followed by a code fence",
                md.display()
            ));
            continue;
        }
        i += 1;
        let start = i;
        while i < lines.len() && !lines[i].trim_start().starts_with("```") {
            i += 1;
        }
        let block = &lines[start..i.min(lines.len())];
        i += 1;
        count += 1;

        let source_path = md.parent().expect("md has a parent").join(target);
        let Ok(source) = fs::read_to_string(&source_path) else {
            errors.push(format!(
                "{}:{marker_line}: snippet source {} not found",
                md.display(),
                source_path.display()
            ));
            continue;
        };
        let source_lines: Vec<&str> = source.lines().map(str::trim_end).collect();
        if let Err(piece) = contains_in_order(&source_lines, block) {
            errors.push(format!(
                "{}:{marker_line}: snippet piece starting with {:?} not found in {}",
                md.display(),
                piece,
                source_path.display()
            ));
        }
    }
    count
}

/// Parses `<!-- file: some/path -->` and returns the path.
fn parse_marker(line: &str) -> Option<&str> {
    let rest = line.trim().strip_prefix("<!-- file:")?;
    let path = rest.strip_suffix("-->")?.trim();
    (!path.is_empty()).then_some(path)
}

/// Splits `block` on `// ...` lines and checks that each piece occurs in
/// `source`, in order. Returns the first line of the first missing piece.
fn contains_in_order<'a>(source: &[&str], block: &[&'a str]) -> Result<(), &'a str> {
    let mut from = 0;
    for piece in block.split(|l| l.trim() == "// ...") {
        let piece = trim_blank_edges(piece);
        if piece.is_empty() {
            continue;
        }
        match find_piece(source, piece, from) {
            Some(end) => from = end,
            None => return Err(piece[0].trim()),
        }
    }
    Ok(())
}

fn trim_blank_edges<'b, 'a>(piece: &'b [&'a str]) -> &'b [&'a str] {
    let start = piece.iter().position(|l| !l.trim().is_empty());
    let end = piece.iter().rposition(|l| !l.trim().is_empty());
    match (start, end) {
        (Some(s), Some(e)) => &piece[s..=e],
        _ => &[],
    }
}

fn indent(line: &str) -> usize {
    line.len() - line.trim_start().len()
}

/// Finds `piece` in `source` at or after line `from`, allowing the whole piece
/// to be indented differently. Returns the line index just past the match.
fn find_piece(source: &[&str], piece: &[&str], from: usize) -> Option<usize> {
    let first = piece[0].trim_end();
    'outer: for start in from..source.len() {
        if source[start].trim() != first.trim() {
            continue;
        }
        let shift = indent(source[start]) as isize - indent(first) as isize;
        if start + piece.len() > source.len() {
            return None;
        }
        for (k, want) in piece.iter().enumerate() {
            let want = want.trim_end();
            let got = source[start + k];
            if want.trim().is_empty() {
                if !got.trim().is_empty() {
                    continue 'outer;
                }
                continue;
            }
            let want_indent = indent(want) as isize + shift;
            if got.trim() != want.trim() || indent(got) as isize != want_indent {
                continue 'outer;
            }
        }
        return Some(start + piece.len());
    }
    None
}

fn check_links(md: &Path, text: &str, errors: &mut Vec<String>) {
    let dir = md.parent().expect("md has a parent");
    let mut in_fence = false;
    for (n, line) in text.lines().enumerate() {
        if line.trim_start().starts_with("```") {
            in_fence = !in_fence;
            continue;
        }
        if in_fence {
            continue;
        }
        let mut rest = line;
        while let Some(pos) = rest.find("](") {
            rest = &rest[pos + 2..];
            let Some(end) = rest.find(')') else { break };
            let target = &rest[..end];
            rest = &rest[end..];
            if target.contains("://") || target.starts_with('#') || target.starts_with("mailto:") {
                continue;
            }
            let path = target.split('#').next().unwrap_or_default();
            if path.is_empty() {
                continue;
            }
            if !dir.join(path).exists() {
                errors.push(format!(
                    "{}:{}: broken link to {target}",
                    md.display(),
                    n + 1
                ));
            }
        }
    }
}
