//! A minimal HTTP/1.1 client, for the demo and the tests.
//!
//! Real programs should use a full client (`reqwest`, `hyper`). This one is
//! small enough to read in a few minutes, and shows what actually travels
//! over the connection: a request line, headers, a blank line, a body; and
//! for streamed responses, `Transfer-Encoding: chunked` framing around
//! server-sent events. One request per connection (`Connection: close`).

use std::io;
use std::net::SocketAddr;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpStream;

/// A response whose body has not been read yet.
pub struct Response {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    body: Body,
    reader: BufReader<TcpStream>,
}

/// How the end of the body is marked.
enum Body {
    /// `Content-Length: n`; the number of bytes still to read.
    Length(usize),
    /// `Transfer-Encoding: chunked`: each piece is preceded by its size.
    Chunked { done: bool },
    /// Neither: the body ends when the server closes the connection.
    UntilClose,
}

/// Sends one request and reads the status line and headers.
pub async fn send(
    addr: SocketAddr,
    method: &str,
    path: &str,
    body: Option<&str>,
) -> io::Result<Response> {
    let mut stream = TcpStream::connect(addr).await?;
    stream.set_nodelay(true)?;
    let body = body.unwrap_or("");
    let request = format!(
        "{method} {path} HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\n\
         Content-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    );
    stream.write_all(request.as_bytes()).await?;
    let mut reader = BufReader::new(stream);

    let mut line = String::new();
    reader.read_line(&mut line).await?;
    // "HTTP/1.1 200 OK"
    let status = line
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| bad(format!("bad status line {line:?}")))?;
    let mut headers = Vec::new();
    loop {
        line.clear();
        reader.read_line(&mut line).await?;
        let l = line.trim_end();
        if l.is_empty() {
            break;
        }
        let (name, value) = l
            .split_once(':')
            .ok_or_else(|| bad(format!("bad header {l:?}")))?;
        headers.push((name.trim().to_ascii_lowercase(), value.trim().to_owned()));
    }
    let find = |name: &str| headers.iter().find(|(n, _)| n == name).map(|(_, v)| v);
    let body = if find("transfer-encoding").is_some_and(|v| v.eq_ignore_ascii_case("chunked")) {
        Body::Chunked { done: false }
    } else if let Some(n) = find("content-length") {
        Body::Length(n.parse().map_err(|_| bad(format!("bad length {n:?}")))?)
    } else {
        Body::UntilClose
    };
    Ok(Response {
        status,
        headers,
        body,
        reader,
    })
}

fn bad(what: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, what)
}

impl Response {
    /// The value of a header (names are compared in lower case).
    pub fn header(&self, name: &str) -> Option<&str> {
        let name = name.to_ascii_lowercase();
        self.headers
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(_, v)| v.as_str())
    }

    /// The next piece of the body as it arrives, or `None` at the end.
    pub async fn chunk(&mut self) -> io::Result<Option<Vec<u8>>> {
        match &mut self.body {
            Body::Length(0) | Body::Chunked { done: true } => Ok(None),
            Body::Length(left) => {
                let mut buf = vec![0; (*left).min(8192)];
                let n = self.reader.read(&mut buf).await?;
                if n == 0 {
                    return Err(bad("connection closed early".into()));
                }
                *left -= n;
                buf.truncate(n);
                Ok(Some(buf))
            }
            Body::Chunked { done } => {
                // "<size in hex>\r\n<size bytes>\r\n", ending with a size of 0.
                let mut line = String::new();
                self.reader.read_line(&mut line).await?;
                let hex = line.trim_end().split(';').next().unwrap_or("");
                let size = usize::from_str_radix(hex, 16)
                    .map_err(|_| bad(format!("bad chunk size {line:?}")))?;
                if size == 0 {
                    *done = true;
                    return Ok(None);
                }
                let mut buf = vec![0; size + 2];
                self.reader.read_exact(&mut buf).await?;
                buf.truncate(size);
                Ok(Some(buf))
            }
            Body::UntilClose => {
                let mut buf = vec![0; 8192];
                let n = self.reader.read(&mut buf).await?;
                buf.truncate(n);
                Ok((n > 0).then_some(buf))
            }
        }
    }

    /// The whole body as text.
    pub async fn text(mut self) -> io::Result<String> {
        let mut all = Vec::new();
        while let Some(piece) = self.chunk().await? {
            all.extend_from_slice(&piece);
        }
        String::from_utf8(all).map_err(|e| bad(e.to_string()))
    }

    /// Reads the body as server-sent events.
    pub fn events(self) -> Events {
        Events {
            response: self,
            buffer: Vec::new(),
        }
    }
}

/// Server-sent events: `data: ...` lines, each event ended by a blank line.
pub struct Events {
    response: Response,
    /// Bytes received but not yet returned. Kept as bytes: a network chunk
    /// may end in the middle of a UTF-8 character, an event never does.
    buffer: Vec<u8>,
}

impl Events {
    /// The `data` of the next event, or `None` at the end of the stream.
    /// Events without data (comments such as keep-alives) are skipped.
    pub async fn next(&mut self) -> io::Result<Option<String>> {
        loop {
            if let Some(end) = self.buffer.windows(2).position(|w| w == b"\n\n") {
                let bytes: Vec<u8> = self.buffer.drain(..end + 2).collect();
                let event = String::from_utf8(bytes).map_err(|e| bad(e.to_string()))?;
                let data: Vec<&str> = event
                    .lines()
                    .filter_map(|l| l.strip_prefix("data:"))
                    .map(|d| d.strip_prefix(' ').unwrap_or(d))
                    .collect();
                if !data.is_empty() {
                    return Ok(Some(data.join("\n")));
                }
                continue;
            }
            match self.response.chunk().await? {
                Some(bytes) => self.buffer.extend_from_slice(&bytes),
                None => return Ok(None),
            }
        }
    }
}
