//! The JSON of OpenAI's chat completions API: the subset this server
//! accepts and everything it returns.
//!
//! Clients written for OpenAI (their SDKs, `curl` scripts, tools that let
//! you set a "base URL") then work against this server unchanged. The
//! types mirror the wire format field for field; `serde` does the
//! (de)serialization.

use serde::{Deserialize, Serialize};

/// `POST /v1/chat/completions` request body.
///
/// Unknown fields are ignored (`serde`'s default), as OpenAI-compatible
/// servers usually do: clients send fields a small server does not use.
#[derive(Clone, Debug, Default, Deserialize)]
pub struct ChatRequest {
    /// Accepted and echoed back; this server has one model.
    #[serde(default)]
    pub model: Option<String>,
    pub messages: Vec<ChatMessage>,
    /// Newer clients send `max_completion_tokens`, older ones `max_tokens`.
    #[serde(default, alias = "max_completion_tokens")]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    /// Not in OpenAI's API; vLLM and others accept it.
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub stop: Option<Stop>,
    /// Number of answers to generate; only 1 is supported.
    #[serde(default)]
    pub n: Option<usize>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
}

/// `stop` is either one string or a list of strings.
#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
pub enum Stop {
    One(String),
    Many(Vec<String>),
}

impl Stop {
    pub fn into_vec(self) -> Vec<String> {
        match self {
            Self::One(s) => vec![s],
            Self::Many(v) => v,
        }
    }
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct StreamOptions {
    /// Send a last chunk with the token counts.
    #[serde(default)]
    pub include_usage: bool,
}

/// One message of the conversation. (OpenAI also allows `content` to be a
/// list of parts, for images and audio; this server accepts text only.)
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// The response when `stream` is false.
#[derive(Clone, Debug, Serialize)]
pub struct ChatCompletion {
    pub id: String,
    pub object: &'static str,
    /// Unix time, in seconds.
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

#[derive(Clone, Debug, Serialize)]
pub struct Choice {
    pub index: usize,
    pub message: ChatMessage,
    /// `"stop"` (end-of-turn token or stop string) or `"length"`.
    pub finish_reason: &'static str,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

/// One server-sent event of a streamed response.
#[derive(Clone, Debug, Serialize)]
pub struct ChatChunk {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChunkChoice>,
    /// Only in the last chunk, and only if the client asked for it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ChunkChoice {
    pub index: usize,
    pub delta: Delta,
    /// `null` until the last content chunk.
    pub finish_reason: Option<&'static str>,
}

/// What this chunk adds to the message.
#[derive(Clone, Debug, Default, Serialize)]
pub struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

/// Every error is returned as `{"error": {...}}` with an HTTP status.
#[derive(Clone, Debug, Serialize)]
pub struct ErrorBody {
    pub error: ErrorDetail,
}

#[derive(Clone, Debug, Serialize)]
pub struct ErrorDetail {
    pub message: String,
    #[serde(rename = "type")]
    pub kind: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<&'static str>,
}

/// `GET /v1/models`.
#[derive(Clone, Debug, Serialize)]
pub struct ModelList {
    pub object: &'static str,
    pub data: Vec<ModelCard>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ModelCard {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub owned_by: &'static str,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_minimal_request_parses_with_defaults() {
        let r: ChatRequest =
            serde_json::from_str(r#"{"messages":[{"role":"user","content":"hi"}]}"#).unwrap();
        assert_eq!(r.messages.len(), 1);
        assert!(!r.stream);
        assert!(r.max_tokens.is_none() && r.stop.is_none());
    }

    #[test]
    fn stop_is_a_string_or_a_list_and_max_tokens_has_two_names() {
        let r: ChatRequest = serde_json::from_str(
            r#"{"messages":[],"stop":"\n","max_completion_tokens":5,"extra":true}"#,
        )
        .unwrap();
        assert_eq!(r.stop.unwrap().into_vec(), vec!["\n"]);
        assert_eq!(r.max_tokens, Some(5));
        let r: ChatRequest = serde_json::from_str(r#"{"messages":[],"stop":["a","b"]}"#).unwrap();
        assert_eq!(r.stop.unwrap().into_vec(), vec!["a", "b"]);
    }

    #[test]
    fn a_chunk_leaves_out_what_it_does_not_carry() {
        let c = ChatChunk {
            id: "x".into(),
            object: "chat.completion.chunk",
            created: 0,
            model: "m".into(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: Delta {
                    content: Some("Hi".into()),
                    ..Delta::default()
                },
                finish_reason: None,
            }],
            usage: None,
        };
        assert_eq!(
            serde_json::to_string(&c).unwrap(),
            r#"{"id":"x","object":"chat.completion.chunk","created":0,"model":"m","choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}]}"#
        );
    }
}
