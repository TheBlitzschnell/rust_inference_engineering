//! SmolLM2's chat template: how a conversation becomes one prompt string.

/// The system message the template inserts when a conversation has none.
pub const DEFAULT_SYSTEM: &str =
    "You are a helpful AI assistant named SmolLM, trained by Hugging Face";

/// One turn of a conversation.
#[derive(Clone, Copy, Debug)]
pub struct Message<'a> {
    /// `"system"`, `"user"` or `"assistant"`.
    pub role: &'a str,
    pub content: &'a str,
}

/// Renders a conversation the way SmolLM2's `chat_template` (a Jinja
/// template in `tokenizer_config.json`) does, ending with the start of the
/// assistant's turn so the model's continuation is its answer:
///
/// ```text
/// <|im_start|>system
/// You are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>
/// <|im_start|>user
/// What is the capital of France?<|im_end|>
/// <|im_start|>assistant
/// ```
pub fn chat_prompt(messages: &[Message<'_>]) -> String {
    let mut prompt = String::new();
    if messages.first().is_none_or(|m| m.role != "system") {
        push_turn(&mut prompt, "system", DEFAULT_SYSTEM);
    }
    for m in messages {
        push_turn(&mut prompt, m.role, m.content);
    }
    prompt.push_str("<|im_start|>assistant\n");
    prompt
}

fn push_turn(prompt: &mut String, role: &str, content: &str) {
    prompt.push_str("<|im_start|>");
    prompt.push_str(role);
    prompt.push('\n');
    prompt.push_str(content);
    prompt.push_str("<|im_end|>\n");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_default_system_message_is_added() {
        let p = chat_prompt(&[Message {
            role: "user",
            content: "Hi",
        }]);
        assert_eq!(
            p,
            "<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by \
             Hugging Face<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn an_explicit_system_message_replaces_the_default() {
        let p = chat_prompt(&[
            Message {
                role: "system",
                content: "Be brief.",
            },
            Message {
                role: "user",
                content: "Hi",
            },
        ]);
        assert!(p.starts_with("<|im_start|>system\nBe brief.<|im_end|>\n"));
        assert!(!p.contains("SmolLM"));
    }
}
