"""Generates chapter 16's reference fixtures with the Hugging Face libraries.

Run from the repository root, after ./tools/download_model.sh:

    python tools/make_fixtures.py

Writes, into chapters/16-real-model/fixtures/:

- tokenizer_cases.json: texts and the token ids Hugging Face's tokenizer
  produces for them;
- reference.safetensors: for one chat prompt, the float32 logits at the last
  position and the hidden state at the last position before every layer
  (plus after the final norm), from transformers running in float32;
- generation.json: prompt ids and greedy continuations for two questions.
"""

import json
import os
import pathlib

import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer
from transformers import AutoModelForCausalLM

ROOT = pathlib.Path(__file__).resolve().parent.parent
MODELS = pathlib.Path(os.environ.get("INFER_MODELS", ROOT / "models"))
MODEL_DIR = MODELS / "smollm2-135m-instruct"
OUT = ROOT / "chapters" / "16-real-model" / "fixtures"

TEXTS = [
    "",
    "Hello, world!",
    "The capital of France is Paris.",
    "  leading spaces",
    "trailing spaces   ",
    "a   b",
    "tabs\tand\nnewlines\n\n\nx",
    "windows\r\nline endings\r\n",
    "numbers 12345 and 3.14159, year 2024",
    "abc123def",
    "don't won't I'm they're we've I'll he'd",
    "DON'T SHOUT",
    "naïve café résumé",
    "日本語のテキスト",
    "emoji 👋🏽 and 🤖",
    "Здравствуй, мир",
    "مرحبا بالعالم",
    "non breaking space",
    "supercalifragilisticexpialidocious",
    "   ",
    "fn main() {\n    println!(\"hi\");\n}\n",
    "<|im_start|>user\nHi<|im_end|>\n",
    "text<|im_end|>more<|endoftext|>",
    "not special: <|im_end|x <im_start>",
    "a control\x04character and a DEL\x7f",
]

SYSTEM = "You are a helpful AI assistant named SmolLM, trained by Hugging Face"


def chat_prompt(user):
    """SmolLM2's chat template, written out (see chapter 16, section 3.4)."""
    return (
        f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def one_per_line(items):
    """A JSON array with one compact item per line: small diffs, easy to read."""
    return "[\n" + ",\n".join(json.dumps(x, ensure_ascii=False) for x in items) + "\n]\n"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))

    novel = (ROOT / "chapters" / "15-sampling" / "data" / "pride-and-prejudice-1-6.txt").read_text()
    cases = [{"text": t, "ids": tok.encode(t, add_special_tokens=False).ids} for t in TEXTS]
    cases.append({"file": "chapters/15-sampling/data/pride-and-prejudice-1-6.txt",
                  "ids": tok.encode(novel, add_special_tokens=False).ids})
    (OUT / "tokenizer_cases.json").write_text(one_per_line(cases))

    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32)
    model.eval()

    # Logits and hidden states for one prompt.
    prompt = chat_prompt("What is the capital of France?")
    ids = tok.encode(prompt, add_special_tokens=False).ids
    with torch.no_grad():
        out = model(torch.tensor([ids]), output_hidden_states=True)
    logits = out.logits[0, -1].contiguous()
    hidden = torch.stack([h[0, -1] for h in out.hidden_states]).contiguous()
    save_file(
        {"logits_last": logits, "hidden_last": hidden},
        str(OUT / "reference.safetensors"),
        metadata={
            "prompt": prompt,
            "prompt_ids": json.dumps(ids),
            "note": "hidden_last[i] is the input of layer i at the last position; "
                    "hidden_last[30] is the output of the final norm",
        },
    )
    print("prompt tokens:", len(ids), "top logits:", torch.topk(logits, 5))

    # Greedy continuations (generation stops at <|im_end|>, id 2).
    runs = []
    for question in ["What is the capital of France?",
                     "Explain in two sentences why the sky is blue."]:
        prompt = chat_prompt(question)
        ids = tok.encode(prompt, add_special_tokens=False).ids
        with torch.no_grad():
            generated = model.generate(torch.tensor([ids]), max_new_tokens=48, do_sample=False)
        new_ids = generated[0, len(ids):].tolist()
        text = tok.decode(new_ids, skip_special_tokens=False)
        print("greedy:", repr(text))
        runs.append({"question": question, "prompt_ids": ids,
                     "greedy_ids": new_ids, "greedy_text": text})
    (OUT / "generation.json").write_text(one_per_line(runs))


if __name__ == "__main__":
    main()
