# Tools

Scripts that support the course. None of them is needed to build or test the Rust code.

## `download_model.sh`

Downloads SmolLM2-Instruct from Hugging Face into `models/` and checks the weights' SHA-256:

```bash
./tools/download_model.sh          # SmolLM2-135M-Instruct, about 270 MB
./tools/download_model.sh 360M     # SmolLM2-360M-Instruct, about 720 MB (chapter 26)
```

Chapters 9 and 16 onward use these files. Their tests skip, with a message, when the files are missing. If you keep models somewhere else, point `INFER_MODELS` at the directory that contains `smollm2-135m-instruct/`; the Rust code and `make_fixtures.py` both read it.

## `make_fixtures.py`

Regenerates the reference outputs in [`chapters/16-real-model/fixtures/`](../chapters/16-real-model/fixtures/) with the Hugging Face Python libraries. The fixtures are committed, so you only need this if you change the prompts or want to check them yourself.

It needs Python 3.10 or later with PyTorch (the CPU build is enough), `transformers`, `tokenizers` and `safetensors`:

```bash
python3 -m venv tools/.venv
tools/.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
tools/.venv/bin/pip install transformers tokenizers safetensors
./tools/download_model.sh
tools/.venv/bin/python tools/make_fixtures.py
```

`tools/.venv` is ignored by git. The fixtures in the repository were made with PyTorch 2.14.0 (CPU), transformers 5.17.0 and tokenizers 0.23.2. The script writes:

| File | Contents |
|---|---|
| `tokenizer_cases.json` | Texts (including awkward ones: whitespace runs, digits, emoji, control bytes, special tokens) and the token ids Hugging Face's tokenizer gives them, plus the ids of a 42 KB text. |
| `reference.safetensors` | For one chat prompt: the `float32` logits at the last position, and the hidden state at the last position before every layer and after the final norm. |
| `generation.json` | Two chat prompts, their token ids and Hugging Face's greedy continuations. |

The model runs in `float32` (the `bf16` weights converted exactly), which is also what our engine computes with, so the two should agree to within rounding.
