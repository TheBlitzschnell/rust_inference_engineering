#!/usr/bin/env bash
# Downloads a SmolLM2 Instruct model (weights, config and tokenizer) from
# Hugging Face into models/, and checks the weights against known hashes.
#
#   ./tools/download_model.sh          # SmolLM2-135M-Instruct (about 270 MB)
#   ./tools/download_model.sh 360M     # SmolLM2-360M-Instruct (about 720 MB)
set -euo pipefail

size="${1:-135M}"
case "$size" in
  135M) sha="5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c" ;;
  360M) sha="e6bffe7435d7ddc10fd3b9a9efd429dafbacb1cb17015fb5562664e7532bf86e" ;;
  *) echo "usage: $0 [135M|360M]" >&2; exit 1 ;;
esac

repo="HuggingFaceTB/SmolLM2-${size}-Instruct"
root="$(cd "$(dirname "$0")/.." && pwd)"
dir="$root/models/smollm2-$(echo "$size" | tr '[:upper:]' '[:lower:]')-instruct"
mkdir -p "$dir"

for f in config.json generation_config.json tokenizer.json tokenizer_config.json \
         special_tokens_map.json model.safetensors; do
  if [ -s "$dir/$f" ]; then
    echo "already have $f"
    continue
  fi
  echo "downloading $f"
  curl -fL --retry 3 -o "$dir/$f.part" "https://huggingface.co/$repo/resolve/main/$f"
  mv "$dir/$f.part" "$dir/$f"
done

echo "checking model.safetensors"
actual="$(sha256sum "$dir/model.safetensors" | cut -d' ' -f1)"
if [ "$actual" != "$sha" ]; then
  echo "hash mismatch for $dir/model.safetensors" >&2
  echo "  expected $sha" >&2
  echo "  got      $actual" >&2
  exit 1
fi
echo "ok: $dir"
