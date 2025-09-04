#!/usr/bin/env bash
set -euo pipefail

# Convert a meme dataset into:
# 1) JSONL for clustering/evaluation
# 2) LLaVA JSON for training

INPUT=${INPUT:-"./data/raw/meme.csv"}
IMAGE_ROOT=${IMAGE_ROOT:-"./data/images"}
OUT_DIR=${OUT_DIR:-"./data/converted"}

IMAGE_COL=${IMAGE_COL:-"image"}
TEXT_COL=${TEXT_COL:-"text"}
LABEL_COL=${LABEL_COL:-"label"}
DESC_COL=${DESC_COL:-"description"}

POS_LABELS=${POS_LABELS:-"1,harmful,hate,offensive"}
NEG_LABELS=${NEG_LABELS:-"0,harmless,benign,safe"}

mkdir -p "$OUT_DIR"

python -m ssmoe.data_convert \
  --input "$INPUT" \
  --image-root "$IMAGE_ROOT" \
  --image-col "$IMAGE_COL" \
  --text-col "$TEXT_COL" \
  --label-col "$LABEL_COL" \
  --desc-col "$DESC_COL" \
  --out-jsonl "$OUT_DIR/meme.jsonl" \
  --out-llava "$OUT_DIR/meme_llava.json" \
  --pos-labels "$POS_LABELS" \
  --neg-labels "$NEG_LABELS"

echo "Converted files saved to: $OUT_DIR"

