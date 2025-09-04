#!/usr/bin/env bash
set -euo pipefail

# Example pipeline to evaluate harmful meme detection with SSMoE components.
# 1) Semantic clustering using Nomic text/vision embeddings
# 2) Cluster-targeted prompt generation via API (OpenAI or Ollama)
# 3) Evaluation with cluster-conditioned prompting and routing

# -------- User-configurable paths --------
DATA_JSONL=${DATA_JSONL:-"./data/meme_val.jsonl"}          # fields: image, text, [label], [description]
IMAGE_DIR=${IMAGE_DIR:-"./data/images"}
WORKDIR=${WORKDIR:-"./out_ssmoe"}

# Model checkpoints
MODEL_BASE=${MODEL_BASE:-"./checkpoints/llava-v1.5-13b"}
MODEL_PATH=${MODEL_PATH:-"./checkpoints/llava-v1.5-13b-lora-mohle"}

# -------- Clustering config --------
CLUSTERS=${CLUSTERS:-3}
TEXT_MODEL=${TEXT_MODEL:-"nomic-ai/nomic-embed-text-v1.5"}
VISION_MODEL=${VISION_MODEL:-"nomic-ai/nomic-embed-vision-v1.5"}

# -------- Prompt generation config --------
PROVIDER=${PROVIDER:-"openai"}  # or "ollama"
OPENAI_MODEL=${OPENAI_MODEL:-"gpt-4o-mini"}
OLLAMA_MODEL=${OLLAMA_MODEL:-"llama3.1:8b-instruct"}

mkdir -p "$WORKDIR"

echo "[1/3] Clustering samples into $CLUSTERS clusters..."
python -m ssmoe.cluster \
  "$DATA_JSONL" "$WORKDIR/cluster" \
  --clusters "$CLUSTERS" \
  --feature_type concat \
  --text_model "$TEXT_MODEL" \
  --vision_model "$VISION_MODEL"

echo "[2/3] Generating cluster-targeted prompts via $PROVIDER..."
if [[ "$PROVIDER" == "openai" ]]; then
  python -m ssmoe.prompt_injection \
    "$DATA_JSONL" "$WORKDIR/cluster/assignments.jsonl" "$WORKDIR/prompts" \
    --provider openai \
    --openai_model "$OPENAI_MODEL"
else
  python -m ssmoe.prompt_injection \
    "$DATA_JSONL" "$WORKDIR/cluster/assignments.jsonl" "$WORKDIR/prompts" \
    --provider ollama \
    --ollama_model "$OLLAMA_MODEL"
fi

echo "[3/3] Running evaluation with cluster-conditioned prompts and routing..."
python -m ssmoe.eval_meme \
  --model-base "$MODEL_BASE" \
  --model-path "$MODEL_PATH" \
  --data "$DATA_JSONL" \
  --image-folder "$IMAGE_DIR" \
  --assignments "$WORKDIR/cluster/assignments.jsonl" \
  --cluster-prompts "$WORKDIR/prompts/cluster_prompts.json" \
  --out "$WORKDIR/preds.jsonl" \
  --batch-size 4 \
  --temperature 0.0

echo "Done. Results saved to $WORKDIR/preds.jsonl"

