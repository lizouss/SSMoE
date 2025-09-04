# SSMoE: Few‑Shot Harmful Meme Detection via Structural Self‑Adaptation

This repo contains the source code of SSMoE

> 🚀 Stay tuned! The full code implementation and dataset access guide will be updated here once our paper is officially published.

## 🧭 Overview

**<b>TL; DR:</b>** SSMoE is a novel Mixture-of-Experts with self-adapting structural design for few-shot harmful meme detection.

<details>
  <summary>🚀 <b>CLICK for the full abstract</b></summary>

The automatic detection of harmful memes is essential for healthy online ecosystems but remains challenging due to the intricate interaction between visual and textual elements. Recently, the remarkable capabilities of multimodal large language models (MLLMs) have significantly enhanced the detection performance, yet scarce labeled data still limits their effectiveness. Although pioneering few-shot studies have explored this regime, they merely leverage surface-level capabilities while ignoring deeper complexities.

To approach the core of the problem, we identify its notorious challenges: (1) heterogeneous multimodal features are complex and may exhibit negative correlations; (2) the semantic patterns underlying single modal are hard to uncover; and (3) the insufficient training samples render models more reliant on commonsense.

To address the challenges, we propose a structural self-adaption mixture-of-experts framework (**SSMoE**) for few-shot harmful meme detection, including universal and specialized experts to foster more effective knowledge sharing, modal synergy, and expert specialization within the MLLM structure. 

Specifically, **SSMoE** integrates four novel components: 

​	(1) **Semantic Data Clustering** module aims to partition heterogeneous source data and mitigate negative transfer; 

​	(2) **Targeted Prompt Injection** module aims to employ a teacher model for providing cluster-specific external guidance; 

​	(3) **Asymmetric Expert Specialization** module aims to introduce shared and specialized experts for efficient parameter adaptation and knowledge specialization; and 

​	(4) **Cluster-conditioned Routing** module aims to dynamically direct inputs to the most relevant expert pathway based on semantic cluster identity. 

Extensive experiments on three benchmark datasets (FHM, MAMI, HarM) demonstrate that SSMoE significantly outperforms state-of-the-art baseline methods, particularly in extremely low-data scenarios.
</details>

## 🛠️ Installation

1. First install `anaconda`, and install `torch`, We recommend installing `torch==2.1.2` and `cuda==11.8`.

```bash
# CUDA 11.8
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
```

2. Then install the packages in `requirements`
```bash
pip install -r requirements.txt
```
## 🗂️ Semantic Clustering
```sh
python -m ssmoe.cluster \
  data/converted/meme.jsonl out_ssmoe/cluster \
  --clusters 8 --feature_type concat \
  --text_model nomic-ai/nomic-embed-text-v1.5 \
  --vision_model nomic-ai/nomic-embed-vision-v1.5
```

## 🧠 Targeted Prompt Injection
Provider can be `OpenAI` or local `Ollama`.
```sh
python -m ssmoe.prompt_injection \
  data/converted/meme.jsonl out_ssmoe/cluster/assignments.jsonl out_ssmoe/prompts \
  --provider openai --openai_model gpt-4o
# or
python -m ssmoe.prompt_injection \
  data/converted/meme.jsonl out_ssmoe/cluster/assignments.jsonl out_ssmoe/prompts \
  --provider ollama --ollama_model gemma3:27b
```
System prompt:
```
You are an expert in multimodal content analysis. I will provide
you with 5 representative meme samples from a cluster of similar memes.
Each sample contains an image description and associated text. Your task
is to generate a concise analytical prompt that would help a model identify
potential harmful patterns common in this cluster.
Input: [Samples with image descriptions and text]
Based on these samples, generate a prompt (50-100 words) that:
1. Identifies key visual-textual patterns to examine
2. Highlights potential harmful elements or rhetorical strategies
3. Notes cultural or contextual factors that may be relevant
4. Avoids being overly prescriptive or biased
Note: Focus on analytical guidance rather than predetermined conclusions.
```

## 🏋️‍♂️ Training
```
python llava/train/train.py \
  --use_lora True --lora_rank 16 --lora_alpha 16 \
  --llm_moe True --llm_moe_num_experts 4 --n_clusters 8 \
  --cluster_prior_weight 1.0 --router_tau 1.0 \
  --cluster_assignments out_ssmoe/cluster/assignments.jsonl \
  --cluster_prompts_file out_ssmoe/prompts/cluster_prompts.json \
  --inject_cluster_prompt True \
  --model_name_or_path ./checkpoints/llava-v1.5-13b \
  --version v1 \
  --freeze_backbone True \
  --data_path ./data/converted/meme_llava.json \
  --image_folder ./data/images \
  --vision_tower ./checkpoints/clip-vit-large-patch14-336 \
  --pretrain_mm_mlp_adapter ./checkpoints/llava-v1.5-13b/mm_projector.bin \
  --mm_projector_type mlp2x_gelu --mm_vision_select_layer -2 \
  --mm_use_im_start_end False --mm_use_im_patch_token False \
  --image_aspect_ratio resize --group_by_modality_length True \
  --bf16 True --output_dir ./checkpoints/llava-ssmoe-13b \
  --num_train_epochs 1 --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 2 --evaluation_strategy "no" \
  --save_strategy steps --save_steps 1000 --save_total_limit 1 \
  --learning_rate 2e-4 --weight_decay 0. --warmup_ratio 0.03 \
  --lr_scheduler_type cosine --logging_steps 1 \
  --tf32 False --model_max_length 2048 \
  --gradient_checkpointing True --dataloader_num_workers 2 \
  --lazy_preprocess True
```
## 🚀 Evaluation
```
python -m ssmoe.eval_meme \
  --model-base ./checkpoints/llava-v1.5-13b \
  --model-path ./checkpoints/llava-ssmoe-13b \
  --data ./data/converted/meme.jsonl \
  --image-folder ./data/images \
  --assignments ./out_ssmoe/cluster/assignments.jsonl \
  --cluster-prompts ./out_ssmoe/prompts/cluster_prompts.json \
  --out ./out_ssmoe/preds.jsonl --batch-size 4 --temperature 0.0
```
## 🏆 Acknowledgements
Our project is built upon [HydraLoRA](https://github.com/Clin0212/HydraLoRA) and [LLaVA](https://github.com/haotian-liu/LLaVA). We are deeply grateful for the excellent codebase they provide. Additionally, we express our appreciation to [Mod_HATE](https://github.com/Social-AI-Studio/Mod_HATE) for their meticulously processed datasets. Their contributions have been of immeasurable value in shaping our work.

## 📄 Citation
We will update the relevant citation information after the paper is formally published.

