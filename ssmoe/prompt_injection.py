import os
import json
import random
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import requests


SYSTEM_PROMPT = (
    "You are an expert in multimodal content analysis. I will provide\n"
    "you with 5 representative meme samples from a cluster of similar memes.\n"
    "Each sample contains an image description and associated text. Your task\n"
    "is to generate a concise analytical prompt that would help a model identify\n"
    "potential harmful patterns common in this cluster.\n"
    "Input: [Samples with image descriptions and text]\n"
    "Based on these samples, generate a prompt (50-100 words) that:\n"
    "1. Identifies key visual-textual patterns to examine\n"
    "2. Highlights potential harmful elements or rhetorical strategies\n"
    "3. Notes cultural or contextual factors that may be relevant\n"
    "4. Avoids being overly prescriptive or biased\n"
    "Note: Focus on analytical guidance rather than predetermined conclusions."
)


@dataclass
class PromptGenConfig:
    provider: str = "openai"  # 'openai' or 'ollama'
    openai_model: str = "gpt-4o"
    openai_api_key: Optional[str] = None  # fallback to env OPENAI_API_KEY
    openai_base_url: Optional[str] = None  # custom base if needed
    ollama_model: str = "gemma3:27b"
    ollama_base_url: str = "http://localhost:11434"
    k_per_cluster: int = 5
    seed: int = 42
    temperature: float = 0.3
    max_tokens: int = 300


def _load_json_or_jsonl(path: str) -> List[Dict]:
    if path.endswith('.jsonl'):
        records = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records
    else:
        with open(path, 'r', encoding='utf-8') as f:
            obj = json.load(f)
        if isinstance(obj, list):
            return obj
        elif isinstance(obj, dict):
            out = []
            for k, v in obj.items():
                if isinstance(v, dict):
                    rec = dict(v)
                    rec['image'] = k
                else:
                    rec = {'image': k, 'cluster_id': v}
                out.append(rec)
            return out
        else:
            raise ValueError("Unsupported JSON structure for input file")


def _build_samples_block(samples: List[Dict]) -> str:
    blocks = []
    for i, s in enumerate(samples, 1):
        desc = s.get('description') or s.get('image_desc') or s.get('caption') or ''
        if not desc:
            desc = f"Image of {os.path.basename(str(s.get('image',''))) or 'unknown'}"
        text = s.get('text') or s.get('ocr') or s.get('meme_text') or ''
        block = f"Sample {i}:\nImage Description: {desc}\nText: {text}"
        blocks.append(block)
    return "\n\n".join(blocks)


def _call_openai(system_prompt: str, user_prompt: str, cfg: PromptGenConfig) -> str:
        from openai import OpenAI
        api_key = cfg.openai_api_key or os.environ.get('OPENAI_API_KEY')
        client = OpenAI(api_key=api_key, base_url=cfg.openai_base_url)
        resp = client.chat.completions.create(
            model=cfg.openai_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
        )
        return resp.choices[0].message.content.strip()


def _call_ollama(system_prompt: str, user_prompt: str, cfg: PromptGenConfig) -> str:
    url = f"{cfg.ollama_base_url.rstrip('/')}/api/generate"
    prompt = f"<system>\n{system_prompt}\n</system>\n{user_prompt}"
    data = {
        "model": cfg.ollama_model,
        "prompt": prompt,
        "options": {"temperature": cfg.temperature},
        "stream": False,
    }
    r = requests.post(url, json=data, timeout=300)
    r.raise_for_status()
    obj = r.json()
    return obj.get('response', '').strip()


def generate_cluster_prompts(
    data_path: str,
    assignments_path: str,
    out_dir: str,
    cfg: PromptGenConfig,
) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, 'cluster_prompts.json')
    existing: Dict[str, str] = {}
    if os.path.exists(out_file):
        with open(out_file, 'r', encoding='utf-8') as f:
            existing = json.load(f)

    data = _load_json_or_jsonl(data_path)
    assigns = _load_json_or_jsonl(assignments_path)

    img2cid: Dict[str, int] = {}
    for a in assigns:
        img = a.get('image')
        cid = a.get('cluster_id')
        if img is not None and cid is not None:
            img2cid[img] = int(cid)

    clusters: Dict[int, List[Dict]] = {}
    for rec in data:
        img = rec.get('image')
        if img in img2cid:
            cid = img2cid[img]
            clusters.setdefault(cid, []).append(rec)

    rng = random.Random(cfg.seed)
    results: Dict[str, str] = dict(existing)
    convo_log: Dict[str, Dict] = {}

    for cid, items in clusters.items():
        key = str(cid)
        if key in results and results[key]:
            continue  # cached
        if not items:
            continue

        reps = items if len(items) <= cfg.k_per_cluster else rng.sample(items, cfg.k_per_cluster)
        user_block = _build_samples_block(reps)

        if cfg.provider == 'openai':
            content = _call_openai(SYSTEM_PROMPT, user_block, cfg)
        elif cfg.provider == 'ollama':
            content = _call_ollama(SYSTEM_PROMPT, user_block, cfg)
        else:
            raise ValueError(f"Unknown provider: {cfg.provider}")

        results[key] = content
        convo_log[key] = {
            "system": SYSTEM_PROMPT,
            "user": user_block,
            "response": content,
        }

        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        with open(os.path.join(out_dir, 'cluster_prompt_conversations.json'), 'w', encoding='utf-8') as f:
            json.dump(convo_log, f, ensure_ascii=False, indent=2)

    return results


def _cli():
    import argparse
    parser = argparse.ArgumentParser(description='Generate cluster-targeted analytical prompts via API (OpenAI or Ollama).')
    parser.add_argument('data', type=str, help='Path to dataset JSON or JSONL with fields: image, text, (optional) description')
    parser.add_argument('assignments', type=str, help='Path to cluster assignments JSON/JSONL with fields: image, cluster_id')
    parser.add_argument('out_dir', type=str, help='Output directory for prompts')
    parser.add_argument('--provider', type=str, default='openai', choices=['openai', 'ollama'])
    parser.add_argument('--openai_model', type=str, default='gpt-4o')
    parser.add_argument('--openai_base_url', type=str, default=None)
    parser.add_argument('--ollama_model', type=str, default='gemma3:27b')
    parser.add_argument('--ollama_base_url', type=str, default='http://localhost:11434')
    parser.add_argument('--k', type=int, default=5, help='Representative samples per cluster')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--temperature', type=float, default=0.3)
    parser.add_argument('--max_tokens', type=int, default=300)
    args = parser.parse_args()

    cfg = PromptGenConfig(
        provider=args.provider,
        openai_model=args.openai_model,
        openai_base_url=args.openai_base_url,
        ollama_model=args.ollama_model,
        ollama_base_url=args.ollama_base_url,
        k_per_cluster=args.k,
        seed=args.seed,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    generate_cluster_prompts(
        data_path=args.data,
        assignments_path=args.assignments,
        out_dir=args.out_dir,
        cfg=cfg,
    )


if __name__ == '__main__':
    _cli()

