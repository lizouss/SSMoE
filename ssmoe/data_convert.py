import os
import json
import csv
from typing import List, Dict, Optional

from llava.constants import DEFAULT_IMAGE_TOKEN


def read_any(path: str, image_col: str, text_col: str, label_col: Optional[str] = None, desc_col: Optional[str] = None) -> List[Dict]:
    records: List[Dict] = []
    if path.endswith('.jsonl'):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                records.append(obj)
    elif path.endswith('.json'):
        with open(path, 'r', encoding='utf-8') as f:
            obj = json.load(f)
        if isinstance(obj, list):
            records = obj
        elif isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, dict):
                    v = dict(v)
                    v.setdefault(image_col, k)
                    records.append(v)
        else:
            raise ValueError('Unsupported JSON format')
    elif path.endswith('.csv'):
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(row)
    else:
        raise ValueError('Unsupported input format, expected .jsonl/.json/.csv')

    out: List[Dict] = []
    for r in records:
        img = r.get(image_col)
        txt = r.get(text_col, '')
        lab = r.get(label_col) if label_col else None
        desc = r.get(desc_col) if desc_col else None
        out.append({
            'image': img,
            'text': txt,
            'label': lab,
            'description': desc,
        })
    return out


def norm_label(v, pos_values: Optional[List[str]] = None, neg_values: Optional[List[str]] = None) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip().lower()
    if pos_values and s in pos_values:
        return 'Harmful'
    if neg_values and s in neg_values:
        return 'Harmless'
    # defaults
    if s in ('1', 'harm', 'harmful', 'toxic', 'offensive'):
        return 'Harmful'
    if s in ('0', 'harmless', 'benign', 'safe', 'non-harmful', 'nonharmful'):
        return 'Harmless'
    return None


def to_jsonl(records: List[Dict], out_path: str, image_root: Optional[str] = None):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        for r in records:
            rec = dict(r)
            if image_root and rec.get('image') and os.path.isabs(rec['image']):
                try:
                    rec['image'] = os.path.relpath(rec['image'], image_root)
                except Exception:
                    pass
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')


def to_llava_json(records: List[Dict], out_path: str, image_root: Optional[str] = None):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    output = []
    for r in records:
        img = r.get('image')
        txt = r.get('text') or ''
        lab = r.get('label')
        label_norm = norm_label(lab)
        if not img:
            continue
        if image_root and os.path.isabs(img):
            try:
                img = os.path.relpath(img, image_root)
            except Exception:
                pass
        user_prompt = f"{DEFAULT_IMAGE_TOKEN}\nClassify the following meme as Harmful or Harmless. Reply with a single word (Harmful/Harmless). Meme text: {txt}"
        conv = [
            {"from": "human", "value": user_prompt},
            {"from": "gpt", "value": (label_norm if label_norm else "Harmless")},
        ]
        output.append({"image": img, "conversations": conv})
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
