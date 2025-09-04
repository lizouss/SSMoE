import os
import json
import argparse
from typing import List, Dict, Optional

import torch
from PIL import Image

from builder import load_pretrained_model
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.mm_utils import tokenizer_image_token


def load_jsonl(path: str) -> List[Dict]:
    items = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def load_json_any(path: str) -> List[Dict]:
    if path.endswith('.jsonl'):
        return load_jsonl(path)
    with open(path, 'r', encoding='utf-8') as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return obj
    elif isinstance(obj, dict):
        return [{"image": k, **(v if isinstance(v, dict) else {"value": v})} for k, v in obj.items()]
    else:
        raise ValueError("Unsupported JSON format")


def build_img(image_folder: str, rel_path: str):
    img = Image.open(os.path.join(image_folder, rel_path)).convert('RGB')
    return img


def parse_label(text: str) -> str:
    s = text.strip().lower()
    if 'harmful' in s and 'harmless' not in s:
        return 'Harmful'
    if 'harmless' in s and 'harmful' not in s:
        return 'Harmless'
    negatives = ['not harmful', 'non-harmful', 'benign', 'safe']
    positives = ['hateful', 'offensive', 'derogatory', 'toxic']
    if any(k in s for k in positives):
        return 'Harmful'
    if any(k in s for k in negatives):
        return 'Harmless'
    return 'Harmless'


def compute_metrics(records: List[Dict], label_key: str = 'label'):
    y_true = []
    y_pred = []
    for r in records:
        if label_key in r:
            y_true.append(r[label_key])
            y_pred.append(r.get('pred_label', 'Harmless'))
    if not y_true:
        return {}
    def to01(x):
        return 1 if str(x).lower().startswith('harm') else 0
    yt = [to01(x) for x in y_true]
    yp = [to01(x) for x in y_pred]
    tp = sum(1 for a, b in zip(yt, yp) if a == 1 and b == 1)
    tn = sum(1 for a, b in zip(yt, yp) if a == 0 and b == 0)
    fp = sum(1 for a, b in zip(yt, yp) if a == 0 and b == 1)
    fn = sum(1 for a, b in zip(yt, yp) if a == 1 and b == 0)
    acc = (tp + tn) / max(1, len(yt))
    def f1(p, r):
        return 0.0 if (p + r) == 0 else 2 * p * r / (p + r)
    p_pos = 0.0 if (tp + fp) == 0 else tp / (tp + fp)
    r_pos = 0.0 if (tp + fn) == 0 else tp / (tp + fn)
    f1_pos = f1(p_pos, r_pos)
    p_neg = 0.0 if (tn + fn) == 0 else tn / (tn + fn)
    r_neg = 0.0 if (tn + fp) == 0 else tn / (tn + fp)
    f1_neg = f1(p_neg, r_neg)
    macro_f1 = (f1_pos + f1_neg) / 2.0
    return {"accuracy": acc, "macro_f1": macro_f1}


def main():
    ap = argparse.ArgumentParser(description='Evaluate')
    ap.add_argument('--model-base', type=str, required=True)
    ap.add_argument('--model-path', type=str, required=True)
    ap.add_argument('--data', type=str, required=True)
    ap.add_argument('--image-folder', type=str, required=True)
    ap.add_argument('--assignments', type=str, required=True)
    ap.add_argument('--cluster-prompts', type=str, required=True)
    ap.add_argument('--conv-mode', type=str, default='llava_v1')
    ap.add_argument('--temperature', type=float, default=0.0)
    ap.add_argument('--top-p', type=float, default=1.0)
    ap.add_argument('--max-new-tokens', type=int, default=64)
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--out', type=str, required=True)
    args = ap.parse_args()

    tokenizer, model, image_processor, _ = load_pretrained_model(args.model_path, model_base=args.model_base, model_name='llava-ssmoe', load_8bit=False, load_4bit=False)
    model.eval()

    data = load_json_any(args.data)
    assigns = load_json_any(args.assignments)
    with open(args.cluster_prompts, 'r', encoding='utf-8') as f:
        prompts_map = json.load(f)
    img2cid = {a['image']: int(a['cluster_id']) for a in assigns if 'image' in a and 'cluster_id' in a}

    conv_template = conv_templates[args.conv_mode].copy()

    records_out = []
    for i in range(0, len(data), args.batch_size):
        batch = data[i:i+args.batch_size]
        images = []
        input_ids_list = []
        cluster_ids = []
        convs = []
        for rec in batch:
            img = build_img(args.image_folder, rec['image'])
            images.append(image_processor.preprocess(img, return_tensors='pt')['pixel_values'][0])

            conv = conv_template.copy()
            cid = img2cid.get(rec['image'], -1)
            cluster_ids.append(cid)
            guidance = prompts_map.get(str(cid), '') if cid >= 0 else ''

            user_text = rec.get('text', '')
            if guidance:
                user_text = f"{guidance}\n\n" + user_text

            inp = DEFAULT_IMAGE_TOKEN + '\n' + user_text
            conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
            input_ids_list.append(input_ids)
            convs.append(conv)

        batch_images = torch.stack(images).to(model.device, dtype=torch.float16)
        batch_input_ids = torch.nn.utils.rnn.pad_sequence(input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id).to(model.device)
        cluster_ids_t = torch.tensor(cluster_ids, dtype=torch.long, device=model.device)

        with torch.inference_mode():
            output_ids = model.generate(
                inputs=batch_input_ids,
                images=batch_images,
                do_sample=args.temperature > 0,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                cluster_ids=cluster_ids_t,
            )
        for j, out in enumerate(output_ids):
            conv = convs[j]
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            text = tokenizer.decode(out.tolist(), skip_special_tokens=True)
            if stop_str in text:
                text = text.split(stop_str)[-1].strip()
            pred = parse_label(text)
            rec_out = dict(batch[j])
            rec_out['cluster_id'] = cluster_ids[j]
            rec_out['prediction'] = text
            rec_out['pred_label'] = pred
            records_out.append(rec_out)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        for r in records_out:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    metrics = compute_metrics(records_out)
    if metrics:
        print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()

