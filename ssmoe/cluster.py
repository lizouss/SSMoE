import os
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple

import torch
from torch import nn
from PIL import Image

from transformers import AutoImageProcessor, AutoModel
from sentence_transformers import SentenceTransformer


@dataclass
class ClusterConfig:
    text_model_name: str = "nomic-ai/nomic-embed-text-v1.5"
    vision_model_name: str = "nomic-ai/nomic-embed-vision-v1.5"
    device: str = "cuda"
    batch_size: int = 16
    n_clusters: int = 8
    feature_type: str = "concat"
    max_iter: int = 100
    n_init: int = 8
    seed: int = 42


class _TorchKMeans:
    def __init__(self, n_clusters: int, max_iter: int = 100, n_init: int = 1, tol: float = 1e-4, seed: int = 42, device: str = "cpu"):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
        self.tol = tol
        self.seed = seed
        self.device = device
        self.centroids: Optional[torch.Tensor] = None

    @staticmethod
    def _pairwise_dist_sq(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x: [N, D], y: [K, D]
        # returns [N, K]
        x2 = (x * x).sum(dim=1, keepdim=True)
        y2 = (y * y).sum(dim=1).unsqueeze(0)
        xy = x @ y.t()
        return x2 + y2 - 2 * xy

    def _kmeans_plus_plus_init(self, X: torch.Tensor) -> torch.Tensor:
        N = X.size(0)
        gen = torch.Generator(device=X.device)
        gen.manual_seed(self.seed)
        first_idx = torch.randint(0, N, (1,), generator=gen, device=X.device)
        centers = [X[first_idx.item()]]
        for _ in range(1, self.n_clusters):
            C = torch.stack(centers, dim=0)
            dist_sq = self._pairwise_dist_sq(X, C).min(dim=1).values
            probs = dist_sq / (dist_sq.sum() + 1e-12)
            next_idx = torch.multinomial(probs, 1, generator=gen)
            centers.append(X[next_idx.item()])
        return torch.stack(centers, dim=0)

    def fit(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        X = X.to(self.device)
        best_inertia = float('inf')
        best_centroids = None
        best_labels = None

        for init_round in range(self.n_init):
            centroids = self._kmeans_plus_plus_init(X)
            last_loss = None
            for _ in range(self.max_iter):
                dists = self._pairwise_dist_sq(X, centroids)
                labels = torch.argmin(dists, dim=1)
                new_centroids = torch.zeros_like(centroids)
                for k in range(self.n_clusters):
                    mask = labels == k
                    if mask.any():
                        new_centroids[k] = X[mask].mean(dim=0)
                    else:
                        # using kmeans++ step
                        d2 = self._pairwise_dist_sq(X, centroids).min(dim=1).values
                        probs = d2 / (d2.sum() + 1e-12)
                        new_idx = torch.multinomial(probs, 1)
                        new_centroids[k] = X[new_idx.item()]
                centroids = new_centroids
                inertia = self._pairwise_dist_sq(X, centroids).min(dim=1).values.mean().item()
                if last_loss is not None and abs(last_loss - inertia) < self.tol:
                    break
                last_loss = inertia

            final_inertia = self._pairwise_dist_sq(X, centroids).min(dim=1).values.sum().item()
            if final_inertia < best_inertia:
                best_inertia = final_inertia
                best_centroids = centroids.clone()
                best_labels = torch.argmin(self._pairwise_dist_sq(X, centroids), dim=1).clone()

        self.centroids = best_centroids
        return best_centroids, best_labels

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        assert self.centroids is not None, "KMeans not fitted"
        X = X.to(self.device)
        dists = self._pairwise_dist_sq(X, self.centroids)
        return torch.argmin(dists, dim=1)


class SemanticClusterer:
    def __init__(self, config: ClusterConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() and config.device.startswith("cuda") else "cpu")
        self._text_model: Optional[SentenceTransformer] = None
        self._vision_model: Optional[AutoModel] = None
        self._vision_proc: Optional[AutoImageProcessor] = None
        self._centroids: Optional[torch.Tensor] = None  # [K, D]
        self._feat_dim: Optional[int] = None

    def _lazy_load_models(self):
        if self._text_model is None:
            self._text_model = SentenceTransformer(self.config.text_model_name, device=str(self.device), trust_remote_code=True)
            self._text_model.eval()
        if self._vision_model is None or self._vision_proc is None:
            self._vision_proc = AutoImageProcessor.from_pretrained(self.config.vision_model_name)
            self._vision_model = AutoModel.from_pretrained(self.config.vision_model_name, trust_remote_code=True).to(self.device)
            self._vision_model.eval()

    @staticmethod
    def _load_image(path: str) -> Image.Image:
        img = Image.open(path).convert("RGB")
        return img

    @torch.inference_mode()
    def _encode_batch(self, batch: List[Dict]) -> torch.Tensor:
        images = [self._load_image(x["image"]) for x in batch]
        texts = [x.get("text", "") or "" for x in batch]

        txt_emb_np = self._text_model.encode(texts, batch_size=len(texts), convert_to_numpy=True, normalize_embeddings=True)
        txt_emb = torch.from_numpy(txt_emb_np).to(self.device)

        proc = self._vision_proc(images=images, return_tensors="pt")
        proc = {k: v.to(self.device) for k, v in proc.items()}
        outputs = self._vision_model(**proc)
        img_cls = outputs.last_hidden_state[:, 0]
        img_emb = torch.nn.functional.normalize(img_cls, p=2, dim=1)

        if self.config.feature_type == "image":
            feats = img_emb
        elif self.config.feature_type == "text":
            feats = txt_emb
        else:
            feats = torch.cat([img_emb, txt_emb], dim=-1)
            feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-12)
        return feats

    @torch.inference_mode()
    def _encode_all(self, items: List[Dict]) -> torch.Tensor:
        self._lazy_load_models()
        feats: List[torch.Tensor] = []
        bs = self.config.batch_size
        for i in range(0, len(items), bs):
            batch = items[i:i+bs]
            feats.append(self._encode_batch(batch).detach().to("cpu"))
        X = torch.cat(feats, dim=0)
        return X

    def fit(self, items: List[Dict]) -> Dict[str, torch.Tensor]:
        X = self._encode_all(items)
        self._feat_dim = X.shape[1]
        kmeans = _TorchKMeans(n_clusters=self.config.n_clusters, max_iter=self.config.max_iter,
                               n_init=self.config.n_init, seed=self.config.seed, device="cpu")
        centroids, labels = kmeans.fit(X)
        self._centroids = centroids.to("cpu")
        return {"centroids": self._centroids, "labels": labels}

    @torch.inference_mode()
    def assign(self, items: List[Dict]) -> torch.Tensor:
        assert self._centroids is not None, "Clusterer not fitted or loaded."
        X = self._encode_all(items)
        dists = self._pairwise_dist_sq_cpu(X, self._centroids)
        return torch.argmin(dists, dim=1)

    @staticmethod
    def _pairwise_dist_sq_cpu(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x2 = (x * x).sum(dim=1, keepdim=True)
        y2 = (y * y).sum(dim=1).unsqueeze(0)
        xy = x @ y.t()
        return x2 + y2 - 2 * xy

    def save(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        meta = asdict(self.config).copy()
        meta.update({
            "feat_dim": int(self._feat_dim or -1),
            "centroids_file": "centroids.pt",
        })
        with open(os.path.join(out_dir, "cluster_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        if self._centroids is not None:
            torch.save(self._centroids.cpu(), os.path.join(out_dir, meta["centroids_file"]))

    def load(self, in_dir: str):
        with open(os.path.join(in_dir, "cluster_meta.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.config = ClusterConfig(**{k: meta[k] for k in ClusterConfig.__dataclass_fields__.keys() if k in meta})
        self._feat_dim = int(meta.get("feat_dim", -1))
        centroids_file = meta.get("centroids_file", "centroids.pt")
        self._centroids = torch.load(os.path.join(in_dir, centroids_file), map_location="cpu")


def _read_jsonl(path: str) -> List[Dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            items.append({"image": obj.get("image"), "text": obj.get("text", "")})
    return items


def _write_jsonl(path: str, records: List[Dict]):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _cli():
    import argparse
    parser = argparse.ArgumentParser(description="Semantic clustering for memes")
    parser.add_argument("data_jsonl", type=str)
    parser.add_argument("out_dir", type=str)
    parser.add_argument("--text_model", type=str, default="nomic-ai/nomic-embed-text-v1.5", help="Text embedding model")
    parser.add_argument("--vision_model", type=str, default="nomic-ai/nomic-embed-vision-v1.5", help="Vision embedding model")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--clusters", type=int, default=8)
    parser.add_argument("--feature_type", type=str, default="concat", choices=["image", "text", "concat"])
    parser.add_argument("--max_iter", type=int, default=100)
    parser.add_argument("--n_init", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--assign_only", action="store_true")
    args = parser.parse_args()

    cfg = ClusterConfig(
        text_model_name=args.text_model,
        vision_model_name=args.vision_model,
        device=args.device,
        batch_size=args.batch_size,
        n_clusters=args.clusters,
        feature_type=args.feature_type,
        max_iter=args.max_iter,
        n_init=args.n_init,
        seed=args.seed,
    )

    items = _read_jsonl(args.data_jsonl)
    clus = SemanticClusterer(cfg)
    os.makedirs(args.out_dir, exist_ok=True)

    if args.assign_only:
        clus.load(args.out_dir)
        labels = clus.assign(items).tolist()
        recs = []
        for it, lab in zip(items, labels):
            r = dict(it)
            r["cluster_id"] = int(lab)
            recs.append(r)
        _write_jsonl(os.path.join(args.out_dir, "assignments.jsonl"), recs)
        return

    out = clus.fit(items)
    clus.save(args.out_dir)
    labels = out["labels"].tolist()
    recs = []
    for it, lab in zip(items, labels):
        r = dict(it)
        r["cluster_id"] = int(lab)
        recs.append(r)
    _write_jsonl(os.path.join(args.out_dir, "assignments.jsonl"), recs)


if __name__ == "__main__":
    _cli()
