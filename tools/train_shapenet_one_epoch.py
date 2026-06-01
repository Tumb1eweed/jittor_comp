import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("nvcc_path", "")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import jittor as jt
from jittor import nn
from tqdm.auto import tqdm

from models.InfoCD import calc_cd_like_InfoV2
from models.pgd import PGDModel


def read_split(path):
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def load_obj_mesh(path):
    vertices = []
    faces = []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                idx = []
                for token in line.split()[1:]:
                    raw = token.split("/")[0]
                    if not raw:
                        continue
                    i = int(raw)
                    idx.append(i - 1 if i > 0 else len(vertices) + i)
                for j in range(1, len(idx) - 1):
                    faces.append([idx[0], idx[j], idx[j + 1]])
    if not vertices or not faces:
        raise ValueError("empty mesh: {}".format(path))
    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int64)


def sample_mesh(vertices, faces, num_points):
    tri = vertices[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    areas = np.linalg.norm(cross, axis=1)
    total = float(areas.sum())
    if total <= 0:
        raise ValueError("mesh has no positive-area faces")
    prob = areas / total
    face_idx = np.random.choice(len(faces), size=num_points, replace=True, p=prob)
    chosen = tri[face_idx]
    u = np.random.rand(num_points, 1).astype(np.float32)
    v = np.random.rand(num_points, 1).astype(np.float32)
    mask = (u + v) > 1.0
    u[mask] = 1.0 - u[mask]
    v[mask] = 1.0 - v[mask]
    pts = chosen[:, 0] + u * (chosen[:, 1] - chosen[:, 0]) + v * (chosen[:, 2] - chosen[:, 0])
    return pts.astype(np.float32)


def normalize_unit_sphere(points):
    p_max = points.max(axis=0, keepdims=True)
    p_min = points.min(axis=0, keepdims=True)
    center = (p_max + p_min) * 0.5
    points = points - center
    scale = np.sqrt((points * points).sum(axis=1)).max()
    if scale <= 0:
        scale = 1.0
    return (points / scale).astype(np.float32)


def make_patch(clean, noisy, patch_size):
    n = noisy.shape[0]
    seed_idx = np.random.randint(0, n)
    seed = noisy[seed_idx:seed_idx + 1]
    dist = ((noisy - seed) ** 2).sum(axis=1)
    if patch_size < n:
        idx = np.argpartition(dist, patch_size)[:patch_size]
    else:
        idx = np.arange(n)
        if patch_size > n:
            pad = np.random.choice(n, size=patch_size - n, replace=True)
            idx = np.concatenate([idx, pad])
    seed_t = noisy[seed_idx]
    return noisy[idx] - seed_t, clean[idx] - seed_t


class ShapeNetPatchDataset:
    def __init__(
        self,
        dataset_root,
        split_file,
        data_name,
        sample_points,
        patch_size,
        noise_std,
        noise_std_min=None,
        noise_std_max=None,
        max_shapes=0,
    ):
        self.dataset_root = Path(dataset_root)
        self.items = read_split(split_file)
        if max_shapes > 0:
            self.items = self.items[:max_shapes]
        self.data_name = data_name
        self.sample_points = sample_points
        self.patch_size = patch_size
        self.noise_std = noise_std
        self.noise_std_min = noise_std_min
        self.noise_std_max = noise_std_max

    def sample_noise_std(self):
        if self.noise_std_min is None or self.noise_std_max is None:
            return self.noise_std
        return float(np.random.uniform(self.noise_std_min, self.noise_std_max))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        rel = self.items[idx]
        path = self.dataset_root / rel / self.data_name
        vertices, faces = load_obj_mesh(path)
        clean = normalize_unit_sphere(sample_mesh(vertices, faces, self.sample_points))
        noise_std = self.sample_noise_std()
        noisy = clean + np.random.normal(0.0, noise_std, size=clean.shape).astype(np.float32)
        patch_noisy, patch_clean = make_patch(clean, noisy, self.patch_size)
        return patch_noisy.astype(np.float32), patch_clean.astype(np.float32), str(path)


def chamfer_loss(pred, clean):
    dist = jt.sum((pred[:, :, None, :] - clean[:, None, :, :]) ** 2, dim=-1)
    d1 = dist.min(dim=2)
    d2 = dist.min(dim=1)
    return d1.mean() + d2.mean()


def get_loss_fn(name):
    if name == "infocd":
        return calc_cd_like_InfoV2
    if name == "chamfer":
        return chamfer_loss
    raise ValueError("unsupported loss: {}".format(name))


def stack_batch(samples):
    noisy = jt.array(np.stack([s[0] for s in samples], axis=0))
    clean = jt.array(np.stack([s[1] for s in samples], axis=0))
    return noisy, clean


def run_epoch(model, dataset, optimizer, batch_size, loss_fn, train=True, desc="train"):
    indices = list(range(len(dataset)))
    if train:
        random.shuffle(indices)
        model.train()
    else:
        model.eval()
    losses = []
    total = int(math.ceil(len(indices) / float(batch_size)))
    pbar = tqdm(range(total), desc=desc)
    for step in pbar:
        batch_idx = indices[step * batch_size:(step + 1) * batch_size]
        samples = [dataset[i] for i in batch_idx]
        noisy, clean = stack_batch(samples)
        pred = noisy + model(noisy)
        loss = loss_fn(pred, clean)
        if train:
            optimizer.step(loss)
        val = float(loss.numpy())
        losses.append(val)
        pbar.set_postfix(loss="{:.6f}".format(val))
    return float(np.mean(losses)) if losses else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", default="/home/dataset_train")
    parser.add_argument("--datalist_dir", default="/home/PGD/datalist")
    parser.add_argument("--data_name", default="models/model_normalized.obj")
    parser.add_argument("--sample_points", type=int, default=10000)
    parser.add_argument("--noise_std", type=float, default=0.025)
    parser.add_argument("--noise_std_min", type=float, default=None)
    parser.add_argument("--noise_std_max", type=float, default=None)
    parser.add_argument("--patch_size", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--loss", choices=["infocd", "chamfer"], default="infocd")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--use_cuda", action="store_true")
    parser.add_argument("--max_train_shapes", type=int, default=0)
    parser.add_argument("--max_val_shapes", type=int, default=0)
    parser.add_argument("--log_dir", default="/home/PGD/experiments/shapenet_10k_gaussian_025_one_epoch")
    args = parser.parse_args()

    if args.use_cuda:
        jt.flags.use_cuda = 1
    random.seed(args.seed)
    np.random.seed(args.seed)
    jt.set_global_seed(args.seed)

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    train_set = ShapeNetPatchDataset(
        args.dataset_root,
        Path(args.datalist_dir) / "train.txt",
        args.data_name,
        args.sample_points,
        args.patch_size,
        args.noise_std,
        noise_std_min=args.noise_std_min,
        noise_std_max=args.noise_std_max,
        max_shapes=args.max_train_shapes,
    )
    val_set = ShapeNetPatchDataset(
        args.dataset_root,
        Path(args.datalist_dir) / "validate.txt",
        args.data_name,
        args.sample_points,
        args.patch_size,
        args.noise_std,
        noise_std_min=args.noise_std_min,
        noise_std_max=args.noise_std_max,
        max_shapes=args.max_val_shapes,
    )
    print("train shapes: {}, val shapes: {}".format(len(train_set), len(val_set)), flush=True)
    if args.noise_std_min is not None and args.noise_std_max is not None:
        print(
            "sample_points: {}, gaussian noise std range: [{}, {}]".format(
                args.sample_points,
                args.noise_std_min,
                args.noise_std_max,
            ),
            flush=True,
        )
    else:
        print("sample_points: {}, gaussian noise std: {}".format(args.sample_points, args.noise_std), flush=True)
    print("loss: {}".format(args.loss), flush=True)

    model = PGDModel(args)
    optimizer = nn.Adam(model.parameters(), lr=args.lr)
    loss_fn = get_loss_fn(args.loss)

    history = []
    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss = run_epoch(model, train_set, optimizer, args.batch_size, loss_fn, train=True, desc="train epoch {:02d}".format(epoch))
        val_loss = run_epoch(model, val_set, optimizer, args.batch_size, loss_fn, train=False, desc="val epoch {:02d}".format(epoch))
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "seconds": time.time() - t0,
        }
        history.append(row)
        with open(log_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)
        ckpt = log_dir / "pgd-shapenet-epoch{:02d}-loss{:.8f}.npz".format(epoch, train_loss)
        model.save_npz(ckpt)
        print("[Epoch {:02d}] train_loss={:.8f} val_loss={:.8f} ckpt={}".format(epoch, train_loss, val_loss, ckpt), flush=True)


if __name__ == "__main__":
    main()
