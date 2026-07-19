import argparse
import json
import math
import os
import random
import re
import subprocess
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
from scipy.spatial import cKDTree
from tqdm.auto import tqdm

from models.InfoCD import calc_cd_like_InfoV2
from models.asdn import ASDNModel
from models.pgd import PGDModel
from utils.noise import DEFAULT_NOISE_TYPES, add_numpy_noise, parse_noise_types, sample_noise_std


def read_split(path):
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def build_category_vocab(datalist_dir, split_names=("train.txt",)):
    items = []
    for name in split_names:
        path = Path(datalist_dir) / name
        if path.exists():
            items.extend(read_split(path))
    categories = sorted({extract_synset_id(item) for item in items})
    return {cat: i + 1 for i, cat in enumerate(categories)}


def extract_synset_id(rel):
    parts = str(rel).split("/")
    if len(parts) >= 2 and parts[0] == "shapenet":
        return parts[1]
    return parts[0]


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


def paired_density_jitter(noisy, clean, drop_ratio):
    """Simulate uneven point density while preserving noisy/clean pairs.

    A point-cloud denoiser receives unordered sets, and test clouds can have
    local density variations even when their noise scale is unknown.  We keep
    a random subset then resample it to the original cardinality; applying the
    same indices to both arrays retains the supervised correspondence and does
    not introduce any validation/test-specific information.
    """
    drop_ratio = float(drop_ratio)
    if drop_ratio <= 0.0 or noisy.shape[0] < 4:
        return noisy, clean
    n = noisy.shape[0]
    keep = max(4, int(round(n * (1.0 - min(drop_ratio, 0.95)))))
    kept = np.random.choice(n, size=keep, replace=False)
    idx = np.random.choice(kept, size=n, replace=True)
    return noisy[idx], clean[idx]


def estimate_patch_normals_np(points, k=16):
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    n = points.shape[0]
    k = int(max(3, min(k, n)))
    # A training launch already runs one process per GPU.  Letting every
    # per-patch normal fit spawn all CPU workers oversubscribes the host and
    # starves all GPUs at once; one worker per training process is faster and
    # reproducible under multi-GPU sweeps.
    _, idx = cKDTree(points).query(points, k=k, workers=1)
    if k == 1:
        idx = idx[:, None]
    neigh = points[idx]
    centered = neigh - neigh.mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", centered, centered) / float(k)
    _, vecs = np.linalg.eigh(cov)
    normals = vecs[:, :, 0]
    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    return (normals / np.maximum(norm, 1e-8)).astype(np.float32)


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
        noise_types=DEFAULT_NOISE_TYPES,
        max_shapes=0,
        precomputed_points_dir="",
        cache_points=False,
        patches_per_shape=1,
        random_z_rotation=False,
        density_jitter_ratio=0.0,
        category_to_id=None,
        return_normals=False,
        normal_k=16,
    ):
        self.dataset_root = Path(dataset_root)
        self.items = read_split(split_file)
        if max_shapes > 0:
            self.items = self.items[:max_shapes]
        if category_to_id is None:
            categories = sorted({item.split("/")[0] for item in self.items})
            category_to_id = {cat: i + 1 for i, cat in enumerate(categories)}
        self.category_to_id = dict(category_to_id)
        self.num_categories = len(self.category_to_id) + 1
        self.data_name = data_name
        self.sample_points = sample_points
        self.patch_size = patch_size
        self.noise_std = noise_std
        self.noise_std_min = noise_std_min
        self.noise_std_max = noise_std_max
        self.noise_types = ",".join(parse_noise_types(noise_types))
        self.precomputed_points_dir = Path(precomputed_points_dir) if precomputed_points_dir else None
        self.cache_points = cache_points
        self.points_cache = {}
        self.patches_per_shape = max(1, int(patches_per_shape))
        self.random_z_rotation = bool(random_z_rotation)
        self.density_jitter_ratio = float(density_jitter_ratio)
        self.return_normals = bool(return_normals)
        self.normal_k = int(normal_k)

    def sample_noise_std(self):
        return sample_noise_std(self.noise_std, self.noise_std_min, self.noise_std_max)

    def __len__(self):
        return len(self.items) * self.patches_per_shape

    def __getitem__(self, idx):
        shape_idx = idx % len(self.items)
        rel = self.items[shape_idx]
        if self.precomputed_points_dir is not None:
            path = self.precomputed_points_dir / "{}.npy".format(rel)
            if self.cache_points and rel in self.points_cache:
                clean = self.points_cache[rel]
            else:
                clean = np.load(path).astype(np.float32)
                if self.cache_points:
                    self.points_cache[rel] = clean
        else:
            path = self.dataset_root / rel / self.data_name
            vertices, faces = load_obj_mesh(path)
            clean = normalize_unit_sphere(sample_mesh(vertices, faces, self.sample_points))
        noise_std = self.sample_noise_std()
        noisy, _ = add_numpy_noise(clean, noise_std, self.noise_types)
        patch_noisy, patch_clean = make_patch(clean, noisy, self.patch_size)
        patch_noisy, patch_clean = paired_density_jitter(
            patch_noisy, patch_clean, self.density_jitter_ratio
        )
        if self.random_z_rotation:
            # ShapeNet objects share an upright axis, while the denoising task
            # itself is invariant to yaw.  Applying the same random yaw to
            # noisy and clean patches teaches that invariance without using
            # any validation/test information or changing inference.
            theta = np.random.uniform(0.0, 2.0 * np.pi)
            c, s = np.cos(theta), np.sin(theta)
            rot = np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
            patch_noisy = np.matmul(patch_noisy, rot)
            patch_clean = np.matmul(patch_clean, rot)
        category_id = self.category_to_id.get(extract_synset_id(rel), 0)
        sample = [patch_noisy.astype(np.float32), patch_clean.astype(np.float32), str(path), float(noise_std), int(category_id)]
        if self.return_normals:
            sample.append(estimate_patch_normals_np(patch_clean, k=self.normal_k))
        return tuple(sample)


def chamfer_loss(pred, clean):
    d1, d2 = directional_chamfer_terms(pred, clean)
    return d1 + d2


def directional_chamfer_terms(pred, clean):
    """Return squared nearest-neighbor CD terms in scorer order.

    ``pred_to_clean`` measures whether every predicted point lands near the
    target surface/sample, while ``clean_to_pred`` measures coverage.  Keeping
    them separate lets training emphasize the direction that is weak in the
    final evaluator without changing the default symmetric Chamfer loss.
    """
    dist = jt.sum((pred[:, :, None, :] - clean[:, None, :, :]) ** 2, dim=-1)
    d1 = dist.min(dim=2)
    d2 = dist.min(dim=1)
    return d1.mean(), d2.mean()


def per_sample_chamfer_terms(pred, clean):
    """Return scorer-order squared-CD terms for every item in a batch.

    The competition averages each sample's *relative* CD improvement, rather
    than taking a ratio of batch-aggregated CDs.  Keeping this helper separate
    from :func:`directional_chamfer_terms` preserves the established composite
    objective while allowing an explicitly score-aligned, train-only term.
    """
    dist = jt.sum((pred[:, :, None, :] - clean[:, None, :, :]) ** 2, dim=-1)
    pred_to_clean = dist.min(dim=2).mean(dim=1)
    clean_to_pred = dist.min(dim=1).mean(dim=1)
    return pred_to_clean, clean_to_pred


def score_aligned_relative_cd_loss(pred, clean, noisy, eps=1e-8):
    """Mean per-sample CD ratio, matching the evaluator's CD aggregation.

    This uses only paired noisy/clean training patches.  It neither consumes
    validation/test data nor requires a noise estimate at inference.  Unlike
    the legacy batch-level relative term, each training sample contributes
    equally after normalization by its own noisy CD, as it does in the final
    scorer.
    """
    pred_a, pred_b = per_sample_chamfer_terms(pred, clean)
    noisy_a, noisy_b = per_sample_chamfer_terms(noisy, clean)
    return score_aligned_relative_cd_from_terms(pred_a, pred_b, noisy_a, noisy_b, eps=eps)


def score_aligned_relative_cd_from_terms(pred_a, pred_b, noisy_a, noisy_b, eps=1e-8):
    """Score-aligned ratio from precomputed directional CD terms."""
    return ((pred_a + pred_b) / ((noisy_a + noisy_b).detach() + eps)).mean()


def huber_loss(x, delta=0.01):
    abs_x = jt.abs(x)
    quad = jt.minimum(abs_x, delta)
    linear = abs_x - quad
    return (0.5 * quad * quad / delta + linear).mean()


def normal_tangent_corr_loss(pred, clean, normals, normal_weight=2.0, tangent_weight=1.0):
    normals = normals / (jt.norm(normals, dim=-1, keepdims=True) + 1e-8)
    error = pred - clean
    normal_error = jt.sum(error * normals, dim=-1, keepdims=True) * normals
    tangent_error = error - normal_error
    # Normal error approximates point-to-surface degradation, while tangent
    # error is needed to improve point placement and hence Chamfer distance.
    # Keep their trade-off explicit so experiments can target CD without
    # silently sacrificing P2S.  These terms are training-only and require no
    # metadata at inference.
    return jt.mean(
        float(normal_weight) * jt.sum(normal_error * normal_error, dim=-1)
        + float(tangent_weight) * jt.sum(tangent_error * tangent_error, dim=-1)
    )


def straight_displacement_loss(pred, noisy, clean, direction_weight=1.0, distance_weight=1.0, eps=1e-8):
    pred_disp = pred - noisy
    target_disp = clean - noisy
    pred_mag = jt.sqrt(jt.maximum(jt.sum(pred_disp * pred_disp, dim=-1), eps))
    target_mag = jt.sqrt(jt.maximum(jt.sum(target_disp * target_disp, dim=-1), eps))
    pred_dir = pred_disp / (pred_mag.unsqueeze(-1) + eps)
    target_dir = target_disp / (target_mag.unsqueeze(-1) + eps)
    valid = (target_mag > eps).float32()
    cosine = jt.sum(pred_dir * target_dir, dim=-1)
    direction = jt.sum((1.0 - cosine) * valid) / (jt.sum(valid) + eps)
    distance = huber_loss(pred_mag - target_mag, delta=0.01)
    total = float(direction_weight) * direction + float(distance_weight) * distance
    return total, direction, distance


def uniform_loss(points):
    dist = jt.sum((points[:, :, None, :] - points[:, None, :, :]) ** 2, dim=-1)
    eye = jt.array(np.eye(points.shape[1], dtype=np.float32)).unsqueeze(0) * 1e6
    nn_dist = jt.sqrt(jt.maximum((dist + eye).min(dim=2), 1e-9))
    mean = nn_dist.mean(dim=1, keepdims=True)
    return jt.mean((nn_dist - mean) ** 2 / (mean.detach() ** 2 + 1e-8))


def density_consistency_loss(pred, clean):
    """Keep local point spacing consistent with the clean training surface.

    This is complementary to a uniformity penalty: uniformity alone can make
    every patch equally spaced, whereas ShapeNet surfaces legitimately have
    density changes from mesh sampling.  Matching nearest-neighbour spacing
    preserves those changes while discouraging collapsed or duplicated output
    points that hurt the clean-to-pred Chamfer term.  It is train-only and
    requires no normal, category, or noise-level label at inference.
    """
    n = pred.shape[1]
    eye = jt.array(np.eye(n, dtype=np.float32)).unsqueeze(0) * 1e6
    pred_dist = jt.sum((pred[:, :, None, :] - pred[:, None, :, :]) ** 2, dim=-1)
    clean_dist = jt.sum((clean[:, :, None, :] - clean[:, None, :, :]) ** 2, dim=-1)
    pred_nn = jt.sqrt(jt.maximum((pred_dist + eye).min(dim=2), 1e-9))
    clean_nn = jt.sqrt(jt.maximum((clean_dist + eye).min(dim=2), 1e-9))
    # Point spacings are much smaller than coordinate-scale errors.  Use a
    # relative error so this term has a meaningful, scale-invariant gradient
    # across patches instead of being numerically drowned out by CD terms.
    relative_error = (pred_nn - clean_nn) / (clean_nn.detach() + 1e-4)
    return huber_loss(relative_error, delta=0.25)


def local_surface_distance_loss(pred, clean, max_points=128):
    """Match local clean-surface geometry without inference-time metadata.

    Point-wise supervision is useful for removing the sampled Gaussian noise,
    but it does not explicitly penalize a denoised patch becoming locally too
    stretched or compressed.  This term compares pairwise distances on a
    random subset and softly concentrates on each point's neighbourhood.  The
    scale is computed per patch, making it agnostic to ShapeNet object size
    and sampling density.  It is used only during training.
    """
    m = min(int(max_points), int(pred.shape[1]))
    pred = pred[:, :m, :]
    clean = clean[:, :m, :]
    clean_sq = jt.sum((clean[:, :, None, :] - clean[:, None, :, :]) ** 2, dim=-1)
    pred_sq = jt.sum((pred[:, :, None, :] - pred[:, None, :, :]) ** 2, dim=-1)
    clean_dist = jt.sqrt(jt.maximum(clean_sq, 1e-12))
    pred_dist = jt.sqrt(jt.maximum(pred_sq, 1e-12))
    # Mean pair distance provides a patch-scale normalizer.  A soft radius
    # keeps the objective focused on local surface structure while retaining
    # gradients for more distant pairs.
    scale = clean_dist.mean(dim=2, keepdims=True).mean(dim=1, keepdims=True).detach() + 1e-6
    weights = jt.exp(-clean_dist / (0.12 * scale))
    eye = jt.array(np.eye(m, dtype=np.float32)).unsqueeze(0)
    weights = weights * (1.0 - eye)
    relative_error = (pred_dist - clean_dist) / scale
    abs_error = jt.abs(relative_error)
    delta = 0.05
    per_pair = jt.minimum(abs_error, delta)
    per_pair = 0.5 * per_pair * per_pair / delta + (abs_error - per_pair)
    return jt.sum(weights * per_pair) / (jt.sum(weights) + 1e-8)


def asdn_training_loss(model, noisy, clean, noise_std, category_id, args):
    out = model(noisy, noise_std=noise_std, category_id=category_id, return_dict=True)
    pred = out["final"]
    corr = huber_loss(pred - clean, delta=args.corr_huber_delta)
    pred_cd, clean_cd = directional_chamfer_terms(pred, clean)
    baseline_pred_cd, baseline_clean_cd = directional_chamfer_terms(noisy, clean)
    baseline = (baseline_pred_cd + baseline_clean_cd).detach()
    relative = (pred_cd + clean_cd) / (baseline + args.relative_eps)
    pred_cd_relative = pred_cd / (baseline_pred_cd.detach() + args.relative_eps)
    clean_cd_relative = clean_cd / (baseline_clean_cd.detach() + args.relative_eps)
    info = calc_cd_like_InfoV2(pred, clean)
    uniform = uniform_loss(pred)
    density = density_consistency_loss(pred, clean) if getattr(args, "loss_density_weight", 0.0) > 0.0 else jt.zeros((), dtype=jt.float32)

    stage_terms = [huber_loss(out["x1"] - clean, delta=args.corr_huber_delta), huber_loss(out["x2"] - clean, delta=args.corr_huber_delta)]
    if "x3" in out:
        stage_terms.append(huber_loss(out["x3"] - clean, delta=args.corr_huber_delta))
    stage = sum(stage_terms) / float(len(stage_terms))

    sigma_terms = []
    for key in ("sigma1_global", "sigma2_global", "sigma3_global"):
        if key not in out:
            continue
        sigma_global = out[key]
        if len(sigma_global.shape) == 1:
            sigma_global = sigma_global.reshape(-1, 1)
        sigma_terms.append(huber_loss(sigma_global - noise_std, delta=0.005))
    sigma_loss = sum(sigma_terms) / float(len(sigma_terms))
    total = (
        args.loss_corr_weight * corr
        + args.loss_relative_weight * relative
        + getattr(args, "loss_pred_cd_weight", 0.0) * pred_cd_relative
        + getattr(args, "loss_clean_cd_weight", 0.0) * clean_cd_relative
        + args.loss_infocd_weight * info
        + args.loss_uniform_weight * uniform
        + getattr(args, "loss_density_weight", 0.0) * density
        + args.loss_stage_weight * stage
        + args.loss_noise_weight * sigma_loss
    )
    metrics = {
        "corr": corr,
        "relative": relative,
        "pred_cd": pred_cd_relative,
        "clean_cd": clean_cd_relative,
        "infocd": info,
        "uniform": uniform,
        "density": density,
        "stage": stage,
        "sigma": sigma_loss,
    }
    return total, metrics


def pgd_training_loss(model, noisy, clean, noise_std, category_id, normals, args):
    out = model(noisy, noise_std=noise_std, category_id=category_id, return_dict=True)
    pred = noisy + out["disp"]
    if getattr(args, "pgd_use_normal_corr_loss", False):
        if normals is None:
            raise ValueError("--pgd_use_normal_corr_loss requires dataset normals")
        corr = normal_tangent_corr_loss(
            pred,
            clean,
            normals,
            normal_weight=getattr(args, "normal_corr_normal_weight", 2.0),
            tangent_weight=getattr(args, "normal_corr_tangent_weight", 1.0),
        )
    else:
        corr = huber_loss(pred - clean, delta=args.corr_huber_delta)
    pred_cd_per, clean_cd_per = per_sample_chamfer_terms(pred, clean)
    baseline_pred_cd_per, baseline_clean_cd_per = per_sample_chamfer_terms(noisy, clean)
    pred_cd = pred_cd_per.mean()
    clean_cd = clean_cd_per.mean()
    baseline_pred_cd = baseline_pred_cd_per.mean()
    baseline_clean_cd = baseline_clean_cd_per.mean()
    baseline = (baseline_pred_cd + baseline_clean_cd).detach()
    relative = (pred_cd + clean_cd) / (baseline + args.relative_eps)
    pred_cd_relative = pred_cd / (baseline_pred_cd.detach() + args.relative_eps)
    clean_cd_relative = clean_cd / (baseline_clean_cd.detach() + args.relative_eps)
    info = calc_cd_like_InfoV2(pred, clean)
    uniform = uniform_loss(pred)
    local_surface = (
        local_surface_distance_loss(pred, clean)
        if getattr(args, "loss_local_surface_weight", 0.0) > 0.0
        else jt.zeros((), dtype=jt.float32)
    )
    score_relative = (
        score_aligned_relative_cd_from_terms(
            pred_cd_per,
            clean_cd_per,
            baseline_pred_cd_per,
            baseline_clean_cd_per,
            eps=args.relative_eps,
        )
        if getattr(args, "loss_score_relative_weight", 0.0) > 0.0
        else jt.zeros((), dtype=jt.float32)
    )
    straight = jt.zeros((), dtype=jt.float32)
    straight_direction = jt.zeros((), dtype=jt.float32)
    straight_distance = jt.zeros((), dtype=jt.float32)
    if getattr(args, "loss_straight_weight", 0.0) > 0.0:
        straight, straight_direction, straight_distance = straight_displacement_loss(
            pred,
            noisy,
            clean,
            direction_weight=args.straight_direction_weight,
            distance_weight=args.straight_distance_weight,
        )
    stage = jt.zeros((), dtype=jt.float32)
    if "x1" in out:
        stage = huber_loss(out["x1"] - clean, delta=args.corr_huber_delta)
    disp_mag = jt.sqrt(jt.maximum(jt.sum(out["disp"] * out["disp"], dim=-1), 1e-12))

    low_noise_disp = jt.zeros((), dtype=jt.float32)

    rotation_consistency = jt.zeros((), dtype=jt.float32)
    rotation_consistency_weight = float(getattr(args, "pgd_rotation_consistency_weight", 0.0))
    if rotation_consistency_weight > 0.0:
        # Enforce yaw equivariance directly.  The same rigid transform is
        # applied to every item in this batch, so category/noise conditioning
        # remains unchanged and the target is available without extra labels.
        theta = np.random.uniform(0.0, 2.0 * np.pi)
        c, s = np.cos(theta), np.sin(theta)
        rot = jt.array(np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32))
        noisy_rot = jt.matmul(noisy, rot)
        out_rot = model(noisy_rot, noise_std=noise_std, category_id=category_id, return_dict=True)
        pred_rot = noisy_rot + out_rot["disp"]
        rotation_consistency = huber_loss(pred_rot - jt.matmul(pred.detach(), rot), delta=args.corr_huber_delta)

    total = (
        args.loss_corr_weight * corr
        + args.loss_relative_weight * relative
        + getattr(args, "loss_pred_cd_weight", 0.0) * pred_cd_relative
        + getattr(args, "loss_clean_cd_weight", 0.0) * clean_cd_relative
        + args.loss_infocd_weight * info
        + args.loss_uniform_weight * uniform
        + getattr(args, "loss_local_surface_weight", 0.0) * local_surface
        + getattr(args, "loss_score_relative_weight", 0.0) * score_relative
        + args.loss_straight_weight * straight
        + args.loss_stage_weight * stage
        + args.pgd_loss_disp_weight * disp_mag.mean()
        + args.pgd_loss_low_noise_disp_weight * low_noise_disp
        + rotation_consistency_weight * rotation_consistency
    )
    metrics = {
        "corr": corr,
        "relative": relative,
        "pred_cd": pred_cd_relative,
        "clean_cd": clean_cd_relative,
        "infocd": info,
        "uniform": uniform,
        "local_surface": local_surface,
        "score_relative": score_relative,
        "straight": straight,
        "straight_dir": straight_direction,
        "straight_dist": straight_distance,
        "stage": stage,
        "disp": disp_mag.mean(),
        "low_disp": low_noise_disp,
        "rot_consistency": rotation_consistency,
    }
    return total, metrics


def sanitize_optimizer_grads(optimizer):
    for pg in optimizer.param_groups:
        for grad in pg["grads"]:
            grad.update(jt.where(jt.isfinite(grad), grad, jt.zeros_like(grad)))


def freeze_batchnorm_stats(model):
    frozen = 0
    for module in model.modules():
        if isinstance(module, nn.BatchNorm):
            module.eval()
            frozen += 1
    return frozen


def select_trainable_parameters(model, args):
    selected_modes = [
        name
        for name in (
            "pgd_train_refine_gate_only",
            "pgd_train_head_only",
            "pgd_train_decoder_head_only",
            "pgd_train_decoder_last_encoder",
        )
        if getattr(args, name, False)
    ]
    if len(selected_modes) > 1:
        raise ValueError("training parameter selection flags are mutually exclusive")
    if getattr(args, "pgd_train_decoder_head_only", False) or getattr(args, "pgd_train_decoder_last_encoder", False):
        allowed = (
            "feature_nets.decoder_blocks.",
            "feature_nets.codebooks.",
            "feature_nets.linear0_1.",
            "feature_nets.linear0_2.",
            "feature_nets.linear0_3.",
        )
        if getattr(args, "pgd_train_decoder_last_encoder", False):
            last_encoder = len(model.feature_nets.encoder_blocks) - 1
            allowed = allowed + ("feature_nets.encoder_blocks.{}.".format(last_encoder),)
        selected = []
        names = []
        for name, param in model.named_parameters():
            if not name.startswith(allowed):
                continue
            if name.endswith(".running_mean") or name.endswith(".running_var"):
                continue
            selected.append(param)
            names.append(name)
        if not selected:
            raise ValueError("--pgd_train_decoder_head_only requires PGD decoder/head parameters")
        return selected, names
    if getattr(args, "pgd_train_head_only", False):
        allowed = (
            "feature_nets.linear0_1.",
            "feature_nets.linear0_2.",
            "feature_nets.linear0_3.",
        )
        selected = []
        names = []
        for name, param in model.named_parameters():
            if name.startswith(allowed):
                if name.endswith(".running_mean") or name.endswith(".running_var"):
                    continue
                selected.append(param)
                names.append(name)
        if not selected:
            raise ValueError("--pgd_train_head_only requires a PGD displacement head")
        return selected, names
    if getattr(args, "pgd_train_refine_gate_only", False):
        allowed = ("refine_gate_fc1.", "refine_gate_fc2.")
        selected = []
        names = []
        for name, param in model.named_parameters():
            if name.startswith(allowed):
                selected.append(param)
                names.append(name)
        if not selected:
            raise ValueError("--pgd_train_refine_gate_only requires --pgd_use_refine_gate")
        return selected, names
    return model.parameters(), []


def get_loss_fn(name):
    if name == "infocd":
        return calc_cd_like_InfoV2
    if name == "chamfer":
        return chamfer_loss
    raise ValueError("unsupported loss: {}".format(name))


def stack_batch(samples):
    noisy = jt.array(np.stack([s[0] for s in samples], axis=0))
    clean = jt.array(np.stack([s[1] for s in samples], axis=0))
    noise_std = jt.array(np.asarray([[s[3]] for s in samples], dtype=np.float32))
    category_id = jt.array(np.asarray([s[4] for s in samples], dtype=np.int32))
    normals = None
    if len(samples[0]) > 5:
        normals = jt.array(np.stack([s[5] for s in samples], axis=0))
    return noisy, clean, noise_std, category_id, normals


def clean_mpi_env():
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(("OMPI_", "PMIX_", "PMI_", "MPI_", "OPAL_")):
            env.pop(key, None)
    env["use_mpi"] = "0"
    return env


def parse_eval_log(log_path):
    result = {}
    if not log_path.exists():
        return result
    text = log_path.read_text(errors="ignore")
    patterns = {
        "cd_score": r"CD\s+得分:\s*([0-9.]+)",
        "p2s_score": r"P2S\s+得分:\s*([0-9.]+)",
        "final_score": r"最终得分.*?:\s*([0-9.]+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            result[key] = float(match.group(1))
    return result


def run_starter_evaluate(args, eval_root):
    cmd = [
        sys.executable,
        str(Path(args.eval_starter_root) / "evaluate.py"),
        "--pred_dir", str(eval_root / "pred"),
        "--gt_dir", str(eval_root / "gt"),
        "--noisy_dir", str(eval_root / "noisy"),
        "--mesh_dir", str(args.dataset_root),
        "--pred_filename", "denoised.npy",
        "--gt_filename", "clean.npy",
        "--noisy_filename", "noisy.npy",
        "--workers", str(args.eval_workers),
        "--verbose",
    ]
    log_path = eval_root / "evaluate.log"
    with open(log_path, "w") as f:
        proc = subprocess.run(cmd, cwd=args.eval_starter_root, stdout=f, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError("starter evaluate failed with code {}; see {}".format(proc.returncode, log_path))


def run_mesh_eval_after_epoch(args, epoch, ckpt):
    eval_root = Path(args.log_dir) / "eval_epoch{:02d}".format(epoch)
    base_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "eval_shapenet_mesh_val.py"),
        "--weights", str(ckpt),
        "--model", args.model,
        "--dataset_root", args.dataset_root,
        "--datalist_dir", args.datalist_dir,
        "--val_list", args.eval_val_list or str(Path(args.datalist_dir) / "validate.txt"),
        "--data_name", args.data_name,
        "--output_root", str(eval_root),
        "--starter_root", args.eval_starter_root,
        "--sample_points", str(args.eval_sample_points if args.eval_sample_points > 0 else args.sample_points),
        "--noise_std_min", str(args.noise_std_min if args.noise_std_min is not None else args.noise_std),
        "--noise_std_max", str(args.noise_std_max if args.noise_std_max is not None else args.noise_std),
        "--seed", str(args.eval_seed + epoch),
        "--patch_size", str(args.patch_size),
        "--seed_k", str(args.eval_seed_k),
        "--seed_k_alpha", str(args.eval_seed_k_alpha),
        "--patch_batch_size", str(args.eval_patch_batch_size),
        "--patch_fusion", args.eval_patch_fusion,
        "--niters", str(args.eval_niters),
        "--category_embed_dim", str(args.category_embed_dim),
        "--noise_embed_dim", str(args.noise_embed_dim),
        "--asdn_max_disp", str(args.asdn_max_disp),
        "--asdn_stage3_noise_threshold", str(args.asdn_stage3_noise_threshold),
        "--asdn_stage3_conf_threshold", str(args.asdn_stage3_conf_threshold),
        "--workers", str(args.eval_workers),
    ]
    if args.asdn_use_codebook:
        base_cmd.append("--asdn_use_codebook")
    if args.asdn_stage3:
        base_cmd.append("--asdn_stage3")
    if args.pgd_two_stage:
        base_cmd.extend(["--pgd_two_stage", "--pgd_second_stage_scale", str(args.pgd_second_stage_scale)])
    if args.pgd_use_refine_gate:
        base_cmd.extend([
            "--pgd_use_refine_gate",
            "--pgd_refine_gate_scale", str(args.pgd_refine_gate_scale),
        ])
    if args.use_cuda:
        base_cmd.append("--use_cuda")
    if args.eval_skip_existing:
        base_cmd.append("--skip_existing")
    eval_root.mkdir(parents=True, exist_ok=True)
    num_shards = max(1, int(args.eval_num_shards))
    env_base = clean_mpi_env()
    devices = [d.strip() for d in args.eval_devices.split(",") if d.strip()]
    if not devices:
        devices = [d.strip() for d in os.environ.get("PGD_MPI_DEVICES", "0").split(",") if d.strip()]
    if num_shards == 1:
        log_path = eval_root / "eval_after_epoch.log"
        cmd = base_cmd + ["--run_evaluate"]
        with open(log_path, "w") as f:
            proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env_base, stdout=f, stderr=subprocess.STDOUT)
        if proc.returncode != 0:
            raise RuntimeError("epoch eval failed with code {}; see {}".format(proc.returncode, log_path))
    else:
        procs = []
        for shard in range(num_shards):
            env = env_base.copy()
            env["CUDA_VISIBLE_DEVICES"] = devices[shard % len(devices)]
            log_path = eval_root / "eval_after_epoch_shard{:02d}.log".format(shard)
            cmd = base_cmd + ["--num_shards", str(num_shards), "--shard_index", str(shard)]
            f = open(log_path, "w")
            proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), env=env, stdout=f, stderr=subprocess.STDOUT)
            procs.append((shard, proc, f, log_path))
        failures = []
        for shard, proc, f, log_path in procs:
            code = proc.wait()
            f.close()
            if code != 0:
                failures.append((shard, code, log_path))
        if failures:
            raise RuntimeError("epoch eval shard failed: {}".format(failures))
        run_starter_evaluate(args, eval_root)
    result = parse_eval_log(eval_root / "evaluate.log")
    result["output_root"] = str(eval_root)
    result["log"] = str(eval_root / "evaluate.log")
    return result


def run_epoch(model, dataset, optimizer, batch_size, loss_fn, args, train=True, desc="train", max_steps=0):
    indices = list(range(len(dataset)))
    if train:
        random.shuffle(indices)
        model.train()
        if getattr(args, "freeze_batchnorm_stats", False):
            freeze_batchnorm_stats(model)
    else:
        model.eval()
    if jt.in_mpi:
        indices = indices[jt.rank::jt.world_size]
    losses = []
    if jt.in_mpi:
        # Jittor's optimizer synchronizes gradients every step. Keep every rank
        # on the same number of full batches so the last all-reduce cannot wait
        # for a rank that has already left the loop.
        total = len(indices) // batch_size
    else:
        total = int(math.ceil(len(indices) / float(batch_size)))
    if max_steps > 0:
        total = min(total, max_steps)
    disable_pbar = jt.in_mpi and jt.rank != 0
    pbar = tqdm(range(total), desc=desc, disable=disable_pbar)
    for step in pbar:
        batch_idx = indices[step * batch_size:(step + 1) * batch_size]
        if not batch_idx:
            break
        samples = [dataset[i] for i in batch_idx]
        noisy, clean, noise_std, category_id, normals = stack_batch(samples)
        if args.model == "asdn":
            loss, metrics = asdn_training_loss(model, noisy, clean, noise_std, category_id, args)
        elif args.pgd_composite_loss:
            loss, metrics = pgd_training_loss(model, noisy, clean, noise_std, category_id, normals, args)
        else:
            pred = noisy + model(noisy, noise_std=noise_std, category_id=category_id)
            loss = loss_fn(pred, clean)
            metrics = {}
        val = float(loss.numpy())
        if not np.isfinite(val):
            pbar.set_postfix(loss="nonfinite")
            continue
        if train:
            if args.grad_clip_norm > 0:
                optimizer.zero_grad()
                optimizer.backward(loss)
                sanitize_optimizer_grads(optimizer)
                optimizer.clip_grad_norm(args.grad_clip_norm)
                optimizer.step()
            else:
                optimizer.step(loss)
        losses.append(val)
        # Persist lightweight, append-only measurements during training.  A
        # full epoch can take hours on 50k-point patches, so an epoch average
        # alone cannot distinguish healthy convergence from a noisy plateau.
        # This records train data only and is deliberately independent of any
        # validation split.
        every = int(getattr(args, "step_metrics_every", 10))
        is_main = (not jt.in_mpi) or jt.rank == 0
        if train and is_main and every > 0 and (step % every == 0 or step + 1 == total):
            payload = {"step": int(step), "loss": val}
            for key, metric in metrics.items():
                payload[key] = float(metric.numpy())
            with open(Path(args.log_dir) / "train_steps.jsonl", "a") as f:
                f.write(json.dumps(payload, sort_keys=True) + "\n")
        if metrics:
            pbar.set_postfix(loss="{:.6f}".format(val), corr="{:.5f}".format(float(metrics["corr"].numpy())))
        else:
            pbar.set_postfix(loss="{:.6f}".format(val))
    return float(np.mean(losses)) if losses else 0.0


def build_model(args):
    if args.model == "asdn":
        return ASDNModel(args)
    return PGDModel(args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", default="/home/dataset_train")
    parser.add_argument("--datalist_dir", default="/home/PGD/datalist")
    parser.add_argument("--data_name", default="models/model_normalized.obj")
    parser.add_argument("--sample_points", type=int, default=50000)
    parser.add_argument("--noise_std", type=float, default=0.025)
    parser.add_argument("--noise_std_min", type=float, default=None)
    parser.add_argument("--noise_std_max", type=float, default=None)
    parser.add_argument("--noise_types", default=DEFAULT_NOISE_TYPES)
    parser.add_argument("--patch_size", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--grad_clip_norm", type=float, default=-1.0)
    parser.add_argument("--freeze_batchnorm_stats", action="store_true")
    parser.add_argument("--model", choices=["pgd", "asdn"], default="pgd")
    parser.add_argument("--loss", choices=["infocd", "chamfer"], default="infocd")
    parser.add_argument("--category_embed_dim", type=int, default=16)
    parser.add_argument("--noise_embed_dim", type=int, default=16)
    parser.add_argument("--asdn_max_disp", type=float, default=1.0)
    parser.add_argument("--asdn_use_codebook", action="store_true")
    parser.add_argument("--asdn_stage3", action="store_true")
    parser.add_argument("--asdn_stage3_noise_threshold", type=float, default=0.018)
    parser.add_argument("--asdn_stage3_conf_threshold", type=float, default=0.45)
    parser.add_argument("--pgd_train_refine_gate_only", action="store_true")
    parser.add_argument("--pgd_train_head_only", action="store_true")
    parser.add_argument("--pgd_train_decoder_head_only", action="store_true")
    parser.add_argument("--pgd_train_decoder_last_encoder", action="store_true")
    parser.add_argument("--pgd_two_stage", action="store_true")
    parser.add_argument("--pgd_second_stage_scale", type=float, default=1.0)
    parser.add_argument("--pgd_use_refine_gate", action="store_true")
    parser.add_argument("--pgd_refine_gate_scale", type=float, default=0.25)
    parser.add_argument("--pgd_train_detach_second_stage_backbone", action="store_true")
    parser.add_argument("--pgd_composite_loss", action="store_true")
    parser.add_argument("--pgd_use_normal_corr_loss", action="store_true")
    parser.add_argument("--normal_k", type=int, default=16)
    parser.add_argument("--normal_corr_normal_weight", type=float, default=2.0,
                        help="training-only weight for normal correction error (P2S-oriented)")
    parser.add_argument("--normal_corr_tangent_weight", type=float, default=1.0,
                        help="training-only weight for tangent correction error (CD-oriented)")
    parser.add_argument("--pgd_loss_disp_weight", type=float, default=0.0)
    parser.add_argument("--pgd_loss_low_noise_disp_weight", type=float, default=0.0)
    parser.add_argument("--pgd_rotation_consistency_weight", type=float, default=0.0,
                        help="training-only yaw equivariance consistency weight")
    parser.add_argument("--loss_corr_weight", type=float, default=1.0)
    parser.add_argument("--loss_relative_weight", type=float, default=0.5)
    parser.add_argument("--loss_pred_cd_weight", type=float, default=0.0)
    parser.add_argument("--loss_clean_cd_weight", type=float, default=0.0)
    parser.add_argument("--loss_infocd_weight", type=float, default=0.15)
    parser.add_argument("--loss_uniform_weight", type=float, default=0.10)
    parser.add_argument("--loss_density_weight", type=float, default=0.0,
                        help="train-only local spacing consistency weight")
    parser.add_argument("--loss_local_surface_weight", type=float, default=0.0,
                        help="train-only local pair-distance geometry weight")
    parser.add_argument("--loss_score_relative_weight", type=float, default=0.0,
                        help="train-only per-sample CD-ratio term aligned with the official CD scorer")
    parser.add_argument("--loss_straight_weight", type=float, default=0.0)
    parser.add_argument("--straight_direction_weight", type=float, default=1.0)
    parser.add_argument("--straight_distance_weight", type=float, default=1.0)
    parser.add_argument("--loss_stage_weight", type=float, default=0.20)
    parser.add_argument("--loss_noise_weight", type=float, default=0.05)
    parser.add_argument("--corr_huber_delta", type=float, default=0.01)
    parser.add_argument("--relative_eps", type=float, default=1e-8)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--use_cuda", action="store_true")
    parser.add_argument("--max_train_shapes", type=int, default=0)
    parser.add_argument("--max_val_shapes", type=int, default=0)
    parser.add_argument("--precomputed_points_dir", default="")
    parser.add_argument("--cache_points", action="store_true")
    parser.add_argument("--patches_per_shape", type=int, default=4)
    parser.add_argument("--random_z_rotation", action="store_true",
                        help="apply a shared random yaw to noisy/clean training patches")
    parser.add_argument("--density_jitter_ratio", type=float, default=0.0,
                        help="train-only paired point resampling ratio to improve density robustness")
    parser.add_argument("--train_steps_per_epoch", type=int, default=0)
    parser.add_argument("--val_steps_per_epoch", type=int, default=0)
    parser.add_argument("--init_weights", default="")
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--eval_after_epoch", action="store_true")
    parser.add_argument("--eval_val_list", default="")
    parser.add_argument("--eval_sample_points", type=int, default=0)
    parser.add_argument("--eval_seed", type=int, default=2026)
    parser.add_argument("--eval_seed_k", type=int, default=5)
    parser.add_argument("--eval_seed_k_alpha", type=float, default=10)
    parser.add_argument("--eval_patch_batch_size", type=int, default=8)
    parser.add_argument("--eval_patch_fusion", choices=["select", "weighted"], default="select")
    parser.add_argument("--eval_pgd_gate_noise_source", choices=["known", "estimate"], default="estimate")
    parser.add_argument("--eval_niters", type=int, default=1)
    parser.add_argument("--eval_workers", type=int, default=8)
    parser.add_argument("--eval_num_shards", type=int, default=1)
    parser.add_argument("--eval_devices", default="")
    parser.add_argument("--eval_starter_root", default="/home/starter_code")
    parser.add_argument("--eval_skip_existing", action="store_true")
    parser.add_argument("--log_dir", default="/home/PGD/experiments/shapenet_50k_gaussian_025_one_epoch")
    parser.add_argument("--step_metrics_every", type=int, default=10,
                        help="write train-only loss metrics every N steps (0 disables)")
    args = parser.parse_args()
    if args.grad_clip_norm < 0:
        args.grad_clip_norm = 1.0 if args.model == "asdn" else 0.0

    if args.use_cuda:
        jt.flags.use_cuda = 1
    is_main = (not jt.in_mpi) or jt.rank == 0
    random.seed(args.seed)
    np.random.seed(args.seed)
    jt.set_global_seed(args.seed)

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    # In a training-only run, do not even read validation/test split files.
    # Categories are present in the training split and are sufficient for the
    # optional category embedding.
    category_to_id = build_category_vocab(args.datalist_dir, split_names=("train.txt",))

    train_set = ShapeNetPatchDataset(
        args.dataset_root,
        Path(args.datalist_dir) / "train.txt",
        args.data_name,
        args.sample_points,
        args.patch_size,
        args.noise_std,
        noise_std_min=args.noise_std_min,
        noise_std_max=args.noise_std_max,
        noise_types=args.noise_types,
        max_shapes=args.max_train_shapes,
        precomputed_points_dir=args.precomputed_points_dir,
        cache_points=args.cache_points,
        patches_per_shape=args.patches_per_shape,
        random_z_rotation=args.random_z_rotation,
        density_jitter_ratio=args.density_jitter_ratio,
        category_to_id=category_to_id,
        return_normals=args.pgd_use_normal_corr_loss,
        normal_k=args.normal_k,
    )
    val_set = None
    if args.val_steps_per_epoch > 0:
        val_set = ShapeNetPatchDataset(
            args.dataset_root,
            Path(args.datalist_dir) / "validate.txt",
            args.data_name,
            args.sample_points,
            args.patch_size,
            args.noise_std,
            noise_std_min=args.noise_std_min,
            noise_std_max=args.noise_std_max,
            noise_types=args.noise_types,
            max_shapes=args.max_val_shapes,
            precomputed_points_dir=args.precomputed_points_dir,
            cache_points=args.cache_points,
            patches_per_shape=1,
            random_z_rotation=False,
            density_jitter_ratio=0.0,
            category_to_id=category_to_id,
            return_normals=args.pgd_use_normal_corr_loss,
            normal_k=args.normal_k,
        )
    args.num_categories = train_set.num_categories if val_set is None else max(train_set.num_categories, val_set.num_categories)
    if is_main:
        args_to_save = vars(args).copy()
        args_to_save["category_to_id"] = category_to_id
        with open(log_dir / "args.json", "w") as f:
            json.dump(args_to_save, f, indent=2)
    if is_main:
        val_count = "disabled" if val_set is None else str(len(val_set))
        print("train shapes: {}, val shapes: {}".format(len(train_set), val_count), flush=True)
        if jt.in_mpi:
            print("mpi world_size: {}, rank0 local_rank: {}".format(jt.world_size, jt.mpi.local_rank()), flush=True)
    if is_main and (args.train_steps_per_epoch > 0 or args.val_steps_per_epoch > 0):
        print(
            "train_steps_per_epoch: {}, val_steps_per_epoch: {}".format(
                args.train_steps_per_epoch,
                args.val_steps_per_epoch,
            ),
            flush=True,
        )
    if is_main and args.noise_std_min is not None and args.noise_std_max is not None:
        print(
            "sample_points: {}, noise types: {}, std range: [{}, {}]".format(
                args.sample_points,
                ",".join(parse_noise_types(args.noise_types)),
                args.noise_std_min,
                args.noise_std_max,
            ),
            flush=True,
        )
    elif is_main:
        print(
            "sample_points: {}, noise types: {}, std: {}".format(
                args.sample_points,
                ",".join(parse_noise_types(args.noise_types)),
                args.noise_std,
            ),
            flush=True,
        )
    if is_main:
        print("model: {}, num_categories: {}".format(args.model, args.num_categories), flush=True)
        print("loss: {}".format(args.loss), flush=True)
        print("grad_clip_norm: {}".format(args.grad_clip_norm), flush=True)
        print("freeze_batchnorm_stats: {}".format(args.freeze_batchnorm_stats), flush=True)
        print("patches_per_shape train: {}, val: 1".format(args.patches_per_shape), flush=True)
    if is_main and args.precomputed_points_dir:
        print("precomputed_points_dir: {} cache_points={}".format(args.precomputed_points_dir, args.cache_points), flush=True)

    model = build_model(args)
    if args.init_weights:
        load_report = model.load_npz(args.init_weights)
        if is_main:
            print("init_weights: {} loaded={} missing={}".format(args.init_weights, load_report["loaded"], len(load_report["missing"])), flush=True)
    trainable_params, trainable_names = select_trainable_parameters(model, args)
    optimizer = nn.Adam(trainable_params, lr=args.lr)
    if is_main:
        print("optimizer params: {}".format(len(trainable_params)), flush=True)
        if trainable_names:
            print("optimizer param names: {}".format(",".join(trainable_names)), flush=True)
    loss_fn = get_loss_fn(args.loss)

    history = []
    history_path = log_dir / "history.json"
    if is_main and history_path.exists():
        with open(history_path, "r") as f:
            loaded_history = json.load(f)
        if isinstance(loaded_history, list):
            history = loaded_history
    end_epoch = args.start_epoch + args.epochs
    for epoch in range(args.start_epoch, end_epoch):
        t0 = time.time()
        train_loss = run_epoch(
            model,
            train_set,
            optimizer,
            args.batch_size,
            loss_fn,
            args,
            train=True,
            desc="train epoch {:02d}".format(epoch),
            max_steps=args.train_steps_per_epoch,
        )
        # A zero value explicitly disables validation.  This keeps training-only
        # runs from touching validation data at all; positive values request a
        # no-gradient diagnostic pass only.
        val_loss = None
        if args.val_steps_per_epoch > 0:
            val_loss = run_epoch(
                model,
                val_set,
                optimizer,
                args.batch_size,
                loss_fn,
                args,
                train=False,
                desc="val epoch {:02d}".format(epoch),
                max_steps=args.val_steps_per_epoch,
            )
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "seconds": time.time() - t0,
        }
        if jt.in_mpi:
            jt.mpi.mpi_barrier()
        if is_main:
            ckpt = log_dir / "pgd-shapenet-epoch{:02d}-loss{:.8f}.npz".format(epoch, train_loss)
            model.save_npz(ckpt)
            if args.eval_after_epoch:
                print("[Epoch {:02d}] running mesh val eval...".format(epoch), flush=True)
                row["eval"] = run_mesh_eval_after_epoch(args, epoch, ckpt)
            history = [item for item in history if item.get("epoch") != epoch]
            history.append(row)
            history.sort(key=lambda item: item.get("epoch", -1))
            with open(history_path, "w") as f:
                json.dump(history, f, indent=2)
            val_text = "disabled" if val_loss is None else "{:.8f}".format(val_loss)
            print("[Epoch {:02d}] train_loss={:.8f} val_loss={} ckpt={}".format(epoch, train_loss, val_text, ckpt), flush=True)
            if args.eval_after_epoch:
                print("[Epoch {:02d}] eval={}".format(epoch, row["eval"]), flush=True)
        if jt.in_mpi:
            jt.mpi.mpi_barrier()


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    if jt.in_mpi:
        jt.mpi.mpi_barrier()
    os._exit(0)
