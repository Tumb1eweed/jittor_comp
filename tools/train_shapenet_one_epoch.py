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
try:
    from tools.manual_dist import ManualDist
except ImportError:
    # When this file is executed directly, Python puts tools/ (not the
    # repository root) first on sys.path.
    from manual_dist import ManualDist
from utils.noise import DEFAULT_NOISE_TYPES, add_numpy_noise, parse_noise_types, sample_noise_std


MANUAL_DIST = None


def read_split(path):
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def build_category_vocab_from_paths(split_paths):
    """Build a stable synset-ID vocabulary from explicit split files only."""
    items = []
    for path in split_paths:
        path = Path(path)
        if path.exists():
            items.extend(read_split(path))
    categories = sorted({extract_synset_id(item) for item in items})
    return {cat: i + 1 for i, cat in enumerate(categories)}


def build_category_vocab(datalist_dir, split_names=("train.txt", "validate.txt", "test.txt")):
    """Build the canonical conditional-embedding vocabulary.

    This reads split *identifiers* only.  It never adds validation or test
    shapes to the training dataset, but preserves a stable category-ID layout
    across training, validation, test inference, and historical checkpoints.
    """
    return build_category_vocab_from_paths([Path(datalist_dir) / name for name in split_names])


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


def make_overlapping_patches(clean, noisy, patch_size):
    """Return two independently centred, overlapping local patches.

    The final three arrays identify identical source points in the two patches.
    They make the overlap-consistency objective supervised only by the training
    cloud's own correspondence; neither validation nor test geometry is read.
    """
    n = noisy.shape[0]
    if patch_size > n:
        raise ValueError("overlap patches require patch_size <= source point count")
    seed_a = np.random.randint(0, n)
    dist_a = ((noisy - noisy[seed_a:seed_a + 1]) ** 2).sum(axis=1)
    idx_a = np.argpartition(dist_a, patch_size)[:patch_size]
    # Picking the second centre inside the first local neighbourhood guarantees
    # meaningful overlap while still changing the patch context and origin.
    seed_b = int(idx_a[np.random.randint(0, len(idx_a))])
    dist_b = ((noisy - noisy[seed_b:seed_b + 1]) ** 2).sum(axis=1)
    idx_b = np.argpartition(dist_b, patch_size)[:patch_size]
    common, local_a, local_b = np.intersect1d(idx_a, idx_b, return_indices=True)
    if len(common) == 0:
        raise RuntimeError("overlap patch sampler produced no shared points")
    map_a = np.zeros((patch_size,), dtype=np.int64)
    map_b = np.zeros((patch_size,), dtype=np.int64)
    mask = np.zeros((patch_size,), dtype=np.float32)
    count = min(patch_size, len(common))
    map_a[:count] = local_a[:count]
    map_b[:count] = local_b[:count]
    mask[:count] = 1.0
    patch_a_noisy = noisy[idx_a] - noisy[seed_a]
    patch_a_clean = clean[idx_a] - noisy[seed_a]
    patch_b_noisy = noisy[idx_b] - noisy[seed_b]
    return patch_a_noisy, patch_a_clean, patch_b_noisy, map_a, map_b, mask


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
        return_overlap_patch=False,
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
        self.return_overlap_patch = bool(return_overlap_patch)
        if self.return_overlap_patch and self.density_jitter_ratio > 0.0:
            raise ValueError("overlap consistency is incompatible with density_jitter_ratio > 0")

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
        overlap = None
        if self.return_overlap_patch:
            patch_noisy, patch_clean, overlap_noisy, overlap_a, overlap_b, overlap_mask = make_overlapping_patches(
                clean, noisy, self.patch_size
            )
            overlap = (overlap_noisy, overlap_a, overlap_b, overlap_mask)
        else:
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
            if overlap is not None:
                overlap = (np.matmul(overlap[0], rot), overlap[1], overlap[2], overlap[3])
        category_id = self.category_to_id.get(extract_synset_id(rel), 0)
        sample = [patch_noisy.astype(np.float32), patch_clean.astype(np.float32), str(path), float(noise_std), int(category_id)]
        if self.return_normals:
            sample.append(estimate_patch_normals_np(patch_clean, k=self.normal_k))
        if overlap is not None:
            sample.append(overlap)
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


def normal_tangent_corr_loss(
    pred,
    clean,
    normals,
    normal_weight=2.0,
    tangent_weight=1.0,
    noisy=None,
    relative=False,
    eps=1e-8,
):
    normals = normals / (jt.norm(normals, dim=-1, keepdims=True) + 1e-8)
    error = pred - clean
    normal_error = jt.sum(error * normals, dim=-1, keepdims=True) * normals
    tangent_error = error - normal_error
    # Normal error approximates point-to-surface degradation, while tangent
    # error is needed to improve point placement and hence Chamfer distance.
    # Keep their trade-off explicit so experiments can target CD without
    # silently sacrificing P2S.  These terms are training-only and require no
    # metadata at inference.
    per_point = (
        float(normal_weight) * jt.sum(normal_error * normal_error, dim=-1)
        + float(tangent_weight) * jt.sum(tangent_error * tangent_error, dim=-1)
    )
    if not relative:
        return jt.mean(per_point)
    if noisy is None:
        raise ValueError("relative normal/tangent correspondence requires noisy points")
    noisy_error = noisy - clean
    noisy_normal = jt.sum(noisy_error * normals, dim=-1, keepdims=True) * normals
    noisy_tangent = noisy_error - noisy_normal
    noisy_per_point = (
        float(normal_weight) * jt.sum(noisy_normal * noisy_normal, dim=-1)
        + float(tangent_weight) * jt.sum(noisy_tangent * noisy_tangent, dim=-1)
    )
    # Match the official scorer's per-sample relative aggregation. This also
    # keeps correspondence supervision on the same O(1) scale as relative CD
    # instead of letting coordinate-scale squared errors vanish in the total.
    return (
        per_point.mean(dim=1)
        / (noisy_per_point.mean(dim=1).detach() + float(eps))
    ).mean()


def stage2_surface_residual_losses(out, clean, normals, delta=0.01):
    """Stage-specific supervision for a surface-constrained PGD refiner."""
    if "x1" not in out or out.get("disp2_normal") is None:
        zero = jt.zeros((), dtype=jt.float32)
        return zero, zero, zero
    normals = normals / (jt.norm(normals, dim=-1, keepdims=True) + 1e-8)
    final = out["final"]
    target = clean - out["x1"]
    target_normal = jt.sum(target * normals, dim=-1, keepdims=True) * normals
    target_tangent = target - target_normal
    final_normal_error = jt.sum((final - clean) * normals, dim=-1, keepdims=True)
    plane = huber_loss(final_normal_error, delta=delta)
    normal_residual = huber_loss(
        out["disp2_normal"] - target_normal, delta=delta
    )
    tangent_target = huber_loss(
        out["disp2_tangent"] - target_tangent, delta=delta
    )
    return plane, normal_residual, tangent_target


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


def anti_cluster_knn_loss(pred, clean, k=8, margin=0.85):
    """Penalize local point collapse without forcing sparse regions inward.

    Corresponding train pairs retain point identities, so each prediction can
    compare its first ``k`` neighbour spacings with the clean surface.  The
    one-sided hinge activates only when the predicted neighbourhood is
    materially *smaller* than the clean one.  This targets visible clusters
    while avoiding a uniformity prior that can pull valid edges/curvature off
    the surface and reduce P2S.
    """
    n = int(pred.shape[1])
    k = max(1, min(int(k), n - 1))
    eye = jt.array(np.eye(n, dtype=np.float32)).unsqueeze(0) * 1e6
    pred_sq = jt.sum((pred[:, :, None, :] - pred[:, None, :, :]) ** 2, dim=-1) + eye
    clean_sq = jt.sum((clean[:, :, None, :] - clean[:, None, :, :]) ** 2, dim=-1) + eye
    pred_knn, _ = jt.topk(pred_sq, k=k, dim=2, largest=False)
    clean_knn, _ = jt.topk(clean_sq, k=k, dim=2, largest=False)
    pred_knn = jt.sqrt(jt.maximum(pred_knn, 1e-12))
    clean_knn = jt.sqrt(jt.maximum(clean_knn, 1e-12))
    ratio = pred_knn / (clean_knn.detach() + 1e-4)
    collapse = jt.maximum(float(margin) - ratio, 0.0) / max(float(margin), 1e-6)
    return huber_loss(collapse, delta=0.25)


def tangent_spacing_consistency_loss(pred, clean, normals, max_points=192, k=8, collapse_margin=0.90):
    """Preserve local *surface* sampling spacing during denoising.

    The clean training cloud supplies a local tangent plane at each point.
    We compare the first few projected neighbour distances of the prediction
    with those of that clean surface, and add a one-sided collapse term.  In
    contrast to global uniformity, this deliberately retains legitimate
    density changes, sharp features and curvature.  Normals are used only
    while training; inference receives no geometric side information.
    """
    n = int(pred.shape[1])
    count = min(n, max(2, int(max_points)))
    k = max(1, min(int(k), count - 1))
    # Patches are randomly sampled by the dataset, so a fixed prefix here is
    # an unbiased and reproducible low-cost subset of each training patch.
    pred = pred[:, :count, :]
    clean = clean[:, :count, :]
    normals = normals[:, :count, :]
    normals = normals / (jt.norm(normals, dim=-1, keepdims=True) + 1e-8)

    def projected_knn(points):
        delta = points[:, :, None, :] - points[:, None, :, :]
        # Each source point uses its clean tangent plane.  This makes the
        # spacing statistic insensitive to normal-direction denoising error.
        normal_component = jt.sum(delta * normals[:, :, None, :], dim=-1, keepdims=True)
        tangent_delta = delta - normal_component * normals[:, :, None, :]
        tangent_sq = jt.sum(tangent_delta * tangent_delta, dim=-1)
        eye = jt.array(np.eye(count, dtype=np.float32)).unsqueeze(0) * 1e6
        values, _ = jt.topk(tangent_sq + eye, k=k, dim=2, largest=False)
        return jt.sqrt(jt.maximum(values, 1e-12))

    pred_knn = projected_knn(pred)
    clean_knn = projected_knn(clean).detach()
    ratio = pred_knn / (clean_knn + 1e-4)
    # Matching all local order statistics catches both overly sparse and
    # overly dense output, while the hinge gives clusters an explicit signal.
    spacing = huber_loss(ratio - 1.0, delta=0.25)
    collapse = huber_loss(
        jt.maximum(float(collapse_margin) - ratio, 0.0) / max(float(collapse_margin), 1e-6),
        delta=0.25,
    )
    return spacing + collapse


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


def sinkhorn_coverage_cost(pred, clean, max_points=128, iterations=5, blur=2.0):
    """Balanced local transport cost from clean targets to predictions.

    Ordinary clean-to-pred Chamfer uses an independent nearest prediction for
    every clean point.  This is precisely the failure mode visible in the
    clustered reconstructions: several nearby clean targets can select the
    same prediction, so the loss gives little gradient to populate a hole.
    Here each clean target and predicted point has equal transport mass.  A
    small entropically regularised Sinkhorn solve therefore makes an occupied
    cluster expensive when it must also cover neighbouring, unoccupied clean
    surface samples.  It is intentionally computed on a small local subset,
    so it is a train-time-only CD-directional regularizer rather than a costly
    inference component or a replacement for point-wise supervision.

    The regularisation scale is tied to clean nearest-neighbour spacing.  This
    makes the same ``blur`` usable on dense and sparse ShapeNet patches.
    """
    m = min(int(max_points), int(pred.shape[1]))
    if m < 2:
        return jt.zeros((), dtype=jt.float32)
    pred = pred[:, :m, :]
    clean = clean[:, :m, :]
    cost = jt.sum((clean[:, :, None, :] - pred[:, None, :, :]) ** 2, dim=-1)

    eye = jt.array(np.eye(m, dtype=np.float32)).unsqueeze(0) * 1e6
    clean_sq = jt.sum((clean[:, :, None, :] - clean[:, None, :, :]) ** 2, dim=-1)
    local_sq = (clean_sq + eye).min(dim=2).mean(dim=1)
    epsilon = jt.maximum(local_sq.detach() * float(blur), 1e-6)
    kernel = jt.exp(-cost / epsilon.reshape((-1, 1, 1))) + 1e-12

    # Scaling against unit (rather than 1/m) marginals is equivalent after the
    # final division by m and keeps the values well conditioned in float32.
    u = jt.ones((pred.shape[0], m), dtype=jt.float32)
    v = jt.ones((pred.shape[0], m), dtype=jt.float32)
    for _ in range(max(1, int(iterations))):
        u = 1.0 / (jt.sum(kernel * v.unsqueeze(1), dim=2) + 1e-8)
        v = 1.0 / (jt.sum(kernel * u.unsqueeze(2), dim=1) + 1e-8)
    transport = u.unsqueeze(2) * kernel * v.unsqueeze(1)
    return jt.sum(transport * cost) / float(pred.shape[0] * m)


def sinkhorn_coverage_relative_loss(pred, clean, noisy, max_points=128, iterations=5, blur=2.0, eps=1e-8):
    """Train-only, score-scaled balanced coverage objective.

    The denominator is the same local OT cost for the noisy input, detached
    so the objective is invariant to per-patch scale/noise while gradients act
    only on the denoiser output.
    """
    pred_cost = sinkhorn_coverage_cost(pred, clean, max_points, iterations, blur)
    noisy_cost = sinkhorn_coverage_cost(noisy, clean, max_points, iterations, blur)
    return pred_cost / (noisy_cost.detach() + eps)


def density_aware_chamfer_relative_loss(pred, clean, noisy, max_points=256, alpha=1.0,
                                        n_lambda=1.0, eps=1e-8):
    """DCD-style, many-to-one-aware CD term, normalized by noisy-input DCD.

    A conventional clean-to-pred Chamfer term allows many clean points to
    select one predicted point; a compact output cluster can consequently
    appear inexpensive.  Density-aware Chamfer Distance (DCD) reweights each
    nearest-neighbour contribution by the inverse number of source points that
    selected that neighbour.  This directly exposes the multiplicity pattern
    created by clusters, while retaining the simple nearest-neighbour gradient
    of CD.  The implementation follows the public NeurIPS-2021 formulation,
    but scales its exponential temperature by each clean patch's nearest-
    neighbour spacing so one setting works across ShapeNet sampling densities.

    It is train-only: neither the ground truth nor multiplicity information is
    used during inference.  A small prefix subset keeps the extra one-hot
    counting operation bounded; random local patches make that subset vary
    across optimization steps.
    """
    m = min(int(max_points), int(pred.shape[1]), int(clean.shape[1]))
    if m < 2:
        return jt.zeros((), dtype=jt.float32)

    pred = pred[:, :m, :]
    clean = clean[:, :m, :]
    noisy = noisy[:, :m, :]

    # The scale is a detached squared clean nearest-neighbour distance.  It
    # turns alpha into a dimensionless sharpness and prevents exp() saturation
    # for different local patch radii and mesh sampling densities.
    eye = jt.array(np.eye(m, dtype=np.float32)).unsqueeze(0) * 1e6
    clean_pair_sq = jt.sum((clean[:, :, None, :] - clean[:, None, :, :]) ** 2, dim=-1)
    local_sq = (clean_pair_sq + eye).min(dim=2).mean(dim=1).detach()
    temperature = jt.maximum(local_sq * float(alpha), 1e-8).reshape((-1, 1))

    def dcd_cost(source):
        # Matrix layout is (clean, source).  clean_to_source is the direction
        # that catches GT holes; source_to_clean protects surface fidelity.
        sq = jt.sum((clean[:, :, None, :] - source[:, None, :, :]) ** 2, dim=-1)
        # topk is used rather than min/argmin because Jittor exposes its
        # nearest value and index together through this stable API.
        clean_values, clean_indices = jt.topk(sq, k=1, dim=2, largest=False)
        source_values, source_indices = jt.topk(sq, k=1, dim=1, largest=False)
        clean_to_source = clean_values[:, :, 0]
        source_to_clean = source_values[:, 0, :]
        clean_idx = clean_indices[:, :, 0].int32()
        source_idx = source_indices[:, 0, :].int32()

        # one_hot+sum is the Jittor equivalent of PyTorch scatter_add for
        # nearest-neighbour multiplicities.  The counts are deliberately
        # detached: the hard assignments are a density statistic, while the
        # distance/exponential path remains differentiable to source points.
        source_count = jt.sum(jt.nn.one_hot(clean_idx, m), dim=1).float32().detach()
        clean_count = jt.sum(jt.nn.one_hot(source_idx, m), dim=1).float32().detach()
        clean_weights = jt.sum(
            jt.nn.one_hot(clean_idx, m).float32() * source_count.unsqueeze(1), dim=2
        ).detach()
        source_weights = jt.sum(
            jt.nn.one_hot(source_idx, m).float32() * clean_count.unsqueeze(1), dim=2
        ).detach()
        power = float(n_lambda)
        clean_weights = jt.maximum(clean_weights, 1.0) ** power
        source_weights = jt.maximum(source_weights, 1.0) ** power
        clean_term = 1.0 - jt.exp(-clean_to_source / temperature) / clean_weights
        source_term = 1.0 - jt.exp(-source_to_clean / temperature) / source_weights
        return 0.5 * (clean_term.mean() + source_term.mean())

    return dcd_cost(pred) / (dcd_cost(noisy).detach() + eps)


def balanced_assignment_collision_relative_loss(pred, clean, noisy, max_points=128,
                                                temperature_scale=0.35, eps=1e-8):
    """Soft balanced matching penalty for many-clean-to-one-pred collapse.

    For a small paired local patch, every clean point distributes one unit of
    mass over predictions with a distance-softmax.  A uniformly sampled clean
    and predicted patch should therefore have unit expected load per predicted
    point.  Penalising deviations of the column loads is a cheap, differentiable
    approximation to balanced OT, while retaining the normal CD terms for
    surface fidelity.  The noisy value is a detached denominator so the term
    is invariant to the sampled noise scale and uses no validation data.
    """
    m = min(int(max_points), int(pred.shape[1]), int(clean.shape[1]))
    if m < 2:
        return jt.zeros((), dtype=jt.float32)

    def cost(source):
        src = source[:, :m, :]
        tgt = clean[:, :m, :]
        sq = jt.sum((tgt[:, :, None, :] - src[:, None, :, :]) ** 2, dim=-1)
        # Use the clean patch's detached second-neighbour spacing as a local
        # temperature.  This avoids a global ShapeNet/category-dependent scale.
        eye = jt.array(np.eye(m, dtype=np.float32)).unsqueeze(0) * 1e6
        pair = jt.sum((tgt[:, :, None, :] - tgt[:, None, :, :]) ** 2, dim=-1)
        nn2 = jt.topk(pair + eye, k=2, dim=2, largest=False)[0][:, :, 1]
        scale = jt.maximum(nn2.mean(dim=1, keepdims=True).detach(), 1e-8)
        logits = -sq / (float(temperature_scale) * scale[:, None, None] + 1e-8)
        assignment = jt.nn.softmax(logits, dim=2)
        loads = assignment.sum(dim=1)
        return jt.mean((loads - 1.0) ** 2)

    return cost(pred) / (cost(noisy).detach() + eps)


def coverage_tail_relative_loss(pred, clean, noisy, max_points=256, tail_fraction=0.20,
                                eps=1e-8):
    """Relative clean->pred tail Chamfer objective for suppressing surface holes.

    The usual mean clean-to-pred term can hide a small set of uncovered GT
    points behind many easy matches.  We optimize the upper tail of nearest
    GT distances (rather than a hard global repulsion prior), so gradients are
    concentrated on the currently worst-covered surface regions.  The noisy
    tail is a detached denominator, preserving the noise-scale invariance of
    the other relative objectives.
    """
    m = min(int(max_points), int(pred.shape[1]), int(clean.shape[1]))
    if m < 2:
        return jt.zeros((), dtype=jt.float32)

    def tail_cost(source):
        sq = jt.sum((clean[:, :m, None, :] - source[:, None, :m, :]) ** 2, dim=-1)
        nearest = jt.topk(sq, k=1, dim=2, largest=False)[0][:, :, 0]
        k = max(1, min(m, int(round(float(m) * float(tail_fraction)))))
        worst = jt.topk(nearest, k=k, dim=1, largest=True)[0]
        return worst.mean()

    return tail_cost(pred) / (tail_cost(noisy).detach() + eps)


def soft_coverage_power_relative_loss(pred, clean, noisy, max_points=256, power=2.0,
                                      eps=1e-8):
    """Smoothly emphasize every under-covered clean target.

    Unlike the hard upper-tail objective, this keeps a gradient for all clean
    points and raises their nearest-neighbour residual to a power.  Thus a
    small hole cannot disappear when it falls just below a top-k cutoff, while
    a many-to-one cluster receives super-linear clean-to-pred pressure.  The
    noisy-input denominator makes the term dimensionless and keeps it useful
    across the sampled noise range.
    """
    m = min(int(max_points), int(pred.shape[1]), int(clean.shape[1]))
    if m < 2:
        return jt.zeros((), dtype=jt.float32)

    def cost(source):
        sq = jt.sum((clean[:, :m, None, :] - source[:, None, :m, :]) ** 2, dim=-1)
        nearest = jt.topk(sq, k=1, dim=2, largest=False)[0][:, :, 0]
        # Normalize by detached clean patch scale to avoid a category/patch
        # size dependent power gradient.
        clean_scale = jt.mean(jt.topk(
            jt.sum((clean[:, :m, None, :] - clean[:, None, :m, :]) ** 2, dim=-1),
            k=2, dim=2, largest=False)[0][:, :, 1], dim=1, keepdims=True).detach()
        ratio = nearest / (clean_scale + 1e-8)
        return jt.mean((ratio + 1e-6) ** float(power))

    return cost(pred) / (cost(noisy).detach() + eps)


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
    density = (
        density_consistency_loss(pred, clean)
        if getattr(args, "loss_density_weight", 0.0) > 0.0
        else jt.zeros((), dtype=jt.float32)
    )
    anti_cluster = (
        anti_cluster_knn_loss(
            pred,
            clean,
            k=getattr(args, "anti_cluster_k", 8),
            margin=getattr(args, "anti_cluster_margin", 0.85),
        )
        if getattr(args, "loss_anti_cluster_weight", 0.0) > 0.0
        else jt.zeros((), dtype=jt.float32)
    )
    tangent_spacing = jt.zeros((), dtype=jt.float32)
    sinkhorn_coverage = (
        sinkhorn_coverage_relative_loss(
            pred, clean, noisy,
            max_points=getattr(args, "sinkhorn_coverage_points", 128),
            iterations=getattr(args, "sinkhorn_coverage_iterations", 5),
            blur=getattr(args, "sinkhorn_coverage_blur", 2.0),
            eps=args.relative_eps,
        )
        if getattr(args, "loss_sinkhorn_coverage_weight", 0.0) > 0.0
        else jt.zeros((), dtype=jt.float32)
    )
    balanced_assignment = (
        balanced_assignment_collision_relative_loss(
            pred, clean, noisy,
            max_points=getattr(args, "balanced_assignment_points", 128),
            temperature_scale=getattr(args, "balanced_assignment_temperature", 0.35),
            eps=args.relative_eps,
        )
        if getattr(args, "loss_balanced_assignment_weight", 0.0) > 0.0
        else jt.zeros((), dtype=jt.float32)
    )
    density_aware_cd = (
        density_aware_chamfer_relative_loss(
            pred, clean, noisy,
            max_points=getattr(args, "density_aware_cd_points", 256),
            alpha=getattr(args, "density_aware_cd_alpha", 1.0),
            n_lambda=getattr(args, "density_aware_cd_lambda", 1.0),
            eps=args.relative_eps,
        )
        if getattr(args, "loss_density_aware_cd_weight", 0.0) > 0.0
        else jt.zeros((), dtype=jt.float32)
    )
    coverage_tail = (
        coverage_tail_relative_loss(
            pred, clean, noisy,
            max_points=getattr(args, "coverage_tail_points", 256),
            tail_fraction=getattr(args, "coverage_tail_fraction", 0.20),
            eps=args.relative_eps,
        )
        if getattr(args, "loss_coverage_tail_weight", 0.0) > 0.0
        else jt.zeros((), dtype=jt.float32)
    )
    soft_coverage_power = (
        soft_coverage_power_relative_loss(
            pred, clean, noisy,
            max_points=getattr(args, "soft_coverage_points", 256),
            power=getattr(args, "soft_coverage_power", 2.0),
            eps=args.relative_eps,
        )
        if getattr(args, "loss_soft_coverage_weight", 0.0) > 0.0
        else jt.zeros((), dtype=jt.float32)
    )
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
        + getattr(args, "loss_anti_cluster_weight", 0.0) * anti_cluster
        + getattr(args, "loss_sinkhorn_coverage_weight", 0.0) * sinkhorn_coverage
        + getattr(args, "loss_balanced_assignment_weight", 0.0) * balanced_assignment
        + getattr(args, "loss_density_aware_cd_weight", 0.0) * density_aware_cd
        + getattr(args, "loss_coverage_tail_weight", 0.0) * coverage_tail
        + getattr(args, "loss_soft_coverage_weight", 0.0) * soft_coverage_power
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
        "anti_cluster": anti_cluster,
        "tangent_spacing": tangent_spacing,
        "sinkhorn_coverage": sinkhorn_coverage,
        "balanced_assignment": balanced_assignment,
        "density_aware_cd": density_aware_cd,
        "coverage_tail": coverage_tail,
        "soft_coverage": soft_coverage_power,
        "stage": stage,
        "sigma": sigma_loss,
    }
    return total, metrics


def batched_gather_points(points, indices):
    """Gather (B,K,3) points from (B,N,3) using Jittor indexing."""
    rows = []
    for batch_index in range(points.shape[0]):
        rows.append(points[batch_index][indices[batch_index].int64()].unsqueeze(0))
    return jt.concat(rows, dim=0)


def pgd_training_loss(model, noisy, clean, noise_std, category_id, normals, args, overlap=None):
    flow_velocity_target = None
    flow_distance_target = None
    if getattr(args, "pgd_use_surface_flow", False):
        if not getattr(args, "pgd_surface_flow_train_intermediate", False):
            flow_distance_target = jt.ones((noisy.shape[0], 1, 1), dtype=jt.float32)
        else:
            batch_size = noisy.shape[0]
            max_time = float(getattr(args, "pgd_surface_flow_max_time", 0.8))
            start_prob = float(getattr(args, "pgd_surface_flow_start_prob", 0.5))
            time_np = np.random.uniform(0.0, max_time, size=(batch_size, 1, 1)).astype(np.float32)
            start_mask = np.random.uniform(0.0, 1.0, size=(batch_size, 1, 1)) < start_prob
            time_np[start_mask] = 0.0
            time = jt.array(time_np)
            flow_velocity_target = clean - noisy
            noisy = noisy + time * flow_velocity_target
            noise_std = noise_std * (1.0 - time.reshape((batch_size, 1)))
            flow_distance_target = 1.0 - time
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
            noisy=noisy,
            relative=getattr(args, "pgd_normal_corr_relative", False),
            eps=args.relative_eps,
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
    density = (
        density_consistency_loss(pred, clean)
        if getattr(args, "loss_density_weight", 0.0) > 0.0
        else jt.zeros((), dtype=jt.float32)
    )
    anti_cluster = (
        anti_cluster_knn_loss(
            pred,
            clean,
            k=getattr(args, "anti_cluster_k", 8),
            margin=getattr(args, "anti_cluster_margin", 0.85),
        )
        if getattr(args, "loss_anti_cluster_weight", 0.0) > 0.0
        else jt.zeros((), dtype=jt.float32)
    )
    tangent_spacing = (
        tangent_spacing_consistency_loss(
            pred,
            clean,
            normals,
            max_points=getattr(args, "tangent_spacing_points", 192),
            k=getattr(args, "tangent_spacing_k", 8),
            collapse_margin=getattr(args, "tangent_spacing_collapse_margin", 0.90),
        )
        if getattr(args, "loss_tangent_spacing_weight", 0.0) > 0.0
        else jt.zeros((), dtype=jt.float32)
    )
    local_surface = (
        local_surface_distance_loss(pred, clean)
        if getattr(args, "loss_local_surface_weight", 0.0) > 0.0
        else jt.zeros((), dtype=jt.float32)
    )
    sinkhorn_coverage = (
        sinkhorn_coverage_relative_loss(
            pred, clean, noisy,
            max_points=getattr(args, "sinkhorn_coverage_points", 128),
            iterations=getattr(args, "sinkhorn_coverage_iterations", 5),
            blur=getattr(args, "sinkhorn_coverage_blur", 2.0),
            eps=args.relative_eps,
        )
        if getattr(args, "loss_sinkhorn_coverage_weight", 0.0) > 0.0
        else jt.zeros((), dtype=jt.float32)
    )
    balanced_assignment = (
        balanced_assignment_collision_relative_loss(
            pred, clean, noisy,
            max_points=getattr(args, "balanced_assignment_points", 128),
            temperature_scale=getattr(args, "balanced_assignment_temperature", 0.35),
            eps=args.relative_eps,
        )
        if getattr(args, "loss_balanced_assignment_weight", 0.0) > 0.0
        else jt.zeros((), dtype=jt.float32)
    )
    density_aware_cd = (
        density_aware_chamfer_relative_loss(
            pred, clean, noisy,
            max_points=getattr(args, "density_aware_cd_points", 256),
            alpha=getattr(args, "density_aware_cd_alpha", 1.0),
            n_lambda=getattr(args, "density_aware_cd_lambda", 1.0),
            eps=args.relative_eps,
        )
        if getattr(args, "loss_density_aware_cd_weight", 0.0) > 0.0
        else jt.zeros((), dtype=jt.float32)
    )
    coverage_tail = (
        coverage_tail_relative_loss(
            pred, clean, noisy,
            max_points=getattr(args, "coverage_tail_points", 256),
            tail_fraction=getattr(args, "coverage_tail_fraction", 0.20),
            eps=args.relative_eps,
        )
        if getattr(args, "loss_coverage_tail_weight", 0.0) > 0.0
        else jt.zeros((), dtype=jt.float32)
    )
    soft_coverage_power = (
        soft_coverage_power_relative_loss(
            pred, clean, noisy,
            max_points=getattr(args, "soft_coverage_points", 256),
            power=getattr(args, "soft_coverage_power", 2.0),
            eps=args.relative_eps,
        )
        if getattr(args, "loss_soft_coverage_weight", 0.0) > 0.0
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
    stage2_plane = jt.zeros((), dtype=jt.float32)
    stage2_normal_residual = jt.zeros((), dtype=jt.float32)
    stage2_tangent_target = jt.zeros((), dtype=jt.float32)
    stage2_surface_requested = (
        float(getattr(args, "loss_stage2_plane_weight", 0.0)) > 0.0
        or float(getattr(args, "loss_stage2_normal_residual_weight", 0.0)) > 0.0
        or float(getattr(args, "loss_stage2_tangent_target_weight", 0.0)) > 0.0
    )
    if stage2_surface_requested:
        if normals is None:
            raise ValueError("stage-2 surface losses require training normals")
        stage2_plane, stage2_normal_residual, stage2_tangent_target = (
            stage2_surface_residual_losses(
                out, clean, normals, delta=args.corr_huber_delta
            )
        )
    disp_mag = jt.sqrt(jt.maximum(jt.sum(out["disp"] * out["disp"], dim=-1), 1e-12))

    low_noise_disp = jt.zeros((), dtype=jt.float32)
    surface_flow_velocity = jt.zeros((), dtype=jt.float32)
    surface_flow_distance = jt.zeros((), dtype=jt.float32)
    if getattr(args, "pgd_use_surface_flow", False):
        if out.get("flow_velocity") is None or out.get("flow_distance") is None:
            raise ValueError("--pgd_use_surface_flow requires PGD flow outputs")
        if flow_velocity_target is None:
            flow_velocity_target = clean - noisy
        velocity_error = (out["flow_velocity"] - flow_velocity_target) ** 2
        velocity_baseline = (flow_velocity_target ** 2).mean().detach()
        surface_flow_velocity = velocity_error.mean() / (
            velocity_baseline + args.relative_eps
        )
        surface_flow_distance = (
            (out["flow_distance"] - flow_distance_target) ** 2
        ).mean()
    surface_head_normal = jt.zeros((), dtype=jt.float32)
    surface_head_plane = jt.zeros((), dtype=jt.float32)
    if getattr(args, "pgd_use_surface_head", False):
        if normals is None:
            raise ValueError("--pgd_use_surface_head requires dataset normals")
        predicted_normal = out.get("surface_head_normal")
        if predicted_normal is None:
            raise ValueError("--pgd_use_surface_head requires PGD surface-head outputs")
        normal_dot = jt.sum(predicted_normal * normals, dim=-1)
        surface_head_normal = (1.0 - jt.abs(normal_dot)).mean()
        plane_error = jt.sum((pred - clean) * normals, dim=-1)
        noisy_plane_error = jt.sum((noisy - clean) * normals, dim=-1)
        surface_head_plane = (plane_error ** 2).mean() / (
            (noisy_plane_error ** 2).mean().detach() + args.relative_eps
        )
    surface_vector_residual = jt.zeros((), dtype=jt.float32)
    surface_vector_tangent = jt.zeros((), dtype=jt.float32)
    if getattr(args, "pgd_use_surface_vector_head", False):
        if normals is None:
            raise ValueError("--pgd_use_surface_vector_head requires dataset normals")
        correction = out.get("surface_vector_correction")
        base_disp = out.get("surface_base_disp")
        if correction is None or base_disp is None:
            raise ValueError("surface-vector head outputs are missing")
        base_pred = noisy + base_disp
        target_clean = clean
        target_normals = normals
        if getattr(args, "pgd_surface_vector_nearest_target", False):
            # Approximate the oracle's current-prediction -> closest-surface
            # vector with the nearest clean training sample.  Projecting onto
            # that sample's normal removes finite-sampling tangent jitter.
            sq_dist = jt.sum(
                (base_pred[:, :, None, :] - clean[:, None, :, :]) ** 2,
                dim=-1,
            )
            nearest = jt.argmin(sq_dist, dim=2)
            batch_index = jt.arange(base_pred.shape[0]).reshape(
                (base_pred.shape[0], 1)
            ).broadcast((base_pred.shape[0], base_pred.shape[1]))
            target_clean = clean[batch_index, nearest].detach()
            target_normals = normals[batch_index, nearest].detach()
        remaining = target_clean - base_pred
        normal_target = (
            jt.sum(remaining * target_normals, dim=-1, keepdims=True)
            * target_normals
        )
        target_scale = (normal_target ** 2).mean().detach()
        surface_vector_residual = ((correction - normal_target) ** 2).mean() / (
            target_scale + args.relative_eps
        )
        tangent_correction = correction - (
            jt.sum(correction * normals, dim=-1, keepdims=True) * normals
        )
        surface_vector_tangent = (tangent_correction ** 2).mean() / (
            target_scale + args.relative_eps
        )

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

    overlap_consistency = jt.zeros((), dtype=jt.float32)
    overlap_consistency_weight = float(getattr(args, "pgd_overlap_consistency_weight", 0.0))
    if overlap_consistency_weight > 0.0:
        if overlap is None:
            raise ValueError("overlap consistency requested but overlap patches were not provided")
        overlap_noisy, overlap_a, overlap_b, overlap_mask = overlap
        overlap_out = model(overlap_noisy, noise_std=noise_std, category_id=category_id, return_dict=True)
        stage2_only = bool(getattr(args, "pgd_overlap_stage2_only", False))
        if stage2_only and out.get("disp2_normal") is not None:
            components = (
                (out["disp2_normal"], overlap_out["disp2_normal"]),
                (out["disp2_tangent"], overlap_out["disp2_tangent"]),
            )
        else:
            components = ((out["disp"], overlap_out["disp"]),)
        delta = float(args.corr_huber_delta)
        component_losses = []
        for first, second in components:
            disp_a = batched_gather_points(first, overlap_a)
            disp_b = batched_gather_points(second, overlap_b)
            diff = disp_a - disp_b
            if getattr(args, "pgd_overlap_consistency_normalize", False):
                scale = (
                    0.5
                    * (
                        jt.norm(disp_a, dim=-1).mean()
                        + jt.norm(disp_b, dim=-1).mean()
                    ).detach()
                    + 1e-5
                )
                diff = diff / scale
                component_delta = 0.25
            else:
                component_delta = delta
            abs_diff = jt.abs(diff)
            per_coord = jt.where(
                abs_diff <= component_delta,
                0.5 * abs_diff * abs_diff / component_delta,
                abs_diff - 0.5 * component_delta,
            )
            per_point = per_coord.mean(dim=-1)
            component_losses.append(
                jt.sum(per_point * overlap_mask)
                / jt.maximum(jt.sum(overlap_mask), 1.0)
            )
        overlap_consistency = sum(component_losses) / float(len(component_losses))

    total = (
        args.loss_corr_weight * corr
        + args.loss_relative_weight * relative
        + getattr(args, "loss_pred_cd_weight", 0.0) * pred_cd_relative
        + getattr(args, "loss_clean_cd_weight", 0.0) * clean_cd_relative
        + args.loss_infocd_weight * info
        + args.loss_uniform_weight * uniform
        + getattr(args, "loss_density_weight", 0.0) * density
        + getattr(args, "loss_anti_cluster_weight", 0.0) * anti_cluster
        + getattr(args, "loss_tangent_spacing_weight", 0.0) * tangent_spacing
        + getattr(args, "loss_local_surface_weight", 0.0) * local_surface
        + getattr(args, "loss_sinkhorn_coverage_weight", 0.0) * sinkhorn_coverage
        + getattr(args, "loss_balanced_assignment_weight", 0.0) * balanced_assignment
        + getattr(args, "loss_density_aware_cd_weight", 0.0) * density_aware_cd
        + getattr(args, "loss_coverage_tail_weight", 0.0) * coverage_tail
        + getattr(args, "loss_soft_coverage_weight", 0.0) * soft_coverage_power
        + getattr(args, "loss_score_relative_weight", 0.0) * score_relative
        + args.loss_straight_weight * straight
        + args.loss_stage_weight * stage
        + getattr(args, "loss_stage2_plane_weight", 0.0) * stage2_plane
        + getattr(args, "loss_stage2_normal_residual_weight", 0.0) * stage2_normal_residual
        + getattr(args, "loss_stage2_tangent_target_weight", 0.0) * stage2_tangent_target
        + args.pgd_loss_disp_weight * disp_mag.mean()
        + args.pgd_loss_low_noise_disp_weight * low_noise_disp
        + rotation_consistency_weight * rotation_consistency
        + overlap_consistency_weight * overlap_consistency
        + getattr(args, "pgd_surface_flow_velocity_weight", 0.0) * surface_flow_velocity
        + getattr(args, "pgd_surface_flow_distance_weight", 0.0) * surface_flow_distance
        + getattr(args, "pgd_surface_head_normal_weight", 0.0) * surface_head_normal
        + getattr(args, "pgd_surface_head_plane_weight", 0.0) * surface_head_plane
        + getattr(args, "pgd_surface_vector_residual_weight", 0.0) * surface_vector_residual
        + getattr(args, "pgd_surface_vector_tangent_weight", 0.0) * surface_vector_tangent
    )
    metrics = {
        "corr": corr,
        "relative": relative,
        "pred_cd": pred_cd_relative,
        "clean_cd": clean_cd_relative,
        "infocd": info,
        "uniform": uniform,
        "density": density,
        "anti_cluster": anti_cluster,
        "tangent_spacing": tangent_spacing,
        "local_surface": local_surface,
        "sinkhorn_coverage": sinkhorn_coverage,
        "balanced_assignment": balanced_assignment,
        "density_aware_cd": density_aware_cd,
        "coverage_tail": coverage_tail,
        "soft_coverage": soft_coverage_power,
        "score_relative": score_relative,
        "straight": straight,
        "straight_dir": straight_direction,
        "straight_dist": straight_distance,
        "stage": stage,
        "stage2_plane": stage2_plane,
        "stage2_normal": stage2_normal_residual,
        "stage2_tangent": stage2_tangent_target,
        "disp": disp_mag.mean(),
        "low_disp": low_noise_disp,
        "rot_consistency": rotation_consistency,
        "overlap_consistency": overlap_consistency,
        "flow_velocity": surface_flow_velocity,
        "flow_distance_loss": surface_flow_distance,
        "surface_head_normal": surface_head_normal,
        "surface_head_plane": surface_head_plane,
        "surface_vector_residual": surface_vector_residual,
        "surface_vector_tangent": surface_vector_tangent,
        "flow_distance": (
            out["flow_distance"].mean()
            if out.get("flow_distance") is not None
            else jt.ones((1,), dtype=jt.float32).mean()
        ),
        "noise_gate1": out.get(
            "stage1_noise_gate", jt.ones((1,), dtype=jt.float32)
        ).mean(),
        "noise_gate2": out.get(
            "stage2_noise_gate", jt.ones((1,), dtype=jt.float32)
        ).mean(),
        "dual_normal_gate": (
            out["stage2_normal_gate"].mean()
            if out.get("stage2_normal_gate") is not None
            else jt.ones((1,), dtype=jt.float32).mean()
        ),
        "dual_tangent_gate": (
            out["stage2_tangent_gate"].mean()
            if out.get("stage2_tangent_gate") is not None
            else jt.ones((1,), dtype=jt.float32).mean()
        ),
    }
    return total, metrics


def sanitize_optimizer_grads(optimizer):
    for pg in optimizer.param_groups:
        for grad in pg["grads"]:
            grad.update(jt.where(jt.isfinite(grad), grad, jt.zeros_like(grad)))


def manual_dist_sync_grads(optimizer):
    """Average optimizer gradients through the local rendezvous server."""
    if MANUAL_DIST is None:
        return
    arrays = []
    shapes = []
    for group in optimizer.param_groups:
        for param, grad in zip(group["params"], group["grads"]):
            if param.is_stop_grad():
                continue
            array = np.asarray(grad.numpy(), dtype=np.float32)
            arrays.append(array.reshape(-1))
            shapes.append(array.shape)
    flat = np.concatenate(arrays, axis=0) if arrays else np.empty((0,), dtype=np.float32)
    averaged = MANUAL_DIST.allreduce_mean(flat)
    offset = 0
    shape_index = 0
    for group in optimizer.param_groups:
        for param, grad in zip(group["params"], group["grads"]):
            if param.is_stop_grad():
                continue
            shape = shapes[shape_index]
            size = int(np.prod(shape))
            grad.update(jt.array(averaged[offset:offset + size].reshape(shape)))
            offset += size
            shape_index += 1


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
            "pgd_train_stage2_dual_gate_only",
            "pgd_train_noise_condition_only",
            "pgd_train_surface_flow_distance_only",
            "pgd_train_surface_flow_head_only",
            "pgd_train_surface_head_only",
            "pgd_train_surface_vector_only",
            "pgd_train_separate_stage2_only",
            "pgd_train_separate_stage2_head_only",
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
    if getattr(args, "pgd_train_surface_flow_head_only", False):
        allowed = (
            "feature_nets.linear0_1.",
            "feature_nets.linear0_2.",
            "feature_nets.linear0_3.",
            "surface_flow_distance_fc1.",
            "surface_flow_distance_fc2.",
        )
        selected = []
        names = []
        for name, param in model.named_parameters():
            if name.startswith(allowed):
                if name.endswith(".running_mean") or name.endswith(".running_var"):
                    continue
                selected.append(param)
                names.append(name)
        if not selected or not hasattr(model, "surface_flow_distance_fc2"):
            raise ValueError(
                "--pgd_train_surface_flow_head_only requires --pgd_use_surface_flow"
            )
        return selected, names
    if getattr(args, "pgd_train_surface_flow_distance_only", False):
        allowed = (
            "surface_flow_distance_fc1.",
            "surface_flow_distance_fc2.",
        )
        selected = []
        names = []
        for name, param in model.named_parameters():
            if name.startswith(allowed):
                selected.append(param)
                names.append(name)
        if not selected or not hasattr(model, "surface_flow_distance_fc2"):
            raise ValueError(
                "--pgd_train_surface_flow_distance_only requires --pgd_use_surface_flow"
            )
        return selected, names
    if getattr(args, "pgd_train_surface_head_only", False):
        allowed = (
            "surface_head_fc1.",
            "surface_head_normal_fc2.",
            "surface_head_distance_fc2.",
        )
        selected = []
        names = []
        for name, param in model.named_parameters():
            if name.startswith(allowed):
                selected.append(param)
                names.append(name)
        if not selected or not hasattr(model, "surface_head_distance_fc2"):
            raise ValueError(
                "--pgd_train_surface_head_only requires --pgd_use_surface_head"
            )
        return selected, names
    if getattr(args, "pgd_train_surface_vector_only", False):
        allowed = ("surface_vector_fc1.", "surface_vector_fc2.")
        selected = []
        names = []
        for name, param in model.named_parameters():
            if name.startswith(allowed):
                selected.append(param)
                names.append(name)
        if not selected or not hasattr(model, "surface_vector_fc2"):
            raise ValueError(
                "--pgd_train_surface_vector_only requires --pgd_use_surface_vector_head"
            )
        return selected, names
    if getattr(args, "pgd_train_separate_stage2_only", False):
        selected = []
        names = []
        for name, param in model.named_parameters():
            if not name.startswith("feature_nets_stage2."):
                continue
            if name.endswith(".running_mean") or name.endswith(".running_var"):
                continue
            selected.append(param)
            names.append(name)
        if not selected or not hasattr(model, "feature_nets_stage2"):
            raise ValueError(
                "--pgd_train_separate_stage2_only requires --pgd_use_separate_stage2"
            )
        return selected, names
    if getattr(args, "pgd_train_separate_stage2_head_only", False):
        allowed = (
            "feature_nets_stage2.linear0_1.",
            "feature_nets_stage2.linear0_2.",
            "feature_nets_stage2.linear0_3.",
        )
        selected = []
        names = []
        for name, param in model.named_parameters():
            if name.startswith(allowed):
                selected.append(param)
                names.append(name)
        if not selected or not hasattr(model, "feature_nets_stage2"):
            raise ValueError(
                "--pgd_train_separate_stage2_head_only requires --pgd_use_separate_stage2"
            )
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
    if getattr(args, "pgd_train_stage2_dual_gate_only", False):
        allowed = ("stage2_dual_gate_fc1.", "stage2_dual_gate_fc2.")
        selected = []
        names = []
        for name, param in model.named_parameters():
            if name.startswith(allowed):
                selected.append(param)
                names.append(name)
        if not selected:
            raise ValueError(
                "--pgd_train_stage2_dual_gate_only requires --pgd_use_stage2_dual_gate"
            )
        return selected, names
    if getattr(args, "pgd_train_noise_condition_only", False):
        allowed = ("noise_condition_fc1.", "noise_condition_fc2.")
        selected = []
        names = []
        for name, param in model.named_parameters():
            if name.startswith(allowed):
                selected.append(param)
                names.append(name)
        if not selected:
            raise ValueError(
                "--pgd_train_noise_condition_only requires --pgd_use_noise_conditioning"
            )
        return selected, names
    return model.parameters(), []


def get_loss_fn(name):
    if name == "infocd":
        return calc_cd_like_InfoV2
    if name == "chamfer":
        return chamfer_loss
    raise ValueError("unsupported loss: {}".format(name))


def stack_batch(samples, has_normals=False, has_overlap=False):
    noisy = jt.array(np.stack([s[0] for s in samples], axis=0))
    clean = jt.array(np.stack([s[1] for s in samples], axis=0))
    noise_std = jt.array(np.asarray([[s[3]] for s in samples], dtype=np.float32))
    category_id = jt.array(np.asarray([s[4] for s in samples], dtype=np.int32))
    cursor = 5
    normals = None
    if has_normals:
        normals = jt.array(np.stack([s[cursor] for s in samples], axis=0))
        cursor += 1
    overlap = None
    if has_overlap:
        overlap = (
            jt.array(np.stack([s[cursor][0] for s in samples], axis=0)),
            jt.array(np.stack([s[cursor][1] for s in samples], axis=0)).int64(),
            jt.array(np.stack([s[cursor][2] for s in samples], axis=0)).int64(),
            jt.array(np.stack([s[cursor][3] for s in samples], axis=0)),
        )
    return noisy, clean, noise_std, category_id, normals, overlap


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
        if getattr(args, "pgd_second_stage_tangent_only", False):
            base_cmd.extend([
                "--pgd_second_stage_tangent_only",
                "--pgd_second_stage_tangent_scale", str(args.pgd_second_stage_tangent_scale),
                "--pgd_second_stage_normal_scale", str(args.pgd_second_stage_normal_scale),
                "--pgd_second_stage_surface_k", str(args.pgd_second_stage_surface_k),
            ])
        if getattr(args, "pgd_use_stage2_dual_gate", False):
            base_cmd.extend([
                "--pgd_use_stage2_dual_gate",
                "--pgd_stage2_dual_gate_scale", str(args.pgd_stage2_dual_gate_scale),
                "--pgd_second_stage_surface_k", str(args.pgd_second_stage_surface_k),
            ])
    if args.pgd_use_refine_gate:
        base_cmd.extend([
            "--pgd_use_refine_gate",
            "--pgd_refine_gate_scale", str(args.pgd_refine_gate_scale),
            "--pgd_gate_noise_source", args.eval_pgd_gate_noise_source,
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
            # Evaluation shards are independent Jittor processes. They need
            # one cache per shard, but must reuse that cache across epochs:
            # a fresh cache forces Jittor 1.3.11 to rebuild cutt under CUDA
            # 12.8, whose first-build path is incompatible with this setup.
            env["cache_name"] = "pgd_cuda_eval_epoch1_shard{:02d}".format(shard)
            env["use_parallel_op_compiler"] = os.environ.get("use_parallel_op_compiler", "1")
            env["disable_lock"] = os.environ.get("disable_lock", "1")
            # libcutt is compiled on Jittor import. With several evaluation
            # shards starting together, Jittor's internal multiprocessing can
            # race while building this shared external dependency.
            env["DISABLE_MULTIPROCESSING"] = os.environ.get("DISABLE_MULTIPROCESSING", "1")
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


def learning_rate_for_step(args, global_step):
    """Return the optimizer learning rate for a zero-based train step."""
    if args.lr_schedule == "constant":
        return float(args.lr)
    if args.lr_schedule != "warmup_cosine":
        raise ValueError("unknown lr schedule: {}".format(args.lr_schedule))
    total_steps = int(getattr(args, "lr_schedule_total_steps", 0))
    if total_steps <= 0:
        total_steps = int(args.epochs) * int(args.train_steps_per_epoch)
    if total_steps <= 0:
        raise ValueError("warmup_cosine requires --train_steps_per_epoch > 0")
    warmup_steps = min(max(0, int(args.lr_warmup_steps)), total_steps)
    peak_lr = float(args.lr)
    start_lr = float(args.lr_warmup_start)
    min_lr = float(args.lr_min)
    if warmup_steps > 0 and global_step < warmup_steps:
        # The first update uses start_lr; the last warmup update reaches peak_lr.
        if warmup_steps == 1:
            return peak_lr
        ratio = float(global_step) / float(warmup_steps - 1)
        return start_lr + (peak_lr - start_lr) * ratio
    decay_steps = max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, float(global_step - warmup_steps) / float(max(1, decay_steps - 1))))
    return min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def set_optimizer_lr(optimizer, lr):
    """Keep Jittor's optimizer and any explicit parameter groups in sync."""
    lr = float(lr)
    optimizer.lr = lr
    for group in optimizer.param_groups:
        if "lr" in group:
            group["lr"] = lr


def run_epoch(model, dataset, optimizer, batch_size, loss_fn, args, train=True, desc="train", max_steps=0,
              global_train_step_offset=0):
    indices = list(range(len(dataset)))
    if train:
        random.shuffle(indices)
        model.train()
        if getattr(args, "freeze_batchnorm_stats", False):
            freeze_batchnorm_stats(model)
    else:
        model.eval()
    if MANUAL_DIST is not None:
        indices = indices[MANUAL_DIST.rank::MANUAL_DIST.world_size]
    elif jt.in_mpi:
        indices = indices[jt.rank::jt.world_size]
    losses = []
    if MANUAL_DIST is not None or jt.in_mpi:
        # Jittor's optimizer synchronizes gradients every step. Keep every rank
        # on the same number of full batches so the last all-reduce cannot wait
        # for a rank that has already left the loop.
        total = len(indices) // batch_size
    else:
        total = int(math.ceil(len(indices) / float(batch_size)))
    if max_steps > 0:
        total = min(total, max_steps)
    disable_pbar = (MANUAL_DIST is not None and MANUAL_DIST.rank != 0) or (jt.in_mpi and jt.rank != 0)
    pbar = tqdm(range(total), desc=desc, disable=disable_pbar)
    for step in pbar:
        batch_idx = indices[step * batch_size:(step + 1) * batch_size]
        if not batch_idx:
            break
        samples = [dataset[i] for i in batch_idx]
        has_overlap = args.model == "pgd" and float(getattr(args, "pgd_overlap_consistency_weight", 0.0)) > 0.0
        has_normals = args.model == "pgd" and (
            args.pgd_use_normal_corr_loss
            or float(getattr(args, "loss_tangent_spacing_weight", 0.0)) > 0.0
            or float(getattr(args, "loss_stage2_plane_weight", 0.0)) > 0.0
            or float(getattr(args, "loss_stage2_normal_residual_weight", 0.0)) > 0.0
            or float(getattr(args, "loss_stage2_tangent_target_weight", 0.0)) > 0.0
            or bool(getattr(args, "pgd_use_surface_head", False))
            or bool(getattr(args, "pgd_use_surface_vector_head", False))
        )
        noisy, clean, noise_std, category_id, normals, overlap = stack_batch(
            samples,
            has_normals=has_normals,
            has_overlap=has_overlap,
        )
        if args.model == "asdn":
            loss, metrics = asdn_training_loss(model, noisy, clean, noise_std, category_id, args)
        elif args.pgd_composite_loss:
            loss, metrics = pgd_training_loss(model, noisy, clean, noise_std, category_id, normals, args, overlap=overlap)
        else:
            pred = noisy + model(noisy, noise_std=noise_std, category_id=category_id)
            loss = loss_fn(pred, clean)
            # Keep the submitted plain-InfoCD objective as the base path.  An
            # optional multiplicity-aware term can be added here without
            # silently switching to the much broader composite objective.
            # This specifically penalizes many clean points collapsing onto
            # one prediction, i.e. the visible clumping/GT->pred CD failure.
            plain_dcd_weight = float(getattr(args, "pgd_plain_infocd_dcd_weight", 0.0))
            if plain_dcd_weight > 0.0:
                plain_dcd = density_aware_chamfer_relative_loss(
                    pred,
                    clean,
                    noisy,
                    max_points=getattr(args, "density_aware_cd_points", 256),
                    alpha=getattr(args, "density_aware_cd_alpha", 1.0),
                    n_lambda=getattr(args, "density_aware_cd_lambda", 1.0),
                    eps=args.relative_eps,
                )
                loss = loss + plain_dcd_weight * plain_dcd
                # The progress reporter expects a ``corr`` field for every
                # PGD path.  Plain InfoCD has no correspondence term, so use
                # an explicit zero rather than changing the objective.
                metrics = {"corr": jt.zeros((), dtype=jt.float32), "plain_dcd": plain_dcd}
            else:
                metrics = {"corr": jt.zeros((), dtype=jt.float32)}
        val = float(loss.numpy())
        if not np.isfinite(val):
            pbar.set_postfix(loss="nonfinite")
            continue
        if train:
            current_lr = learning_rate_for_step(args, global_train_step_offset + step)
            set_optimizer_lr(optimizer, current_lr)
            if MANUAL_DIST is not None:
                optimizer.zero_grad()
                optimizer.backward(loss)
                manual_dist_sync_grads(optimizer)
                sanitize_optimizer_grads(optimizer)
                if args.grad_clip_norm > 0:
                    optimizer.clip_grad_norm(args.grad_clip_norm)
                optimizer.step()
            elif args.grad_clip_norm > 0:
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
        is_main = (MANUAL_DIST is None or MANUAL_DIST.rank == 0) and ((not jt.in_mpi) or jt.rank == 0)
        if train and is_main and every > 0 and (step % every == 0 or step + 1 == total):
            payload = {"step": int(step), "global_step": int(global_train_step_offset + step), "loss": val,
                       "lr": float(current_lr)}
            for key, metric in metrics.items():
                payload[key] = float(metric.numpy())
            with open(Path(args.log_dir) / "train_steps.jsonl", "a") as f:
                f.write(json.dumps(payload, sort_keys=True) + "\n")
        if metrics:
            pbar.set_postfix(loss="{:.6f}".format(val), corr="{:.5f}".format(float(metrics["corr"].numpy())))
        else:
            pbar.set_postfix(loss="{:.6f}".format(val))
        if train:
            # Large 50k-point patches create multi-GB temporary fused
            # buffers. Drop the Python references first; otherwise jt.clean()
            # cannot reclaim the current graph and the allocator grows until
            # a later backward pass OOMs even when the per-step batch fits.
            del loss, noisy, clean, noise_std, category_id, normals
            del samples, batch_idx
            if "pred" in locals():
                del pred
            if "metrics" in locals():
                del metrics
            # Jittor launches CUDA work asynchronously. Synchronize before
            # collecting temporaries, otherwise buffers still referenced by
            # pending kernels accumulate across steps despite jt.clean().
            jt.sync_all()
            jt.clean()
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
    parser.add_argument("--lr_schedule", choices=["constant", "warmup_cosine"], default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--lr_warmup_start", type=float, default=0.0)
    parser.add_argument("--lr_min", type=float, default=0.0)
    parser.add_argument("--lr_schedule_total_steps", type=int, default=0,
                        help="override total schedule steps when resuming without optimizer state (0=epochs*steps)")
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
    parser.add_argument("--pgd_train_stage2_dual_gate_only", action="store_true")
    parser.add_argument("--pgd_train_noise_condition_only", action="store_true")
    parser.add_argument("--pgd_train_surface_flow_distance_only", action="store_true")
    parser.add_argument("--pgd_train_surface_flow_head_only", action="store_true")
    parser.add_argument("--pgd_train_surface_head_only", action="store_true")
    parser.add_argument("--pgd_train_surface_vector_only", action="store_true")
    parser.add_argument("--pgd_train_separate_stage2_only", action="store_true")
    parser.add_argument("--pgd_train_separate_stage2_head_only", action="store_true")
    parser.add_argument("--pgd_train_head_only", action="store_true")
    parser.add_argument("--pgd_train_decoder_head_only", action="store_true")
    parser.add_argument("--pgd_train_decoder_last_encoder", action="store_true")
    parser.add_argument("--pgd_two_stage", action="store_true")
    parser.add_argument("--pgd_use_separate_stage2", action="store_true",
                        help="use a checkpoint-initialized non-shared backbone for stage 2")
    parser.add_argument("--pgd_second_stage_scale", type=float, default=1.0)
    parser.add_argument("--pgd_second_stage_tangent_only", action="store_true",
                        help="decompose stage-2 displacement into tangent/normal PCA components")
    parser.add_argument("--pgd_second_stage_tangent_scale", type=float, default=1.0)
    parser.add_argument("--pgd_second_stage_normal_scale", type=float, default=0.15)
    parser.add_argument("--pgd_second_stage_surface_k", type=int, default=16)
    parser.add_argument("--pgd_use_stage2_dual_gate", action="store_true")
    parser.add_argument("--pgd_stage2_dual_gate_scale", type=float, default=0.90)
    parser.add_argument("--pgd_use_refine_gate", action="store_true")
    parser.add_argument("--pgd_refine_gate_scale", type=float, default=0.25)
    parser.add_argument("--pgd_use_noise_conditioning", action="store_true")
    parser.add_argument("--pgd_noise_condition_hidden_dim", type=int, default=16)
    parser.add_argument("--pgd_noise_condition_scale", type=float, default=0.50)
    parser.add_argument("--pgd_noise_condition_min", type=float, default=0.005)
    parser.add_argument("--pgd_noise_condition_max", type=float, default=0.020)
    parser.add_argument("--pgd_train_detach_second_stage_backbone", action="store_true")
    parser.add_argument("--pgd_train_detach_second_stage_features", action="store_true",
                        help="detach stage-2 decoded features but train its displacement head")
    parser.add_argument("--pgd_use_surface_flow", action="store_true",
                        help="StraightPCF-style velocity field times learned patch distance")
    parser.add_argument("--pgd_surface_flow_hidden_dim", type=int, default=32)
    parser.add_argument("--pgd_surface_flow_log_scale_min", type=float, default=-2.0)
    parser.add_argument("--pgd_surface_flow_log_scale_max", type=float, default=0.4)
    parser.add_argument("--pgd_surface_flow_train_intermediate", action="store_true",
                        help="train on random noisy-to-clean interpolation states")
    parser.add_argument("--pgd_surface_flow_max_time", type=float, default=0.8)
    parser.add_argument("--pgd_surface_flow_start_prob", type=float, default=0.5)
    parser.add_argument("--pgd_surface_flow_velocity_weight", type=float, default=0.25)
    parser.add_argument("--pgd_surface_flow_distance_weight", type=float, default=0.25)
    parser.add_argument("--pgd_use_surface_head", action="store_true",
                        help="learn a per-point normal and signed residual surface distance")
    parser.add_argument("--pgd_surface_head_hidden_dim", type=int, default=64)
    parser.add_argument("--pgd_surface_head_max_distance", type=float, default=0.02)
    parser.add_argument("--pgd_surface_head_normal_weight", type=float, default=0.10)
    parser.add_argument("--pgd_surface_head_plane_weight", type=float, default=0.25)
    parser.add_argument("--pgd_use_surface_vector_head", action="store_true",
                        help="learn a directly supervised residual surface-correction vector")
    parser.add_argument("--pgd_surface_vector_hidden_dim", type=int, default=64)
    parser.add_argument("--pgd_surface_vector_max_distance", type=float, default=0.02)
    parser.add_argument("--pgd_surface_vector_unit_slope", action="store_true",
                        help="use a bounded vector output with unit derivative at zero")
    parser.add_argument("--pgd_surface_vector_nearest_target", action="store_true",
                        help="supervise with the closest clean point's normal residual")
    parser.add_argument("--pgd_surface_vector_residual_weight", type=float, default=1.0)
    parser.add_argument("--pgd_surface_vector_tangent_weight", type=float, default=0.10)
    parser.add_argument("--pgd_composite_loss", action="store_true")
    parser.add_argument("--pgd_plain_infocd_dcd_weight", type=float, default=0.0,
                        help="optional DCD multiplicity coverage term added to plain InfoCD only")
    parser.add_argument("--pgd_use_normal_corr_loss", action="store_true")
    parser.add_argument("--pgd_normal_corr_relative", action="store_true",
                        help="normalize paired normal/tangent error by each sample's noisy baseline")
    parser.add_argument("--normal_k", type=int, default=16)
    parser.add_argument("--normal_corr_normal_weight", type=float, default=2.0,
                        help="training-only weight for normal correction error (P2S-oriented)")
    parser.add_argument("--normal_corr_tangent_weight", type=float, default=1.0,
                        help="training-only weight for tangent correction error (CD-oriented)")
    parser.add_argument("--pgd_loss_disp_weight", type=float, default=0.0)
    parser.add_argument("--pgd_loss_low_noise_disp_weight", type=float, default=0.0)
    parser.add_argument("--pgd_rotation_consistency_weight", type=float, default=0.0,
                        help="training-only yaw equivariance consistency weight")
    parser.add_argument("--pgd_overlap_consistency_weight", type=float, default=0.0,
                        help="train-only displacement agreement for shared points in overlapping patches")
    parser.add_argument("--pgd_overlap_stage2_only", action="store_true",
                        help="apply overlap agreement separately to stage-2 normal/tangent residuals")
    parser.add_argument("--pgd_overlap_consistency_normalize", action="store_true",
                        help="normalize overlap differences by stage-2 displacement magnitude")
    parser.add_argument("--loss_corr_weight", type=float, default=1.0)
    parser.add_argument("--loss_relative_weight", type=float, default=0.5)
    parser.add_argument("--loss_pred_cd_weight", type=float, default=0.0)
    parser.add_argument("--loss_clean_cd_weight", type=float, default=0.0)
    parser.add_argument("--loss_infocd_weight", type=float, default=0.15)
    parser.add_argument("--loss_uniform_weight", type=float, default=0.10)
    parser.add_argument("--loss_density_weight", type=float, default=0.0,
                        help="train-only local spacing consistency weight")
    parser.add_argument("--loss_anti_cluster_weight", type=float, default=0.0,
                        help="train-only one-sided kNN collapse penalty")
    parser.add_argument("--anti_cluster_k", type=int, default=8,
                        help="number of local clean/pred neighbours for anti-cluster loss")
    parser.add_argument("--anti_cluster_margin", type=float, default=0.85,
                        help="pred/clean spacing ratio below which anti-cluster loss activates")
    parser.add_argument("--loss_tangent_spacing_weight", type=float, default=0.0,
                        help="training-only tangent-plane local-spacing consistency weight")
    parser.add_argument("--tangent_spacing_points", type=int, default=192,
                        help="local point count for tangent-plane spacing loss")
    parser.add_argument("--tangent_spacing_k", type=int, default=8,
                        help="number of tangent-plane neighbours to match")
    parser.add_argument("--tangent_spacing_collapse_margin", type=float, default=0.90,
                        help="pred/clean tangent spacing ratio below which collapse is penalized")
    parser.add_argument("--loss_local_surface_weight", type=float, default=0.0,
                        help="train-only local pair-distance geometry weight")
    parser.add_argument("--loss_sinkhorn_coverage_weight", type=float, default=0.0,
                        help="train-only balanced local OT coverage weight (clean-to-pred/CD-oriented)")
    parser.add_argument("--sinkhorn_coverage_points", type=int, default=128,
                        help="local point count used by the balanced coverage OT loss")
    parser.add_argument("--sinkhorn_coverage_iterations", type=int, default=5,
                        help="Sinkhorn scaling iterations for the balanced coverage loss")
    parser.add_argument("--sinkhorn_coverage_blur", type=float, default=2.0,
                        help="OT entropy scale in units of clean squared nearest-neighbour spacing")
    parser.add_argument("--loss_balanced_assignment_weight", type=float, default=0.0,
                        help="train-only soft balanced assignment collision penalty")
    parser.add_argument("--balanced_assignment_points", type=int, default=128,
                        help="local point count used by the soft assignment collision loss")
    parser.add_argument("--balanced_assignment_temperature", type=float, default=0.35,
                        help="temperature in local clean-NN-spacing units for soft assignments")
    parser.add_argument("--loss_density_aware_cd_weight", type=float, default=0.0,
                        help="train-only DCD multiplicity term for clustered many-to-one matches")
    parser.add_argument("--density_aware_cd_points", type=int, default=256,
                        help="local point count used by the DCD multiplicity term")
    parser.add_argument("--density_aware_cd_alpha", type=float, default=1.0,
                        help="DCD exponential sharpness in clean-NN-spacing units")
    parser.add_argument("--density_aware_cd_lambda", type=float, default=1.0,
                        help="DCD nearest-neighbour multiplicity exponent")
    parser.add_argument("--loss_coverage_tail_weight", type=float, default=0.0,
                        help="train-only upper-tail clean-to-pred coverage weight")
    parser.add_argument("--coverage_tail_points", type=int, default=256,
                        help="local point count used by the coverage-tail term")
    parser.add_argument("--coverage_tail_fraction", type=float, default=0.20,
                        help="fraction of worst-covered GT points emphasized")
    parser.add_argument("--loss_soft_coverage_weight", type=float, default=0.0,
                        help="train-only smooth power-weighted clean-to-pred coverage")
    parser.add_argument("--soft_coverage_points", type=int, default=256,
                        help="local point count used by smooth coverage term")
    parser.add_argument("--soft_coverage_power", type=float, default=2.0,
                        help="power applied to normalized clean-to-pred residuals")
    parser.add_argument("--loss_score_relative_weight", type=float, default=0.0,
                        help="train-only per-sample CD-ratio term aligned with the official CD scorer")
    parser.add_argument("--loss_straight_weight", type=float, default=0.0)
    parser.add_argument("--straight_direction_weight", type=float, default=1.0)
    parser.add_argument("--straight_distance_weight", type=float, default=1.0)
    parser.add_argument("--loss_stage_weight", type=float, default=0.20)
    parser.add_argument("--loss_stage2_plane_weight", type=float, default=0.0)
    parser.add_argument("--loss_stage2_normal_residual_weight", type=float, default=0.0)
    parser.add_argument("--loss_stage2_tangent_target_weight", type=float, default=0.0)
    parser.add_argument("--loss_noise_weight", type=float, default=0.05)
    parser.add_argument("--corr_huber_delta", type=float, default=0.01)
    parser.add_argument("--relative_eps", type=float, default=1e-8)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--use_cuda", action="store_true")
    parser.add_argument("--manual_dist", action="store_true",
                        help="use the single-host socket gradient synchronizer")
    parser.add_argument("--manual_dist_host", default="127.0.0.1")
    parser.add_argument("--manual_dist_port", type=int, default=0)
    parser.add_argument("--manual_dist_rank", type=int, default=0)
    parser.add_argument("--manual_dist_world_size", type=int, default=1)
    parser.add_argument("--max_train_shapes", type=int, default=0)
    parser.add_argument("--max_val_shapes", type=int, default=0)
    parser.add_argument("--train_list", default="",
                        help="explicit training split; defaults to datalist_dir/train.txt")
    parser.add_argument("--holdout_list", default="",
                        help="internal train-only holdout used when val_steps_per_epoch is positive")
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
    if args.lr_schedule == "warmup_cosine" and args.train_steps_per_epoch <= 0:
        raise ValueError("--lr_schedule warmup_cosine requires --train_steps_per_epoch > 0")

    if args.use_cuda:
        jt.flags.use_cuda = 1
    global MANUAL_DIST
    if args.manual_dist:
        if args.manual_dist_world_size < 2 or args.manual_dist_port <= 0:
            raise ValueError("manual distributed mode requires world_size >= 2 and a valid port")
        MANUAL_DIST = ManualDist(
            args.manual_dist_host,
            args.manual_dist_port,
            args.manual_dist_rank,
            args.manual_dist_world_size,
        )
    is_main = (MANUAL_DIST is None or MANUAL_DIST.rank == 0) and ((not jt.in_mpi) or jt.rank == 0)
    random.seed(args.seed)
    np.random.seed(args.seed)
    jt.set_global_seed(args.seed)

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    train_list = Path(args.train_list) if args.train_list else Path(args.datalist_dir) / "train.txt"
    if not train_list.exists():
        raise FileNotFoundError("training split not found: {}".format(train_list))
    # The category vocabulary must be derived from training identifiers only.
    # This prevents validation/test split metadata from influencing a strict
    # train-only experiment while retaining deterministic sorted synset IDs.
    category_to_id = build_category_vocab_from_paths([train_list])

    train_set = ShapeNetPatchDataset(
        args.dataset_root,
        train_list,
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
        return_normals=(
            args.pgd_use_normal_corr_loss
            or args.loss_tangent_spacing_weight > 0.0
            or args.loss_stage2_plane_weight > 0.0
            or args.loss_stage2_normal_residual_weight > 0.0
            or args.loss_stage2_tangent_target_weight > 0.0
            or args.pgd_use_surface_head
            or args.pgd_use_surface_vector_head
        ),
        normal_k=args.normal_k,
        return_overlap_patch=args.pgd_overlap_consistency_weight > 0.0,
    )
    val_set = None
    if args.val_steps_per_epoch > 0:
        holdout_list = Path(args.holdout_list) if args.holdout_list else Path(args.datalist_dir) / "validate.txt"
        if not holdout_list.exists():
            raise FileNotFoundError("holdout split not found: {}".format(holdout_list))
        val_set = ShapeNetPatchDataset(
            args.dataset_root,
            holdout_list,
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
            return_normals=(
                args.pgd_use_normal_corr_loss
                or args.loss_tangent_spacing_weight > 0.0
                or args.loss_stage2_plane_weight > 0.0
                or args.loss_stage2_normal_residual_weight > 0.0
                or args.loss_stage2_tangent_target_weight > 0.0
                or args.pgd_use_surface_head
                or args.pgd_use_surface_vector_head
            ),
            normal_k=args.normal_k,
            return_overlap_patch=args.pgd_overlap_consistency_weight > 0.0,
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
        if MANUAL_DIST is not None:
            print("manual_dist world_size={}, rank={}".format(MANUAL_DIST.world_size, MANUAL_DIST.rank), flush=True)
        elif jt.in_mpi:
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
        print(
            "lr schedule: {} peak={} warmup_start={} warmup_steps={} min={}".format(
                args.lr_schedule,
                args.lr,
                args.lr_warmup_start,
                args.lr_warmup_steps,
                args.lr_min,
            ),
            flush=True,
        )
        if trainable_names:
            print("optimizer param names: {}".format(",".join(trainable_names)), flush=True)
    loss_fn = get_loss_fn(args.loss)

    # All ranks must finish dataset/model/optimizer setup before any rank can
    # reach the first gradient all-reduce.  This is especially important for
    # the manual TCP fallback because Jittor compiles the first-use CUDA
    # operators lazily and compilation time differs substantially by GPU.
    if MANUAL_DIST is not None:
        MANUAL_DIST.barrier()

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
            global_train_step_offset=epoch * args.train_steps_per_epoch,
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
        if args.train_steps_per_epoch > 0:
            start_step = epoch * args.train_steps_per_epoch
            end_step = start_step + args.train_steps_per_epoch - 1
            row["lr_start"] = learning_rate_for_step(args, start_step)
            row["lr_end"] = learning_rate_for_step(args, end_step)
        if MANUAL_DIST is not None:
            MANUAL_DIST.barrier()
        elif jt.in_mpi:
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
        if MANUAL_DIST is not None:
            MANUAL_DIST.barrier()
        elif jt.in_mpi:
            jt.mpi.mpi_barrier()


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    if MANUAL_DIST is not None:
        MANUAL_DIST.close()
    elif jt.in_mpi:
        jt.mpi.mpi_barrier()
    os._exit(0)
