#!/usr/bin/env python3
"""Lightweight test-time tangent-plane repulsion for denoised point clouds.

Moves points a small distance away from locally dense neighbors, projecting the
repulsion onto a PCA-estimated tangent plane to avoid changing the surface
normal direction. No training data or GT is used.
"""
import argparse
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

def process(points, k=12, strength=0.05, iters=1):
    x = points.astype(np.float32, copy=True)
    for _ in range(int(iters)):
        tree = cKDTree(x)
        d, idx = tree.query(x, k=min(k + 1, len(x)))
        neigh = x[idx[:, 1:]]
        # repulsion from neighbors; normalize by local scale for robustness
        diff = x[:, None, :] - neigh
        dist = np.maximum(d[:, 1:, None], 1e-6)
        rep = (diff / dist).mean(axis=1)
        # Estimate tangent from neighborhood covariance and remove normal comp.
        centered = neigh - x[:, None, :]
        cov = np.einsum('nki,nkj->nij', centered, centered) / max(1, neigh.shape[1])
        vals, vecs = np.linalg.eigh(cov)
        normal = vecs[:, :, 0]
        rep = rep - (rep * normal).sum(axis=1, keepdims=True) * normal
        # step proportional to local spacing, clipped to prevent artifacts
        radius = np.median(d[:, 1:], axis=1, keepdims=True)
        step = np.clip(float(strength) * radius, 0.0, 0.25 * radius)
        x = x + step * rep
    return x.astype(np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source_root', required=True)
    ap.add_argument('--output_root', required=True)
    ap.add_argument('--k', type=int, default=12)
    ap.add_argument('--strength', type=float, default=0.05)
    ap.add_argument('--iters', type=int, default=1)
    a = ap.parse_args(); src=Path(a.source_root); out=Path(a.output_root)
    for p in (src/'pred').glob('**/denoised.npy'):
        rel=p.relative_to(src/'pred'); arr=np.load(p)
        q=out/'pred'/rel; q.parent.mkdir(parents=True,exist_ok=True); np.save(q,process(arr,a.k,a.strength,a.iters))
        for sub,fn in [('gt','clean.npy'),('noisy','noisy.npy')]:
            s=src/sub/rel.parent/fn; t=out/sub/rel.parent/fn
            if s.exists():
                t.parent.mkdir(parents=True,exist_ok=True); np.save(t,np.load(s))
if __name__=='__main__': main()
