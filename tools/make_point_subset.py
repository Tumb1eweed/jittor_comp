#!/usr/bin/env python3
"""Deterministically subsample prediction/noisy/GT trees for cheap screens."""
import argparse, shutil
from pathlib import Path
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input_root', required=True); ap.add_argument('--output_root', required=True)
    ap.add_argument('--points', type=int, default=5000)
    args = ap.parse_args(); src=Path(args.input_root); dst=Path(args.output_root)
    paths=sorted((src/'pred').rglob('denoised.npy'))
    for pp in paths:
        rel=pp.relative_to(src/'pred'); parent=rel.parent
        pred=np.load(pp).astype(np.float32); n=min(args.points,len(pred))
        # Fixed evenly spaced indices avoid favouring patch/order-local clusters.
        idx=np.linspace(0,len(pred)-1,n,dtype=np.int64)
        out=dst/'pred'/rel; out.parent.mkdir(parents=True,exist_ok=True); np.save(out,pred[idx])
        for sub,fn in [('gt','clean.npy'),('noisy','noisy.npy')]:
            q=src/sub/parent/fn; o=dst/sub/parent/fn; o.parent.mkdir(parents=True,exist_ok=True)
            arr=np.load(q).astype(np.float32); np.save(o,arr[idx[:min(n,len(arr))]])
    print('subsampled',len(paths),'samples to',args.points)
if __name__=='__main__': main()
