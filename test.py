import argparse
import os
from pathlib import Path

import numpy as np
import jittor as jt

from models.pgd import PGDModel
from utils.misc import seed_all, str_list
from utils.transforms import NormalizeUnitSphere


def input_iter(input_dir):
    for fn in sorted(os.listdir(input_dir)):
        if not fn.endswith(".xyz"):
            continue
        pcl = np.loadtxt(os.path.join(input_dir, fn), dtype=np.float32)
        pcl_norm, center, scale = NormalizeUnitSphere.normalize_np(pcl)
        yield {"pcl_noisy": pcl_norm, "name": fn[:-4], "center": center, "scale": scale}


def main(noise):
    for resolution in args.resolutions:
        input_dir = os.path.join(args.input_root, f"{args.dataset}_{resolution}_{noise}")
        save_title = f"{args.dataset}_Ours{'' if args.niters == 1 else str(args.niters) + 'x'}_{args.tag}_{resolution}_{noise}"
        output_dir = Path(args.output_root) / save_title
        output_dir.mkdir(parents=True, exist_ok=True)
        model = PGDModel.load_from_npz(args.weights)
        for data in input_iter(input_dir):
            pcl_next = jt.array(data["pcl_noisy"].astype(np.float32))
            for _ in range(args.niters):
                pcl_next = model.patch_based_denoise(
                    pcl_noisy=pcl_next,
                    patch_size=args.patch_size,
                    seed_k=args.seed_k,
                    seed_k_alpha=args.seed_k_alpha,
                    patch_batch_size=args.patch_batch_size,
                )
            denoised = pcl_next.numpy() * data["scale"] + data["center"]
            np.savetxt(output_dir / f"{data['name']}.xyz", denoised, fmt="%.8f")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="weights/pgd-epoch19-val_loss0.00024044.npz")
    parser.add_argument("--input_root", type=str, default="./data/examples")
    parser.add_argument("--output_root", type=str, default="./data/results/PGD")
    parser.add_argument("--dataset", type=str, default="PUNet")
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--resolutions", type=str_list, default=["10000_poisson", "50000_poisson"])
    parser.add_argument("--noise_lvls", type=str_list, default=["0.03"])
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--niters", type=int, default=1)
    parser.add_argument("--patch_size", type=int, default=1000)
    parser.add_argument("--seed_k", type=int, default=5)
    parser.add_argument("--seed_k_alpha", type=float, default=10)
    parser.add_argument("--patch_batch_size", type=int, default=8)
    parser.add_argument("--use_cuda", action="store_true")
    args = parser.parse_args()
    seed_all(args.seed)
    np.random.seed(args.seed)
    jt.set_global_seed(args.seed)
    if args.use_cuda:
        jt.flags.use_cuda = 1
    for noise in args.noise_lvls:
        main(noise)
