import argparse
import json
import os
import random
import shutil
from pathlib import Path

os.environ.setdefault("nvcc_path", "")

import numpy as np
import jittor as jt
from jittor import nn
from tqdm.auto import tqdm

from datasets.pcl import PointCloudDataset
from datasets.patch import PairedPatchDataset
from models.pgd import PGDModel
from utils.misc import get_log_dir_name_tblogger, seed_all, str_list
from utils.transforms import standard_train_transforms


def get_mpi_info():
    if getattr(jt, "in_mpi", False):
        return jt.mpi.world_rank(), jt.mpi.world_size(), jt.mpi.local_rank()
    return 0, 1, 0


def is_main_process():
    rank, _, _ = get_mpi_info()
    return rank == 0


def stack_batch(items, key):
    return jt.array(np.stack([np.asarray(item[key], dtype=np.float32) for item in items], axis=0))


def chamfer_loss(pred, clean):
    dist = jt.sum((pred[:, :, None, :] - clean[:, None, :, :]) ** 2, dim=-1)
    d1 = dist.min(dim=2)
    d2 = dist.min(dim=1)
    return d1.mean() + d2.mean()


def sample_batch(dataset, batch_size, rng):
    return [dataset[rng.randrange(len(dataset))] for _ in range(batch_size)]


def train_epoch(model, dataset, optimizer, args, epoch, steps_per_epoch):
    model.train()
    rank, world_size, _ = get_mpi_info()
    rng = random.Random(args.seed + epoch * 1000003 + rank * 9176)
    losses = []
    pbar = tqdm(
        range(steps_per_epoch),
        desc=f"Train epoch {epoch:02d}",
        disable=(rank != 0 and not args.mpi_log_all_ranks),
    )
    for _ in pbar:
        batch = sample_batch(dataset, args.train_batch_size, rng)
        noisy = stack_batch(batch, "pcl_noisy")
        clean = stack_batch(batch, "pcl_clean")
        seeds = stack_batch(batch, "seed_pnts")
        noisy_centered = noisy - seeds
        clean_centered = clean - seeds
        pred_disp = model(noisy_centered)
        pred = noisy_centered + pred_disp
        loss = chamfer_loss(pred, clean_centered)
        optimizer.step(loss)
        val = float(loss.numpy())
        if world_size > 1:
            val = float(jt.array([val]).mpi_all_reduce("mean").numpy()[0])
        losses.append(val)
        pbar.set_postfix(loss=f"{val:.6f}")
    return float(np.mean(losses))


def main(args):
    if args.use_cuda:
        jt.flags.use_cuda = 1

    rank, world_size, local_rank = get_mpi_info()
    if args.mpi and world_size == 1:
        raise RuntimeError("--mpi was set, but Jittor is not running inside mpirun/mpiexec")
    if world_size > 1 and not args.mpi:
        print("[MPI] Detected mpirun/mpiexec; enabling distributed Jittor optimizer sync.", flush=True)

    rank_seed = args.seed + rank * 100003
    seed_all(rank_seed)
    np.random.seed(rank_seed)
    random.seed(rank_seed)
    jt.set_global_seed(args.seed)

    log_name = args.tag or get_log_dir_name_tblogger(name=f"PGD_{args.dataset}_")
    log_dir = Path(args.log_root) / log_name
    if is_main_process():
        log_dir.mkdir(parents=True, exist_ok=True)
        for file_ in ["./models/feature.py", "./models/blocks.py", "./models/utils.py", "./models/pgd.py", "./train.py"]:
            shutil.copyfile(file_, log_dir / Path(file_).name)
        args_to_save = vars(args).copy()
        args_to_save.update({"mpi_world_size": world_size, "mpi_rank": rank, "mpi_local_rank": local_rank})
        with open(log_dir / "args.json", "w") as f:
            json.dump(args_to_save, f, indent=2)
    if world_size > 1:
        jt.sync_all()
    print(f"[MPI] rank={rank} local_rank={local_rank} world_size={world_size}", flush=True)

    train_dset = PairedPatchDataset(
        datasets=[
            PointCloudDataset(
                root=args.dataset_root,
                dataset=args.dataset,
                split="train",
                resolution=resl,
                transform=standard_train_transforms(
                    noise_std_max=args.noise_max,
                    noise_std_min=args.noise_min,
                    rotate=args.aug_rotate,
                ),
                max_shapes=args.max_shapes,
            )
            for resl in args.resolutions
        ],
        patch_size=args.patch_size,
        num_patches=args.patches_per_shape_per_epoch,
    )

    model = PGDModel(args)
    if args.init_from_weights:
        report = model.load_npz(args.init_from_weights)
        if is_main_process():
            print("Initialized from", report)
    if world_size > 1 and args.mpi_sync_initial_params:
        model.mpi_param_broadcast()
    optimizer = nn.Adam(model.parameters(), lr=args.lr)

    steps_per_epoch = args.steps_per_epoch
    if steps_per_epoch <= 0:
        steps_per_epoch = max(1, len(train_dset) // args.train_batch_size)
    if world_size > 1 and args.mpi_scale_steps_by_world_size:
        steps_per_epoch = max(1, steps_per_epoch // world_size)

    history = []
    for epoch in range(args.max_epochs):
        loss = train_epoch(model, train_dset, optimizer, args, epoch, steps_per_epoch)
        history.append({"epoch": epoch, "loss": loss})
        if is_main_process():
            print(f"[Epoch {epoch:02d}] loss={loss:.8f}")
        if is_main_process() and ((epoch + 1) % args.save_interval == 0 or epoch + 1 == args.max_epochs):
            ckpt = log_dir / f"pgd-jittor-epoch{epoch:02d}-loss{loss:.8f}.npz"
            model.save_npz(ckpt)
            with open(log_dir / "history.json", "w") as f:
                json.dump(history, f, indent=2)
        if world_size > 1:
            jt.sync_all()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, default="./data")
    parser.add_argument("--dataset", type=str, default="PUNet")
    parser.add_argument("--patches_per_shape_per_epoch", type=int, default=1000)
    parser.add_argument("--resolutions", type=str_list, default=["10000_poisson", "30000_poisson", "50000_poisson"])
    parser.add_argument("--noise_min", type=float, default=0.005)
    parser.add_argument("--noise_max", type=float, default=0.02)
    parser.add_argument("--train_batch_size", type=int, default=20)
    parser.add_argument("--save_interval", type=int, default=5)
    parser.add_argument("--aug_rotate", type=eval, default=True, choices=[True, False])
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--log_root", type=str, default="./logs/PGD")
    parser.add_argument("--max_epochs", type=int, default=800)
    parser.add_argument("--steps_per_epoch", type=int, default=0)
    parser.add_argument("--patch_size", type=int, default=1000)
    parser.add_argument("--init_from_weights", type=str, default=None)
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--use_cuda", action="store_true")
    parser.add_argument("--max_shapes", type=int, default=0)
    parser.add_argument("--mpi", action="store_true")
    parser.add_argument("--mpi_log_all_ranks", action="store_true")
    parser.add_argument("--mpi_sync_initial_params", action="store_true", default=True)
    parser.add_argument("--no_mpi_sync_initial_params", dest="mpi_sync_initial_params", action="store_false")
    parser.add_argument("--mpi_scale_steps_by_world_size", action="store_true")
    args = parser.parse_args()
    main(args)
