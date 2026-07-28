import argparse
import json
import os
import sys
import zlib
from pathlib import Path
from multiprocessing import Pool

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from tqdm.auto import tqdm

from tools.train_shapenet_one_epoch import load_obj_mesh, normalize_unit_sphere, read_split, sample_mesh


def item_output_path(output_dir, rel):
    return output_dir / "{}.npy".format(rel)


def prepare_one(task):
    rel, dataset_root, data_name, output_dir, sample_points, overwrite = task
    out_path = item_output_path(output_dir, rel)
    if out_path.exists() and not overwrite:
        return "skipped", rel
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.random.seed(zlib.crc32(rel.encode("utf-8")) & 0xffffffff)
    mesh_path = dataset_root / rel / data_name
    vertices, faces = load_obj_mesh(mesh_path)
    points = normalize_unit_sphere(sample_mesh(vertices, faces, sample_points))
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp.{}".format(os.getpid()))
    np.save(tmp_path, points.astype(np.float32))
    os.replace(str(tmp_path) + ".npy", out_path)
    return "written", rel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", default="/home/dataset_train")
    parser.add_argument("--datalist_dir", default="/home/PGD/datalist")
    parser.add_argument("--data_name", default="models/model_normalized.obj")
    parser.add_argument("--output_dir", default="/home/dataset_train_pgd_points_50k")
    parser.add_argument("--sample_points", type=int, default=50000)
    parser.add_argument("--splits", nargs="+", default=["train", "validate"])
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--max_items", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    np.random.seed(args.seed)
    dataset_root = Path(args.dataset_root)
    datalist_dir = Path(args.datalist_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "dataset_root": str(dataset_root),
        "datalist_dir": str(datalist_dir),
        "data_name": args.data_name,
        "sample_points": args.sample_points,
        "splits": args.splits,
        "seed": args.seed,
    }
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    items = []
    seen = set()
    for split in args.splits:
        split_file = datalist_dir / "{}.txt".format(split)
        for rel in read_split(split_file):
            if rel not in seen:
                seen.add(rel)
                items.append(rel)
    if args.max_items > 0:
        items = items[:args.max_items]

    tasks = [
        (rel, dataset_root, args.data_name, output_dir, args.sample_points, args.overwrite)
        for rel in items
    ]
    skipped = 0
    written = 0
    if args.workers > 1:
        with Pool(args.workers) as pool:
            iterator = pool.imap_unordered(prepare_one, tasks, chunksize=8)
            for status, _ in tqdm(iterator, total=len(tasks), desc="pre-sampling meshes"):
                if status == "written":
                    written += 1
                elif status == "skipped":
                    skipped += 1
    else:
        for task in tqdm(tasks, desc="pre-sampling meshes"):
            status, _ = prepare_one(task)
            if status == "written":
                written += 1
            elif status == "skipped":
                skipped += 1

    print("items: {} written: {} skipped: {} output_dir: {}".format(len(items), written, skipped, output_dir), flush=True)


if __name__ == "__main__":
    main()
