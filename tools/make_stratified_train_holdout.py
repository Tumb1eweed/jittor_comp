#!/usr/bin/env python3.7
"""Create deterministic category-stratified train/holdout split files."""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def synset_id(entry):
    parts = entry.split("/")
    return parts[1] if len(parts) >= 2 and parts[0] == "shapenet" else parts[0]


def read_entries(path):
    with Path(path).open("r") as f:
        return [line.strip() for line in f if line.strip()]


def write_entries(path, entries):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("\n".join(entries) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="source train.txt")
    parser.add_argument("--train-output", required=True)
    parser.add_argument("--holdout-output", required=True)
    parser.add_argument("--screen-output", default="",
                        help="optional category-balanced subset of holdout for mesh screening")
    parser.add_argument("--screen-per-category", type=int, default=2)
    parser.add_argument("--holdout-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=8200)
    args = parser.parse_args()
    if not 0.0 < args.holdout_ratio < 1.0:
        raise ValueError("--holdout-ratio must be in (0, 1)")

    groups = defaultdict(list)
    for entry in read_entries(args.input):
        groups[synset_id(entry)].append(entry)
    rng = random.Random(args.seed)
    train, holdout, screen = [], [], []
    for category in sorted(groups):
        entries = sorted(groups[category])
        rng.shuffle(entries)
        count = max(1, int(round(len(entries) * args.holdout_ratio)))
        category_holdout = entries[:count]
        holdout.extend(category_holdout)
        if args.screen_output:
            screen.extend(category_holdout[:min(len(category_holdout), args.screen_per_category)])
        train.extend(entries[count:])
    train.sort()
    holdout.sort()
    screen.sort()
    if set(train) & set(holdout):
        raise RuntimeError("train and holdout overlap")
    if len(train) + len(holdout) != sum(len(group) for group in groups.values()):
        raise RuntimeError("split lost entries")
    write_entries(args.train_output, train)
    write_entries(args.holdout_output, holdout)
    if args.screen_output:
        write_entries(args.screen_output, screen)
    metadata = {
        "input": str(Path(args.input).resolve()),
        "seed": args.seed,
        "holdout_ratio": args.holdout_ratio,
        "train_count": len(train),
        "holdout_count": len(holdout),
        "screen_count": len(screen),
        "category_counts": {category: len(groups[category]) for category in sorted(groups)},
    }
    with Path(args.holdout_output).with_suffix(".json").open("w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
