#!/usr/bin/env python3
"""Select exactly one round-2 candidate using only fixed train-holdout logs."""

import argparse
import json
import re
from pathlib import Path


# The scorer prints ``最终得分 (0.5×CD + 0.5×P2S): 77.73``.  Match only
# the numeric token after the score separator, not either formula coefficient.
SCORE_RE = re.compile(r"最终得分.*?[:：]\s*([0-9]+(?:\.[0-9]+)?)")


def score_from_log(path):
    text = path.read_text(errors="replace")
    values = SCORE_RE.findall(text)
    if not values:
        raise ValueError("could not parse final score from {}".format(path))
    return float(values[-1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round_root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.round_root)
    candidates = []
    for path in sorted((root / "holdout_screen").glob("*/evaluate.log")):
        name = path.parent.name
        candidates.append({"name": name, "score": score_from_log(path), "log": str(path)})
    if not candidates:
        raise RuntimeError("no holdout evaluation logs")
    # Stable deterministic tiebreaker prevents a hidden full-validation choice.
    candidates.sort(key=lambda item: (-item["score"], item["name"]))
    winner = candidates[0]
    base = winner["name"].split("_pca015_tangent125", 1)[0]
    weight = list((root / base).glob("pgd-shapenet-epoch*.npz"))
    if len(weight) != 1:
        raise RuntimeError("expected one checkpoint for {}, got {}".format(base, len(weight)))
    payload = {"winner": winner, "candidates": candidates, "checkpoint": str(weight[0]),
               "postprocess": winner["name"].endswith("_pca015_tangent125")}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
