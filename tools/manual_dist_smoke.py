"""Two-rank Jittor CUDA gradient-synchronization smoke test."""

import argparse
import os
import sys

import numpy as np
import jittor as jt
from jittor import nn

try:
    from tools.manual_dist import ManualDist
except ImportError:
    from manual_dist import ManualDist


def sync_grads(optimizer, dist):
    arrays = []
    shapes = []
    for group in optimizer.param_groups:
        for param, grad in zip(group["params"], group["grads"]):
            if param.is_stop_grad():
                continue
            value = np.asarray(grad.numpy(), dtype=np.float32)
            arrays.append(value.reshape(-1))
            shapes.append(value.shape)
    flat = np.concatenate(arrays) if arrays else np.empty((0,), dtype=np.float32)
    flat = dist.allreduce_mean(flat)
    offset = 0
    shape_index = 0
    for group in optimizer.param_groups:
        for param, grad in zip(group["params"], group["grads"]):
            if param.is_stop_grad():
                continue
            shape = shapes[shape_index]
            size = int(np.prod(shape))
            grad.update(jt.array(flat[offset:offset + size].reshape(shape)))
            offset += size
            shape_index += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manual_dist", action="store_true")
    parser.add_argument("--manual_dist_host", default=os.environ.get("manual_dist_host", "127.0.0.1"))
    parser.add_argument("--manual_dist_port", type=int, default=int(os.environ.get("manual_dist_port", "0")))
    parser.add_argument("--manual_dist_rank", type=int, default=int(os.environ.get("manual_dist_rank", "0")))
    parser.add_argument("--manual_dist_world_size", type=int, default=int(os.environ.get("manual_dist_world_size", "1")))
    args = parser.parse_args()
    if not args.manual_dist:
        raise RuntimeError("run this smoke test through tools/launch_manual_multigpu.py")
    jt.flags.use_cuda = 1
    dist = ManualDist(args.manual_dist_host, args.manual_dist_port, args.manual_dist_rank, args.manual_dist_world_size)
    jt.set_global_seed(1234)
    model = nn.Linear(4, 1)
    optimizer = nn.SGD(model.parameters(), 0.05)
    x = jt.array(np.full((8, 4), args.manual_dist_rank + 1, dtype=np.float32))
    y = jt.array(np.full((8, 1), 2.0, dtype=np.float32))
    losses = []
    for step in range(3):
        pred = model(x)
        loss = ((pred - y) ** 2).mean()
        optimizer.zero_grad()
        optimizer.backward(loss)
        sync_grads(optimizer, dist)
        optimizer.step()
        losses.append(float(loss.numpy()))
    dist.barrier()
    print("manual_dist_smoke rank={} gpu={} losses={} weight_mean={:.8f}".format(
        args.manual_dist_rank,
        os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        [round(item, 6) for item in losses],
        float(model.weight.numpy().mean()),
    ), flush=True)
    dist.close()
    # Jittor 1.3.11 can double-free CUDA runtime state during normal Python
    # finalization after a multi-process CUDA run; match the training entry
    # point and exit after all output/collectives are complete.
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
