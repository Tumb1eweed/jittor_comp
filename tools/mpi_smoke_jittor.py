import jittor as jt


def main():
    jt.flags.use_cuda = 1
    rank = jt.mpi.world_rank() if jt.in_mpi else 0
    world = jt.mpi.world_size() if jt.in_mpi else 1
    local = jt.mpi.local_rank() if jt.in_mpi else 0
    x = jt.ones((2, 2)) * (rank + 1)
    y = x.mpi_all_reduce("mean") if world > 1 else x
    print(
        f"rank={rank} local_rank={local} world_size={world} mean={y.numpy().tolist()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
