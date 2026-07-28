#!/bin/sh
set -eu

# Jittor asks for Open MPI's split --showme flags, while MPICH exposes one
# combined command through -show.  Keep compile and link flags separate;
# returning -lmpich as a compile flag can leave Jittor's MPI custom op only
# partially built.
case "${1:-}" in
    --showme:compile)
        /usr/bin/mpicc.mpich -show | awk '{
            for (i = 2; i <= NF; ++i)
                if ($i ~ /^(-I|-D|-f|-W|-pthread)/) printf "%s ", $i;
            printf "\n";
        }'
        ;;
    --showme:link)
        /usr/bin/mpicc.mpich -show | awk '{
            for (i = 2; i <= NF; ++i)
                if ($i ~ /^(-L|-l|-Wl,|-pthread)/) printf "%s ", $i;
            printf "\n";
        }'
        ;;
    *)
        exec /usr/bin/mpicc.mpich "$@"
        ;;
esac
