#!/bin/sh
set -eu

# Open MPI's rsh launcher normally calls ssh/rsh with the target hostname as
# its first argument.  This machine is a single host with no sshd; execute the
# remaining command locally instead.  The script is intentionally limited to
# local MPI launches and must be selected explicitly with
# --mca plm_rsh_agent.
if [ "$#" -lt 2 ]; then
    echo "mpi_local_rsh.sh expects HOST COMMAND..." >&2
    exit 2
fi
if [ "${PGD_MPI_DEBUG:-0}" = "1" ]; then
    printf '%s\n' "mpi_local_rsh args: $*" >> /tmp/pgd_mpi_local_rsh.log
fi
shift
exec "$@"
