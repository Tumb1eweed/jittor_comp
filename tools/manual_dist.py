"""Small single-host process-group transport for Jittor training.

This is deliberately independent of Jittor/MPI initialization.  Each worker
still runs its model on its own CUDA_VISIBLE_DEVICES GPU; only the flattened
gradient buffer is copied to host memory for the rendezvous and copied back to
the worker GPU.  It is a functional fallback while the machine's MPI/NCCL
launcher is being repaired.
"""

import argparse
import os
import socket
import struct
import sys
import threading

import numpy as np


MAGIC = b"PGDMD1"
HEADER = struct.Struct("!6sBII")
ALLREDUCE = 1
BARRIER = 2
BROADCAST = 3


def _recv_exact(sock, size):
    chunks = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise RuntimeError("manual distributed peer closed the connection")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _recv_message(sock):
    magic, kind, rank, size = HEADER.unpack(_recv_exact(sock, HEADER.size))
    if magic != MAGIC:
        raise RuntimeError("invalid manual distributed message header")
    return kind, rank, _recv_exact(sock, size)


def _send_message(sock, kind, rank, payload=b""):
    payload = bytes(payload)
    sock.sendall(HEADER.pack(MAGIC, int(kind), int(rank), len(payload)) + payload)


class ManualDist:
    """Persistent client used by one training worker."""

    # The first CUDA/Jittor operator build can take several minutes on a
    # fresh per-GPU cache.  A worker that finishes its first batch early must
    # therefore be allowed to wait for slower ranks at the first collective.
    def __init__(self, host, port, rank, world_size, timeout=None):
        self.host = host
        self.port = int(port)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if timeout is None:
            # Epoch-end mesh validation can take longer than the first
            # operator compilation.  Keep ranks parked at the collective
            # while rank 0 evaluates instead of letting the socket expire.
            timeout = float(os.environ.get("manual_dist_timeout", "7200"))
        self.sock.settimeout(float(timeout))
        self.sock.connect((host, self.port))

    def allreduce_mean(self, values):
        array = np.ascontiguousarray(values, dtype=np.float32)
        _send_message(self.sock, ALLREDUCE, self.rank, array.tobytes())
        kind, _, payload = _recv_message(self.sock)
        if kind != ALLREDUCE:
            raise RuntimeError("manual distributed allreduce response mismatch")
        result = np.frombuffer(payload, dtype=np.float32)
        if result.size != array.size:
            raise RuntimeError("manual distributed allreduce shape mismatch")
        return result.reshape(array.shape).copy()

    def barrier(self):
        _send_message(self.sock, BARRIER, self.rank)
        kind, _, payload = _recv_message(self.sock)
        if kind != BARRIER or payload:
            raise RuntimeError("manual distributed barrier response mismatch")

    def broadcast(self, values):
        payload = b"" if self.rank != 0 else np.ascontiguousarray(values, dtype=np.float32).tobytes()
        _send_message(self.sock, BROADCAST, self.rank, payload)
        kind, _, response = _recv_message(self.sock)
        if kind != BROADCAST:
            raise RuntimeError("manual distributed broadcast response mismatch")
        result = np.frombuffer(response, dtype=np.float32)
        if self.rank == 0 and result.size != len(payload) // 4:
            raise RuntimeError("manual distributed broadcast shape mismatch")
        return result.copy()

    def close(self):
        try:
            self.sock.close()
        except Exception:
            pass


class _Server:
    def __init__(self, host, port, world_size):
        self.host = host
        self.port = int(port)
        self.world_size = int(world_size)
        self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.listener.bind((host, self.port))
        self.listener.listen(world_size)
        self.connections = {}

    def accept_workers(self):
        while len(self.connections) < self.world_size:
            sock, _ = self.listener.accept()
            # Keep the accept-side socket timeout consistent with the worker
            # timeout; otherwise the server can abandon a rank while it is
            # still compiling its first CUDA operators.
            sock.settimeout(float(os.environ.get("manual_dist_timeout", "7200")))
            kind, rank, payload = _recv_message(sock)
            if rank in self.connections:
                raise RuntimeError("duplicate manual distributed rank {}".format(rank))
            self.connections[rank] = sock
            self._pending = getattr(self, "_pending", {})
            self._pending[rank] = (kind, payload)
        print("manual_dist_ready world_size={} port={}".format(self.world_size, self.port), flush=True)

    def _next_round(self):
        messages = {}
        for rank in sorted(self.connections):
            if rank in getattr(self, "_pending", {}):
                messages[rank] = self._pending.pop(rank)
            else:
                kind, msg_rank, payload = _recv_message(self.connections[rank])
                if msg_rank != rank:
                    raise RuntimeError("manual distributed rank changed")
                messages[rank] = (kind, payload)
        return messages

    def serve(self):
        self.accept_workers()
        while True:
            try:
                messages = self._next_round()
            except RuntimeError as exc:
                if "peer closed" in str(exc):
                    return
                raise
            kinds = {kind for kind, _ in messages.values()}
            if len(kinds) != 1:
                raise RuntimeError("manual distributed workers reached different collectives")
            kind = next(iter(kinds))
            if kind == ALLREDUCE:
                arrays = [np.frombuffer(messages[r][1], dtype=np.float32) for r in sorted(messages)]
                if not arrays:
                    result = np.empty((0,), dtype=np.float32)
                else:
                    size = arrays[0].size
                    if any(a.size != size for a in arrays):
                        raise RuntimeError("manual distributed gradient lengths differ")
                    result = np.mean(np.stack(arrays, axis=0), axis=0, dtype=np.float32)
                payload = np.ascontiguousarray(result, dtype=np.float32).tobytes()
                for rank, sock in self.connections.items():
                    _send_message(sock, ALLREDUCE, rank, payload)
            elif kind == BARRIER:
                for rank, sock in self.connections.items():
                    _send_message(sock, BARRIER, rank)
            elif kind == BROADCAST:
                payload = messages[0][1]
                for rank, sock in self.connections.items():
                    _send_message(sock, BROADCAST, rank, payload)
            else:
                raise RuntimeError("unknown manual distributed collective {}".format(kind))


def run_server(host, port, world_size):
    server = _Server(host, port, world_size)
    try:
        server.serve()
    finally:
        for sock in server.connections.values():
            try:
                sock.close()
            except Exception:
                pass
        server.listener.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("server", choices=["server"])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    args = parser.parse_args()
    run_server(args.host, args.port, args.world_size)


if __name__ == "__main__":
    main()
