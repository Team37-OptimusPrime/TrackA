#!/usr/bin/env python3
"""
CPU-centric Workload: Matrix Multiplication (Compute-bound, BLAS 수준)
+ 교차 실험: GEMM CPU/GPU
device 인수로 CPU/GPU 전환
"""

import argparse
import sys
import os


def run_cpu(threads: int, size: int, loop: bool):
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads)

    try:
        import numpy as np
    except ImportError:
        print("[ERROR] numpy 미설치.", file=sys.stderr)
        sys.exit(1)

    A = np.random.randn(size, size).astype(np.float16)
    B = np.random.randn(size, size).astype(np.float16)

    print(f"[INFO] GEMM CPU 시작 | threads={threads} | size={size}x{size}")
    iteration = 0
    while True:
        _ = np.dot(A, B)
        iteration += 1
        if not loop:
            break
    print(f"[INFO] 완료 ({iteration} iterations)")


def run_gpu(size: int, loop: bool):
    try:
        import torch
    except ImportError:
        print("[ERROR] torch 미설치.", file=sys.stderr)
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    A = torch.randn(size, size, device=device, dtype=torch.float16)
    B = torch.randn(size, size, device=device, dtype=torch.float16)

    print(f"[INFO] GEMM GPU 시작 | device={device} | size={size}x{size}")
    iteration = 0
    while True:
        with torch.no_grad():
            _ = torch.mm(A, B)
            torch.cuda.synchronize()
            iteration += 1
            if not loop:
                break
            del _
            torch.cuda.empty_cache()
    print(f"[INFO] 완료 ({iteration} iterations)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GEMM 워크로드 (CPU/GPU)")
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--threads", type=int, default=8, help="CPU 스레드 수")
    parser.add_argument("--size", type=int, default=4096, help="행렬 크기 N (NxN)")
    parser.add_argument("--loop", action="store_true")
    args = parser.parse_args()

    if args.device == "cpu":
        run_cpu(args.threads, args.size, args.loop)
    else:
        run_gpu(args.size, args.loop)
