#!/usr/bin/env python3
"""
CPU-centric Workload: 순수 CPU 행렬 곱 (통제 실험)
스레드 수 조절로 부하율 제어
실험마다 스레드 수 조절 필요.
"""

import argparse
import os
import sys


def run(threads: int, size: int, loop: bool):
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads)

    try:
        import numpy as np
    except ImportError:
        print("[ERROR] numpy 미설치. pip install numpy", file=sys.stderr)
        sys.exit(1)

    A = np.random.randn(size, size).astype(np.float32)
    B = np.random.randn(size, size).astype(np.float32)

    print(f"[INFO] MatMul CPU 시작 | threads={threads} | {size}x{size}")
    iteration = 0
    while True:
        C = np.dot(A, B)
        iteration += 1
        if not loop:
            break
    print(f"[INFO] 완료 ({iteration} iterations)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--size", type=int, default=4096)
    parser.add_argument("--loop", action="store_true")
    args = parser.parse_args()
    run(args.threads, args.size, args.loop)
