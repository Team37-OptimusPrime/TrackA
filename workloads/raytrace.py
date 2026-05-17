#!/usr/bin/env python3
"""
교차 실험 Workload: Ray Tracing (CPU / GPU)
동일 장면을 CPU렌더/GPU렌더로 비교 (Mixed-bound)

CPU: numpy 기반 소프트웨어 레이 트레이싱
GPU: CUDA/cupy 기반 레이 트레이싱
"""

import argparse
import sys
import math


# ─── 공통 장면 설정 ───────────────────────────────────────────────────────────
IMAGE_W = 800
IMAGE_H = 600
MAX_DEPTH = 4
NUM_SAMPLES = 4   # anti-aliasing samples per pixel


def run_cpu(threads: int, loop: bool):
    import os
    os.environ["OMP_NUM_THREADS"] = str(threads)

    try:
        import numpy as np
    except ImportError:
        print("[ERROR] numpy 미설치.", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Ray Tracing CPU 시작 | threads={threads} | {IMAGE_W}x{IMAGE_H}")

    # 간단한 구형 장면 정의
    # sphere: [cx, cy, cz, radius, r, g, b]
    spheres = np.array([
        [0,    0,  -5, 1.0, 1.0, 0.2, 0.2],
        [-2,   0,  -6, 1.0, 0.2, 1.0, 0.2],
        [2,    0,  -6, 1.0, 0.2, 0.2, 1.0],
        [0,  -101, -5, 100, 0.8, 0.8, 0.8],  # 바닥
    ], dtype=np.float32)

    def ray_trace_frame():
        """전체 프레임 렌더링 (벡터화)"""
        # 픽셀 좌표 그리드
        x = np.linspace(-1, 1, IMAGE_W)
        y = np.linspace(-0.75, 0.75, IMAGE_H)
        xx, yy = np.meshgrid(x, y)  # (H, W)

        # 카메라 원점
        origin = np.array([0, 0, 0], dtype=np.float32)

        # 방향 벡터 (H, W, 3)
        dir_x = xx.ravel()
        dir_y = yy.ravel()
        dir_z = np.full_like(dir_x, -1.0)
        norms = np.sqrt(dir_x**2 + dir_y**2 + dir_z**2)
        dir_x /= norms; dir_y /= norms; dir_z /= norms

        # 각 구와 교차 검사 (최근접)
        N = len(dir_x)
        t_min = np.full(N, 1e9, dtype=np.float32)
        hit_sphere = np.full(N, -1, dtype=np.int32)

        ox, oy, oz = origin

        for i, (cx, cy, cz, r, *_) in enumerate(spheres):
            ocx = ox - cx; ocy = oy - cy; ocz = oz - cz
            a = dir_x**2 + dir_y**2 + dir_z**2
            b = 2*(dir_x*ocx + dir_y*ocy + dir_z*ocz)
            c = ocx**2 + ocy**2 + ocz**2 - r**2
            disc = b**2 - 4*a*c
            valid = disc >= 0
            t = np.where(valid, (-b - np.sqrt(np.maximum(disc, 0))) / (2*a), 1e9)
            t = np.where(t > 1e-4, t, 1e9)
            closer = t < t_min
            t_min = np.where(closer, t, t_min)
            hit_sphere = np.where(closer, i, hit_sphere)

        # 색상 결정
        colors = np.zeros((N, 3), dtype=np.float32)
        for i, (_, _, _, _, cr, cg, cb) in enumerate(spheres):
            mask = hit_sphere == i
            colors[mask] = [cr, cg, cb]

        return colors.reshape(IMAGE_H, IMAGE_W, 3)

    iteration = 0
    while True:
        _ = ray_trace_frame()
        iteration += 1
        if not loop:
            break

    print(f"[INFO] 완료 ({iteration} frames)")


def run_gpu(loop: bool):
    try:
        import cupy as cp
    except ImportError:
        print("[WARN] cupy 미설치. PyTorch CUDA fallback 사용.", file=sys.stderr)
        _run_gpu_torch(loop)
        return

    print(f"[INFO] Ray Tracing GPU (CuPy) 시작 | {IMAGE_W}x{IMAGE_H}")

    spheres = cp.array([
        [0,    0,  -5, 1.0, 1.0, 0.2, 0.2],
        [-2,   0,  -6, 1.0, 0.2, 1.0, 0.2],
        [2,    0,  -6, 1.0, 0.2, 0.2, 1.0],
        [0,  -101, -5, 100, 0.8, 0.8, 0.8],
    ], dtype=cp.float32)

    x = cp.linspace(-1, 1, IMAGE_W)
    y = cp.linspace(-0.75, 0.75, IMAGE_H)
    xx, yy = cp.meshgrid(x, y)

    def render():
        dir_x = xx.ravel().copy()
        dir_y = yy.ravel().copy()
        dir_z = cp.full_like(dir_x, -1.0)
        norms = cp.sqrt(dir_x**2 + dir_y**2 + dir_z**2)
        dir_x /= norms; dir_y /= norms; dir_z /= norms

        N = len(dir_x)
        t_min = cp.full(N, 1e9, dtype=cp.float32)
        hit_sphere = cp.full(N, -1, dtype=cp.int32)

        for i in range(len(spheres)):
            cx, cy, cz, r = spheres[i, 0], spheres[i, 1], spheres[i, 2], spheres[i, 3]
            ocx = -cx; ocy = -cy; ocz = -cz
            a = dir_x**2 + dir_y**2 + dir_z**2
            b = 2*(dir_x*ocx + dir_y*ocy + dir_z*ocz)
            c = ocx**2 + ocy**2 + ocz**2 - r**2
            disc = b**2 - 4*a*c
            t = cp.where(disc >= 0, (-b - cp.sqrt(cp.maximum(disc, 0))) / (2*a), 1e9)
            t = cp.where(t > 1e-4, t, 1e9)
            closer = t < t_min
            t_min = cp.where(closer, t, t_min)
            hit_sphere = cp.where(closer, i, hit_sphere)

        colors = cp.zeros((N, 3), dtype=cp.float32)
        for i in range(len(spheres)):
            cr, cg, cb = spheres[i, 4], spheres[i, 5], spheres[i, 6]
            mask = hit_sphere == i
            colors[mask, 0] = cr; colors[mask, 1] = cg; colors[mask, 2] = cb

        return colors.reshape(IMAGE_H, IMAGE_W, 3)

    iteration = 0
    while True:
        _ = render()
        cp.cuda.Stream.null.synchronize()
        iteration += 1
        if not loop:
            break

    print(f"[INFO] 완료 ({iteration} frames)")


def _run_gpu_torch(loop: bool):
    """cupy 미설치 시 PyTorch CUDA 기반 fallback"""
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Ray Tracing GPU (Torch) 시작 | device={device}")

    spheres = torch.tensor([
        [0,    0,  -5, 1.0, 1.0, 0.2, 0.2],
        [-2,   0,  -6, 1.0, 0.2, 1.0, 0.2],
        [2,    0,  -6, 1.0, 0.2, 0.2, 1.0],
        [0,  -101, -5, 100, 0.8, 0.8, 0.8],
    ], device=device)

    x = torch.linspace(-1, 1, IMAGE_W, device=device)
    y = torch.linspace(-0.75, 0.75, IMAGE_H, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")

    iteration = 0
    while True:
        dir_x = xx.reshape(-1).clone()
        dir_y = yy.reshape(-1).clone()
        dir_z = torch.full_like(dir_x, -1.0)
        norms = torch.sqrt(dir_x**2 + dir_y**2 + dir_z**2)
        dir_x /= norms; dir_y /= norms; dir_z /= norms

        N = dir_x.numel()
        t_min = torch.full((N,), 1e9, device=device)
        hit = torch.full((N,), -1, dtype=torch.int32, device=device)

        for i in range(spheres.shape[0]):
            cx, cy, cz, r = spheres[i, 0], spheres[i, 1], spheres[i, 2], spheres[i, 3]
            a = dir_x**2 + dir_y**2 + dir_z**2
            b = 2*(dir_x*(-cx) + dir_y*(-cy) + dir_z*(-cz))
            c = cx**2 + cy**2 + cz**2 - r**2
            disc = b**2 - 4*a*c
            t = torch.where(disc >= 0, (-b - torch.sqrt(disc.clamp(min=0)))/(2*a),
                            torch.tensor(1e9, device=device))
            t = torch.where(t > 1e-4, t, torch.tensor(1e9, device=device))
            closer = t < t_min
            t_min = torch.where(closer, t, t_min)
            hit = torch.where(closer, torch.tensor(i, device=device), hit)

        torch.cuda.synchronize()
        iteration += 1
        if not loop:
            break

    print(f"[INFO] 완료 ({iteration} frames)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ray Tracing 워크로드")
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--loop", action="store_true")
    args = parser.parse_args()

    if args.device == "cpu":
        run_cpu(args.threads, args.loop)
    else:
        run_gpu(args.loop)
