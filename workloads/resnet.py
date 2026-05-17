#!/usr/bin/env python3
"""
GPU-centric Workload: ResNet-50 Inference (Compute-bound)
배치 크기로 GPU 점유율 제어
"""

import argparse
import time
import sys

try:
    import torch
    import torchvision.models as models
except ImportError:
    #torch 미설치 시
    print("[ERROR] torch/torchvision 미설치. pip install torch torchvision", file=sys.stderr)
    sys.exit(1)


def run(batch_size: int, loop: bool):
    #gpu로 실험 가능하면 gpu, 불
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("[WARN] CUDA 없음. CPU로 실행됩니다.", file=sys.stderr)

    model = models.resnet50(pretrained=False).to(device).eval()
    dummy = torch.randn(batch_size, 3, 224, 224, device=device)

    print(f"[INFO] ResNet-50 추론 시작 | device={device} | batch={batch_size}")

    with torch.no_grad():
        iteration = 0
        while True:
            _ = model(dummy)
            iteration += 1
            if not loop:
                break

    print(f"[INFO] 완료 ({iteration} iterations)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--loop", action="store_true", help="종료 신호까지 반복")
    args = parser.parse_args()
    run(args.batch_size, args.loop)
