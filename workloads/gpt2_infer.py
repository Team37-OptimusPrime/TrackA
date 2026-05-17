#!/usr/bin/env python3
"""
GPU-centric Workload: GPT-2 Inference (Memory-bound, KV-cache 대역폭 집약)
배치 크기로 GPU 메모리 대역폭 부하 조절
"""

import argparse
import sys

try:
    import torch
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
except ImportError:
    print("[ERROR] transformers 미설치. pip install transformers torch", file=sys.stderr)
    sys.exit(1)


def run(batch_size: int, max_new_tokens: int, loop: bool):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[INFO] GPT-2 로딩 중 (device={device})...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()

    # ✅ GPT-2 padding 명시 (중요)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    prompt = "The quick brown fox jumps over the lazy dog"

    # ✅ padding + mask 생성
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    # ✅ 배치 복제 (ids + mask 모두!)
    input_ids = inputs["input_ids"].repeat(batch_size, 1).to(device)
    attention_mask = inputs["attention_mask"].repeat(batch_size, 1).to(device)

    print(f"[INFO] GPT-2 추론 시작 | batch={batch_size} | max_new_tokens={max_new_tokens}")

    iteration = 0
    with torch.no_grad():
        while True:
            _ = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,  # ⭐ 핵심 수정
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,  # KV-cache 활성화
            )
            iteration += 1
            if not loop:
                break

    print(f"[INFO] 완료 ({iteration} iterations)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--loop", action="store_true")
    args = parser.parse_args()
    run(args.batch_size, args.max_new_tokens, args.loop)