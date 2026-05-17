#!/usr/bin/env python3
"""
run_experiment.py  (v4 - 버그 수정판)
실험 오케스트레이션 메인 스크립트

수정 사항:
  1. gemm_gpu batch[100] = 64 → size = 64*1024 = 65536 → VRAM 32 GB 필요
     → batch 값을 줄여 OOM 방지 (아래 주석 참고)
  2. 나머지 워크로드 주석 제거 → --workload 인자로 선택 실행

실험 사이클 (워크로드 × 부하 × 반복):
  Idle 안정화 (CPU ≤ 45°C, GPU ≤ 50°C, 최소 2분)
  └─ 부하 20% : 워크로드 기동 → Warmup 30초 → 측정 60초 → 종료
  └─ 부하 50% : 워크로드 기동 → Warmup 30초 → 측정 60초 → 종료
  └─ 부하 100%: 워크로드 기동 → Warmup 30초 → 측정 60초 → 종료
  Cooldown (CPU ≤ 45°C, GPU ≤ 50°C)

출력:
  results/<workload>/rep<NN>/load<L>.csv
"""

import os
import sys
import time
import signal
import logging
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
from collect_metric import MetricCollector, read_cpu_temp, read_gpu_metrics

# ────────────────────────────────────────────────
# 전역 설정
# ────────────────────────────────────────────────
BASE_DIR     = Path("/home/optimus/siyeon/exp_v3")
WORKLOAD_DIR = BASE_DIR / "workloads"
RESULTS_DIR  = BASE_DIR / "results_v4"
LOG_DIR      = BASE_DIR / "logs"

IDLE_SECS        = 60
STABILIZE_SECS   = 30
MEASURE_SECS     = 60
COOLDOWN_CPU_C   = 45.0
COOLDOWN_GPU_C   = 60.0
COOLDOWN_POLL_S  = 10
COOLDOWN_TIMEOUT = 120
REPEAT           = 1
SAMPLE_INTERVAL  = 0.1

LOAD_LEVELS = [20, 50, 100]

# 부하율별 할당 코어 (20코어 시스템 기준)
CPU_CORES = {
    20 : "0-3",    # 4코어
    50 : "0-9",    # 10코어
    100: "0-19",   # 20코어 전체
}

# ────────────────────────────────────────────────
# 워크로드 정의
# ────────────────────────────────────────────────
# gemm_gpu size 계산:
#   size = batch * 1024
#   VRAM 사용량 ≈ 2 * size^2 * 4 bytes (float32, 행렬 A·B)
#   batch=4  → size=4096   → ~0.13 GB  (안전)
#   batch=8  → size=8192   → ~0.5 GB   (안전)
#   batch=16 → size=16384  → ~2 GB     (안전)
#   batch=32 → size=32768  → ~8 GB     (주의, 40GB GPU 이상 필요)
#   batch=64 → size=65536  → ~32 GB    (OOM 위험 — 수정됨)
#
# GPU VRAM 용량에 맞게 아래 batch 값을 조정하세요.
# 기본값은 40GB GPU 기준으로 안전한 값입니다.

WORKLOADS = {
    "resnet": {
        "script"  : WORKLOAD_DIR / "resnet.py",
        "type"    : "gpu",
        "category": "GPU-centric",
        "batch"   : {20: 2, 50: 4, 100: 8},
    },
    "gpt2": {
        "script"  : WORKLOAD_DIR / "gpt2_infer.py",
        "type"    : "gpu",
        "category": "GPU-centric",
        "batch"   : {20: 2, 50: 4, 100: 8},
    },
    "matmul": {
        "script"  : WORKLOAD_DIR / "matmul.py",
        "type"    : "cpu",
        "category": "CPU-centric",
        "threads" : {20: 4, 50: 10, 100: 20},
    },
    "gromacs": {
        "script"  : WORKLOAD_DIR / "gromacs_run.py",
        "type"    : "cpu",
        "category": "CPU-centric",
        "threads" : {20: 4, 50: 10, 100: 20},
    },
    "gemm_cpu": {
        "script"  : WORKLOAD_DIR / "gemm.py",
        "type"    : "cpu",
        "category": "cross",
        "threads" : {20: 4, 50: 10, 100: 20},
    },
    "gemm_gpu": {
        "script"  : WORKLOAD_DIR / "gemm.py",
        "type"    : "gpu",
        "category": "cross",
        # 수정: 64→16 (size 16384, ~2GB VRAM) — OOM 방지
        # GPU VRAM이 충분하면 더 올려도 됨
        "batch"   : {20: 2, 50: 4, 100: 8},
    },
}

# ────────────────────────────────────────────────
# 로깅
# ────────────────────────────────────────────────
LOG_DIR.mkdir(parents=True, exist_ok=True)
log_file = LOG_DIR / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ────────────────────────────────────────────────
# 시스템 제어
# ────────────────────────────────────────────────
def apply_cpu_control():
    log.info("CPU 제어 조건 적용 중...")
    for p in Path("/sys/devices/system/cpu").glob("cpu*/cpufreq/scaling_governor"):
        try:
            p.write_text("performance")
        except PermissionError:
            log.warning(f"  governor 쓰기 실패: {p}")

    turbo = Path("/sys/devices/system/cpu/intel_pstate/no_turbo")
    if turbo.exists():
        try:
            turbo.write_text("1")
            log.info("  Intel Turbo Boost 비활성화")
        except PermissionError:
            log.warning("  Turbo Boost 쓰기 실패 (root 필요)")

    for p in Path("/sys/devices/system/cpu").glob("cpu*/cpuidle/state*/disable"):
        try:
            p.write_text("1")
        except PermissionError:
            pass
    log.info("  C-State 비활성화 완료")


def apply_gpu_control(tdp_w: int = 300):
    log.info("GPU 제어 조건 적용 중...")
    cmds = [
        ["nvidia-smi", "--lock-gpu-clocks=1200,1200"],
        ["nvidia-smi", "--lock-memory-clocks=9251,9251"],
        ["nvidia-smi", "-pl", str(tdp_w)],
    ]
    for cmd in cmds:
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            log.warning(f"  GPU 설정 실패: {' '.join(cmd)}")
        else:
            log.info(f"  {' '.join(cmd)} → OK")


# ────────────────────────────────────────────────
# 온도 대기
# ────────────────────────────────────────────────
def wait_for_cooldown(timeout: int = COOLDOWN_TIMEOUT) -> bool:
    log.info("Cooldown 대기 중...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        cpu_t = read_cpu_temp()
        gpu_d = read_gpu_metrics()
        gpu_t = gpu_d.get("temperature.gpu")
        cpu_ok = (cpu_t is None) or (cpu_t <= COOLDOWN_CPU_C)
        gpu_ok = (gpu_t is None) or (gpu_t <= COOLDOWN_GPU_C)
        log.info(f"  온도 → CPU: {cpu_t}°C  GPU: {gpu_t}°C")
        if cpu_ok and gpu_ok:
            log.info("  Cooldown 완료")
            return True
        time.sleep(COOLDOWN_POLL_S)
    log.warning("  Cooldown 타임아웃 — 강제 진행")
    return False


def wait_idle_stable(
    min_duration: int = IDLE_SECS,
    timeout: int = COOLDOWN_TIMEOUT,
) -> bool:
    log.info(f"Idle 안정화 대기 (최소 {min_duration}초 + 온도 기준)")
    start    = time.time()
    deadline = start + timeout
    while True:
        elapsed = time.time() - start
        cpu_t   = read_cpu_temp()
        gpu_d   = read_gpu_metrics()
        gpu_t   = gpu_d.get("temperature.gpu")
        cpu_ok  = (cpu_t is None) or (cpu_t <= COOLDOWN_CPU_C)
        gpu_ok  = (gpu_t is None) or (gpu_t <= COOLDOWN_GPU_C)
        log.info(f"  Idle 경과: {elapsed:.1f}s  CPU: {cpu_t}°C  GPU: {gpu_t}°C")
        if elapsed >= min_duration and cpu_ok and gpu_ok:
            log.info("  Idle 안정 조건 만족")
            return True
        if time.time() > deadline:
            log.warning("  Idle 대기 타임아웃 → 강제 진행")
            return False
        time.sleep(COOLDOWN_POLL_S)


# ────────────────────────────────────────────────
# 워크로드 기동
# ────────────────────────────────────────────────
def build_command(name: str, wl: dict, load: int) -> list:
    script = str(wl["script"])
    cores  = CPU_CORES[load]

    if name == "gemm_cpu":
        threads = wl["threads"][load]
        return [
            "taskset", "-c", cores,
            "numactl", "--cpunodebind=0", "--membind=0",
            sys.executable, script,
            "--device", "cpu",
            "--threads", str(threads),
            "--size", "4096",
            "--loop",
        ]

    if name == "gemm_gpu":
        batch = wl["batch"][load]
        size  = batch * 1024
        log.info(f"  gemm_gpu: batch={batch}, size={size} "
                 f"(예상 VRAM ≈ {2*size*size*4/1e9:.2f} GB)")
        return [
            sys.executable, script,
            "--device", "gpu",
            "--size", str(size),
            "--loop",
        ]

    if wl["type"] == "cpu":
        threads = wl["threads"][load]
        return [
            "taskset", "-c", cores,
            "numactl", "--cpunodebind=0", "--membind=0",
            sys.executable, script,
            "--threads", str(threads),
            "--loop",
        ]

    if wl["type"] == "gpu":
        batch = wl["batch"][load]
        return [
            sys.executable, script,
            "--batch-size", str(batch),
            "--loop",
        ]

    raise RuntimeError(f"알 수 없는 워크로드 유형: {wl['type']}")


def launch_workload(name: str, wl: dict, load: int) -> subprocess.Popen:
    cmd = build_command(name, wl, load)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    log.info(f"  워크로드 실행: {' '.join(cmd)}")
    return subprocess.Popen(
    cmd,
    stdout=None,
    stderr=None,
    env=env,
    preexec_fn=os.setsid,
)


def kill_workload(proc: subprocess.Popen):
    if proc and proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=10)
        except (subprocess.TimeoutExpired, ProcessLookupError):
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass


# ────────────────────────────────────────────────
# 단일 실험 사이클
# ────────────────────────────────────────────────
def run_single_experiment(
    name       : str,
    wl         : dict,
    repeat     : int,
    enable_dram: bool = False,
):
    result_dir = RESULTS_DIR / name / f"rep{repeat:02d}"
    result_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"\n{'='*60}")
    log.info(f"실험 시작: {name}  repeat={repeat}  결과→{result_dir}")

    wait_idle_stable(IDLE_SECS)

    for load in LOAD_LEVELS:
        cores_str = CPU_CORES.get(load, "")
        log.info(f"\n--- 부하 {load}%  할당코어: {cores_str} ---")

        proc = launch_workload(name, wl, load)

        # 워크로드 조기 종료 확인
        time.sleep(2)
        if proc.poll() is not None:
            log.error(f"  워크로드 시작 실패 (2초 내 종료): {name} load={load}")
            log.error(f"  → gemm_gpu OOM 가능성 확인 필요. batch 값을 줄이세요.")
            continue

        log.info(f"  Warmup {STABILIZE_SECS}초...")
        time.sleep(STABILIZE_SECS - 2)  # 위에서 2초 소비

        # 워크로드 살아있는지 재확인
        if proc.poll() is not None:
            log.error(f"  워크로드가 Warmup 중 종료됨: {name} load={load}")
            continue

        csv_path = str(result_dir / f"load{load}.csv")
        mc = MetricCollector(
            output_csv     = csv_path,
            interval       = SAMPLE_INTERVAL,
            workload       = name,
            load_level     = load,
            repeat         = repeat,
            assigned_cores = cores_str,
            collect_dram   = enable_dram,
        )

        mc.start()
        log.info(f"  측정 {MEASURE_SECS}초...")
        time.sleep(MEASURE_SECS)
        rows = mc.stop() or []

        kill_workload(proc)
        log.info(f"  부하 {load}% 완료: {len(rows)}샘플  → {csv_path}")

        if load != LOAD_LEVELS[-1]:
            log.info("  다음 부하 전환 전 10초 대기...")
            time.sleep(10)

    log.info(f"\n실험 사이클 완료 → Cooldown")
    wait_for_cooldown()


# ────────────────────────────────────────────────
# 전체 실험 루프
# ────────────────────────────────────────────────
def run_all(
    target     : str = None,
    repeats    : int = REPEAT,
    enable_dram: bool = False,
):
    targets = {target: WORKLOADS[target]} if target else WORKLOADS

    for name, wl in targets.items():
        log.info(f"\n{'#'*60}")
        log.info(f"워크로드: {name}  category={wl['category']}  반복={repeats}회")

        if not wl["script"].exists():
            log.error(f"  스크립트 없음: {wl['script']}  → 스킵")
            continue

        for rep in range(1, repeats + 1):
            log.info(f"\n[반복 {rep}/{repeats}]")
            try:
                run_single_experiment(name, wl, rep, enable_dram=enable_dram)
            except KeyboardInterrupt:
                log.warning("  사용자 중단 (Ctrl+C)")
                sys.exit(0)
            except Exception as e:
                log.exception(f"  오류 발생: {e}")

    log.info(f"\n모든 실험 완료 → {RESULTS_DIR}")


# ────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="실험 실행 오케스트레이터 v4-fix")
    parser.add_argument(
        "--workload", "-w",
        choices=list(WORKLOADS.keys()) + ["all"],
        default="all",
        help="실행할 워크로드 (기본: all). 예: --workload gemm_gpu",
    )
    parser.add_argument("--repeat",       "-r", type=int, default=REPEAT)
    parser.add_argument("--tdp",                type=int, default=300,
                        help="GPU TDP (W)")
    parser.add_argument("--skip-control",       action="store_true",
                        help="CPU/GPU 제어 조건 건너뜀 (개발용)")
    parser.add_argument("--dram-bw",            action="store_true",
                        help="DRAM 대역폭 수집 (perf_event_paranoid ≤ 1 필요)")
    args = parser.parse_args()

    if not args.skip_control:
        apply_cpu_control()
        apply_gpu_control(tdp_w=args.tdp)

    target = None if args.workload == "all" else args.workload
    run_all(target=target, repeats=args.repeat, enable_dram=args.dram_bw)