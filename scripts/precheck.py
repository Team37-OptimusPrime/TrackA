#!/usr/bin/env python3
"""
precheck.py
실험 실행 전 체크리스트 검증 스크립트
- 워크로드 스크립트 존재 여부
- 시스템 제어 조건 (DVFS, Turbo Boost, C-State, GPU 클럭/Power limit)
- 필요 라이브러리 임포트 가능 여부
- CPU 코어 할당 / GPU 할당 확인
- 부하 생성 도구(taskset, nvidia-smi) 사용 가능 여부
- 수집 메트릭 도구(perf, sensors) 존재 여부
"""

import os
import sys
import subprocess
import shutil
import importlib
import glob
from pathlib import Path

# ────────────────────────────────────────────────
# 경로 설정
# ────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[1]
WORKLOAD_DIR = BASE_DIR / "workloads"
SCRIPT_DIR   = BASE_DIR / "scripts"

WORKLOAD_SCRIPTS = {
    "resnet"              : WORKLOAD_DIR / "resnet.py",
    "gpt2"                : WORKLOAD_DIR / "gpt2_infer.py",
    "gromacs"             : WORKLOAD_DIR / "gromacs_run.py",
    "matmul"              : WORKLOAD_DIR / "matmul.py",
    "gemm"                : WORKLOAD_DIR / "gemm.py",
}

GEMM_SCRIPT = WORKLOAD_DIR / "gemm.py"

REQUIRED_PYTHON_LIBS = [
    "torch", "torchvision",
    "transformers",
    "numpy", "pandas",
    "psutil", "subprocess",
]

REQUIRED_BINS = ["taskset", "numactl", "perf", "sensors", "nvidia-smi"]

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
WARN = "\033[93m[WARN]\033[0m"
INFO = "\033[94m[INFO]\033[0m"

results = {}   # key -> bool

# ────────────────────────────────────────────────
# 헬퍼
# ────────────────────────────────────────────────
def run(cmd):
    return subprocess.run(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

def check(name: str, ok: bool, detail: str = ""):
    tag = PASS if ok else FAIL
    print(f"  {tag} {name}" + (f"  →  {detail}" if detail else ""))
    results[name] = ok
    return ok

def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

# ────────────────────────────────────────────────
# 1. 워크로드 스크립트 존재 여부
# ────────────────────────────────────────────────
def check_workload_scripts():
    section("1. 워크로드 스크립트 존재 여부")
    for name, script_path in WORKLOAD_SCRIPTS.items():
        check(f"워크로드 스크립트: {name}", script_path.exists(), str(script_path))

    # gemm.py는 --device cpu / --device gpu 양쪽 인수 파싱 및 1회 실행 검증
    if GEMM_SCRIPT.exists():
        for device in ["cpu", "gpu"]:
           r = run(f"{sys.executable} {GEMM_SCRIPT} --device {device} --size 64")
           ok = r.returncode == 0
           detail = r.stdout.strip().splitlines()[-1] if r.stdout else "OK"
           check(f"gemm.py --device {  device} 실행 검증", ok, detail)
    else:
        check("gemm.py --device 실행 검증 (스킵: 파일 없음)", False, str(GEMM_SCRIPT))


# ────────────────────────────────────────────────
# 2. Python 라이브러리
# ────────────────────────────────────────────────
def check_python_libs():
    section("2. Python 라이브러리 임포트")
    for lib in REQUIRED_PYTHON_LIBS:
        try:
            importlib.import_module(lib)
            check(lib, True)
        except Exception as e:
            check(lib, False, str(e))

    # PyTorch CUDA 가용성
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        check("torch.cuda.is_available()", cuda_ok,
              f"device count={torch.cuda.device_count()}" if cuda_ok else "CUDA 없음")
        if cuda_ok:
            cvd = os.environ.get("CUDA_VISIBLE_DEVICES")

        if cvd is None:
            cvd = "0"

        check(
            "CUDA_VISIBLE_DEVICES=0",
            cvd == "0",
            f"현재: {cvd}"
        )
    except ImportError:
        check("torch CUDA 검사", False, "torch 없음")

# ────────────────────────────────────────────────
# 3. 시스템 바이너리
# ────────────────────────────────────────────────
def check_binaries():
    section("3. 필수 바이너리 / 도구")
    for b in REQUIRED_BINS:
        check(b, shutil.which(b) is not None, shutil.which(b) or "not found")

# ────────────────────────────────────────────────
# 4. CPU 제어 조건
# ────────────────────────────────────────────────
def check_cpu_control():
    section("4. CPU 실험 통제 조건")

    # 4-1. cpufreq governor == performance
    gov_files = glob.glob("/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor")
    if gov_files:
        govs = set(Path(f).read_text().strip() for f in gov_files)
        check("cpufreq governor = performance",
              govs == {"performance"}, f"현재: {govs}")
    else:
        print(f"  {WARN} cpufreq 파일 없음 (가상환경이거나 governor 미지원)")

    # 4-2. Intel Turbo Boost 비활성화
    turbo_path = Path("/sys/devices/system/cpu/intel_pstate/no_turbo")
    if turbo_path.exists():
        val = turbo_path.read_text().strip()
        check("Intel Turbo Boost 비활성화 (no_turbo=1)", val == "1", f"현재: {val}")
    else:
        # AMD / non-pstate 경우
        alt = Path("/sys/devices/system/cpu/cpufreq/boost")
        if alt.exists():
            val = alt.read_text().strip()
            check("CPU Boost 비활성화 (boost=0)", val == "0", f"현재: {val}")
        else:
            print(f"  {WARN} Turbo Boost 제어 파일 없음 — BIOS 설정 필요")

    # 4-3. C-State 비활성화
    cstate_files = glob.glob("/sys/devices/system/cpu/cpu*/cpuidle/state*/disable")
    if cstate_files:
        not_disabled = [f for f in cstate_files if Path(f).read_text().strip() != "1"]
        check("C-State 비활성화", len(not_disabled) == 0,
              f"미비활성화 항목 {len(not_disabled)}개" if not_disabled else "")
    else:
        print(f"  {WARN} C-State 파일 없음")

    # 4-4. CPU 코어 수 확인 (20코어 할당 가능 여부)
    r = run("nproc")
    nproc = int(r.stdout.strip()) if r.returncode == 0 else 0
    check("CPU 코어 ≥ 20 (taskset 0-19 할당 가능)", nproc >= 20, f"현재 nproc={nproc}")

    # 4-5. NUMA 단일 노드 바인딩 가능
    r = run("numactl --hardware 2>/dev/null | grep 'available:'")
    numa_ok = "available:" in r.stdout
    check("numactl NUMA 정보 확인", numa_ok, r.stdout.strip()[:80] if numa_ok else "")

# ────────────────────────────────────────────────
# 5. GPU 제어 조건
# ────────────────────────────────────────────────
def check_gpu_control():
    section("5. GPU 실험 통제 조건")

    if shutil.which("nvidia-smi") is None:
        print(f"  {WARN} nvidia-smi 없음 — GPU 검사 전체 스킵")
        return

    # 5-1. GPU 1장 이상 인식
    r = run("nvidia-smi --query-gpu=name,count --format=csv,noheader")
    gpu_ok = r.returncode == 0 and r.stdout.strip() != ""
    check("GPU 인식", gpu_ok, r.stdout.strip()[:80])

    # 5-2. GPU 클럭 고정 여부 (applications.clocks.gr)
    r = run("nvidia-smi --query-gpu=clocks_throttle_reasons.gpu_is_throttled "
            "--format=csv,noheader")
    if r.returncode == 0:
        throttled = r.stdout.strip().lower()
        check("GPU 쓰로틀링 없음", "not active" in throttled or "0" in throttled,
              f"현재: {throttled}")

    # 5-3. GPU lock-gpu-clocks 적용 확인 (applications.clocks 출력)
    r = run("nvidia-smi --query-gpu=clocks.gr,clocks.mem --format=csv,noheader")
    check("GPU 클럭 조회 가능", r.returncode == 0, r.stdout.strip()[:80])

    # 5-4. Power limit 설정 확인
    r = run("nvidia-smi --query-gpu=power.limit,power.draw --format=csv,noheader")
    check("GPU Power limit 조회 가능", r.returncode == 0, r.stdout.strip()[:80])

    # 5-6. 메모리 클럭 고정 확인
    r = run("nvidia-smi --query-gpu=clocks.mem --format=csv,noheader")
    check("GPU 메모리 클럭 조회 가능", r.returncode == 0, r.stdout.strip()[:40])

# ────────────────────────────────────────────────
# 6. 부하 생성 도구
# ────────────────────────────────────────────────
def check_load_tools():
    section("6. 부하 생성 도구 확인")

    # taskset 동작 테스트
    r = run("taskset -c 0-3 echo ok")
    check("taskset 동작 (0-3 코어)", r.returncode == 0, r.stdout.strip())

    # numactl 동작 테스트
    r = run("numactl --cpunodebind=0 --membind=0 echo ok")
    check("numactl 동작 (node 0)", r.returncode == 0, r.stdout.strip())

    # nvidia-smi loop 조회 (GPU 배치 조절 가능성 확인)
    r = run("nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader")
    check("nvidia-smi utilization 조회", r.returncode == 0, r.stdout.strip()[:40])

# ────────────────────────────────────────────────
# 7. 수집 메트릭 도구
# ────────────────────────────────────────────────
def check_metric_tools():
    section("7. 수집 메트릭 도구")

    # RAPL
    rapl_path = Path("/sys/class/powercap/intel-rapl")
    check("RAPL 인터페이스 존재", rapl_path.exists(), str(rapl_path))

    # perf stat 기본 동작
    r = run("perf stat -e cycles,instructions echo test 2>&1")
    check("perf 이벤트 cycles/instructions",
      "not supported" not in r.stderr and r.returncode == 0,
      "OK" if r.returncode == 0 else r.stderr.strip()[:80])

    # perf 이벤트 가용 여부 (mem_load_retired.l3_miss)
    r = run("perf stat -e cache-misses echo test 2>&1")
    check("perf 이벤트 cache-misses",
      "not supported" not in r.stderr and r.returncode == 0,
      "OK" if r.returncode == 0 else r.stderr.strip()[:80])

    # sensors (CPU 온도)
    r = run("sensors 2>/dev/null | head -5")
    check("sensors (CPU 온도)", r.returncode == 0 and r.stdout.strip() != "",
          r.stdout.strip()[:80])

    # nvidia-smi power.draw
    r = run("nvidia-smi --query-gpu=power.draw,temperature.gpu,"
            "utilization.gpu,utilization.memory --format=csv,noheader")
    check("nvidia-smi 전력/온도/활용률 조회", r.returncode == 0, r.stdout.strip()[:80])

    # /proc/stat 존재 여부
    check("/proc/stat 존재 (CPU Busy 계산)", Path("/proc/stat").exists())

# ────────────────────────────────────────────────
# 8. 실험 사이클 파라미터 확인 (설정값 출력)
# ────────────────────────────────────────────────
def show_experiment_params():
    section("8. 실험 사이클 파라미터 (참고용 출력)")
    params = {
        "Idle 시간"          : "2분",
        "Warmup"             : "측정 제외",
        "부하 단계"          : "20% → 50% → 100%",
        "각 부하 안정화"     : "30초",
        "각 부하 측정"       : "1분",
        "반복 횟수"          : "5회 (체크리스트 기준 3회)",
        "Cooldown 기준 CPU"  : "≤ 45°C",
        "Cooldown 기준 GPU"  : "≤ 50°C",
        "Sampling rate"      : "100ms",
    }
    for k, v in params.items():
        print(f"  {INFO} {k:<30} {v}")

# ────────────────────────────────────────────────
# 최종 요약
# ────────────────────────────────────────────────
def summary():
    section("최종 체크리스트 요약")
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    failed = total - passed
    print(f"  총 항목: {total}  |  통과: {passed}  |  실패: {failed}")
    if failed:
        print(f"\n  {FAIL} 아래 항목을 수정하세요:")
        for name, ok in results.items():
            if not ok:
                print(f"    - {name}")
    else:
        print(f"\n  {PASS} 모든 체크 통과 — 실험 시작 가능")
    return failed == 0

# ────────────────────────────────────────────────
# main
# ────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  실험 전 체크리스트 (precheck.py)")
    print("=" * 60)

    check_workload_scripts()
    check_python_libs()
    check_binaries()
    check_cpu_control()
    check_gpu_control()
    check_load_tools()
    check_metric_tools()
    show_experiment_params()
    ok = summary()
    print("DEBUG CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))

    sys.exit(0 if ok else 1)
