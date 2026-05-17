#!/usr/bin/env python3
"""
collect_metric.py  (v3 - 버그 수정판)

수정 사항:
  1. total_rapl_power_w ≠ cpu_power_w 로 분리
     - cpu_power_w    : package 도메인만 (코어+언코어)
     - total_rapl_power_w : package + dram 도메인 합산
  2. sample_dt_s 컬럼 추가 → 에너지 계산 정확도 향상
  3. cpu_busy_pct : 할당 코어 평균 사용률 (정상 동작 확인됨, 유지)

수집 컬럼:
  timestamp, elapsed_s, workload, load_level, repeat, assigned_cores,
  sample_dt_s,
  cpu_power_w, total_rapl_power_w, gpu_power_w,
  cpu_busy_pct, cpu_freq_mhz,
  gpu_util_pct, gpu_mem_util_pct, gpu_mem_used_mb,
  gpu_mem_bw_gb_s, gpu_sm_clock_mhz, gpu_mem_clock_mhz,
  dram_bw_gb_s,
  cpu_temp_c, gpu_temp_c
"""

import csv
import os
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

import psutil

# ── nvml ─────────────────────────────────────────────────────────────────────
try:
    import pynvml
    pynvml.nvmlInit()
    _NVML_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
    _NVML_OK     = True
except Exception:
    _NVML_OK     = False
    _NVML_HANDLE = None

# ── RAPL 도메인 분리 ──────────────────────────────────────────────────────────
# package 도메인 (intel-rapl:N, 콜론 1개)
_RAPL_PKG = sorted(
    [p for p in Path("/sys/class/powercap").glob("intel-rapl:*")
     if p.name.count(":") == 1],
    key=lambda p: p.name,
)

# dram 서브도메인 (intel-rapl:N:M 중 name 파일에 "dram" 포함)
def _is_dram(p: Path) -> bool:
    try:
        return "dram" in (p / "name").read_text().strip().lower()
    except Exception:
        return False

_RAPL_DRAM = sorted(
    [p for p in Path("/sys/class/powercap").glob("intel-rapl:*:*")
     if _is_dram(p)],
    key=lambda p: p.name,
)

# ── uncore IMC ───────────────────────────────────────────────────────────────
_IMC_DIRS = sorted(
    Path("/sys/bus/event_source/devices").glob("uncore_imc_*")
)


# ─────────────────────────────────────────────────────────────────────────────
# CPU 온도
# ─────────────────────────────────────────────────────────────────────────────
def read_cpu_temp() -> Optional[float]:
    try:
        temps = psutil.sensors_temperatures()
        for key in ("coretemp", "k10temp", "zenpower", "cpu_thermal"):
            if key in temps:
                pkg = [t for t in temps[key]
                       if "package" in t.label.lower() or t.label == ""]
                return (pkg[0] if pkg else temps[key][0]).current
    except Exception:
        pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
# RAPL 에너지 읽기
# ─────────────────────────────────────────────────────────────────────────────
def _rapl_uj(path: Path) -> Optional[int]:
    try:
        return int((path / "energy_uj").read_text())
    except Exception:
        return None


def _rapl_diff_w(e0_list, e1_list, dt: float) -> float:
    """energy_uj 차분 목록 → 전력(W). overflow 처리 포함."""
    if dt <= 0:
        return float("nan")
    total_uj = 0
    valid = False
    for a, b in zip(e0_list, e1_list):
        if a is None or b is None:
            continue
        diff = b - a
        if diff < 0:
            diff += 2 ** 32   # counter wrap
        total_uj += diff
        valid = True
    return total_uj / dt / 1e6 if valid else float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# GPU 지표
# ─────────────────────────────────────────────────────────────────────────────
def _nvml_gpu_util_pct() -> float:
    if not _NVML_OK:
        return float("nan")
    h = _NVML_HANDLE
    try:
        _, samples = pynvml.nvmlDeviceGetSamples(h, 1, 0)
        if samples:
            vals = [s.sampleValue.uiVal for s in samples]
            return float(sum(vals) / len(vals))
    except (pynvml.NVMLError, AttributeError):
        pass
    try:
        return float(pynvml.nvmlDeviceGetUtilizationRates(h).gpu)
    except pynvml.NVMLError:
        return float("nan")


def _nvml_mem_util_pct() -> float:
    if not _NVML_OK:
        return float("nan")
    h = _NVML_HANDLE
    try:
        _, samples = pynvml.nvmlDeviceGetSamples(h, 2, 0)
        if samples:
            vals = [s.sampleValue.uiVal for s in samples]
            return float(sum(vals) / len(vals))
    except (pynvml.NVMLError, AttributeError):
        pass
    try:
        return float(pynvml.nvmlDeviceGetUtilizationRates(h).memory)
    except pynvml.NVMLError:
        return float("nan")


def read_gpu_metrics() -> dict:
    out = dict(
        power_w       = float("nan"),
        util_pct      = float("nan"),
        mem_util_pct  = float("nan"),
        mem_used_mb   = float("nan"),
        mem_bw_gb_s   = float("nan"),
        sm_clock_mhz  = float("nan"),
        mem_clock_mhz = float("nan"),
        temp_c        = float("nan"),
        **{"temperature.gpu": None},
    )
    if not _NVML_OK:
        return out
    h = _NVML_HANDLE
    try:
        try:
            out["power_w"] = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
        except pynvml.NVMLError:
            pass

        out["util_pct"]    = _nvml_gpu_util_pct()
        out["mem_util_pct"] = _nvml_mem_util_pct()

        try:
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            out["mem_used_mb"] = mem.used / 1024 / 1024
        except pynvml.NVMLError:
            pass

        try:
            rx = pynvml.nvmlDeviceGetPcieThroughput(
                h, pynvml.NVML_PCIE_UTIL_RX_BYTES)
            tx = pynvml.nvmlDeviceGetPcieThroughput(
                h, pynvml.NVML_PCIE_UTIL_TX_BYTES)
            out["mem_bw_gb_s"] = (rx + tx) / 1024 / 1024
        except (pynvml.NVMLError, AttributeError):
            pass

        try:
            out["sm_clock_mhz"]  = float(
                pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM))
            out["mem_clock_mhz"] = float(
                pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_MEM))
        except pynvml.NVMLError:
            pass

        try:
            t = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
            out["temp_c"]          = float(t)
            out["temperature.gpu"] = float(t)
        except pynvml.NVMLError:
            pass

    except Exception:
        pass
    return out


# ─────────────────────────────────────────────────────────────────────────────
# CPU 사용률 (할당 코어 기준)
# ─────────────────────────────────────────────────────────────────────────────
def _parse_cores(cores_str: str) -> list:
    """'0-3,8-11' → [0,1,2,3,8,9,10,11]"""
    result = []
    for part in cores_str.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            result.extend(range(int(a), int(b) + 1))
        elif part:
            result.append(int(part))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# DRAM 대역폭
# ─────────────────────────────────────────────────────────────────────────────
def read_dram_bw_gb_s(interval: float = 0.5) -> float:
    if not _IMC_DIRS:
        return float("nan")

    events = []
    for imc in _IMC_DIRS:
        name = imc.name
        events.append(f"{name}/event=0x04,umask=0x0c/")
        events.append(f"{name}/event=0x04,umask=0x30/")

    cmd = [
        "perf", "stat",
        "-e", ",".join(events),
        "-a", "--no-big-num", "-x", ",",
        "sleep", str(interval),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=interval + 5)
        total_cas = 0
        for line in r.stderr.splitlines():
            parts = line.split(",")
            if not parts:
                continue
            try:
                total_cas += int(parts[0].strip())
            except ValueError:
                pass
        if total_cas <= 0:
            return float("nan")
        return round(total_cas * 64 / interval / 1e9, 3)
    except Exception:
        return float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# CSV 컬럼 정의
# ─────────────────────────────────────────────────────────────────────────────
CSV_FIELDS = [
    "timestamp", "elapsed_s",
    "workload", "load_level", "repeat", "assigned_cores",
    "sample_dt_s",                           # ← 신규: 실제 샘플 간격
    "cpu_power_w", "total_rapl_power_w",     # pkg만 / pkg+dram
    "gpu_power_w",
    "cpu_busy_pct", "cpu_freq_mhz",
    "gpu_util_pct", "gpu_mem_util_pct", "gpu_mem_used_mb",
    "gpu_mem_bw_gb_s", "gpu_sm_clock_mhz", "gpu_mem_clock_mhz",
    "dram_bw_gb_s",
    "cpu_temp_c", "gpu_temp_c",
]


# ─────────────────────────────────────────────────────────────────────────────
# MetricCollector
# ─────────────────────────────────────────────────────────────────────────────
class MetricCollector:
    """
    백그라운드 스레드에서 메트릭을 수집해 CSV 저장.

    Parameters
    ----------
    output_csv     : 저장 경로
    interval       : 샘플 간격 (초)
    workload       : 워크로드 이름
    load_level     : 부하 레벨 (20 / 50 / 100)
    repeat         : 반복 번호
    assigned_cores : 할당 코어 문자열 (예: "0-3") 또는 None
    collect_dram   : True 면 perf stat 으로 DRAM BW 측정
    """

    def __init__(
        self,
        output_csv    : str,
        interval      : float = 0.1,
        workload      : str   = "unknown",
        load_level    : int   = 0,
        repeat        : int   = 0,
        assigned_cores: Optional[str] = None,
        collect_dram  : bool  = False,
    ):
        self.output_csv     = output_csv
        self.interval       = interval
        self.workload       = workload
        self.load_level     = load_level
        self.repeat         = repeat
        self.assigned_cores = assigned_cores or ""
        self.collect_dram   = collect_dram
        self._core_list     = _parse_cores(assigned_cores) if assigned_cores else []
        self._rows: list            = []
        self._stop_evt              = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._start_time: float     = 0.0

    # ── 단일 샘플 ─────────────────────────────────────────────────────────────
    def _sample(self) -> dict:
        now = time.time()

        # ── RAPL e0 + 타이머 시작 (동시에) ───────────────────────────────────
        e0_pkg  = [_rapl_uj(p) for p in _RAPL_PKG]
        e0_dram = [_rapl_uj(p) for p in _RAPL_DRAM]
        t0      = time.time()                  # e0 직후에 t0 기록

        # cpu_percent 기준점 (이전 구간 대비값을 버림)
        psutil.cpu_percent(percpu=True)

        # ── sleep 또는 DRAM BW 측정 ───────────────────────────────────────────
        if self.collect_dram:
            dram_bw = read_dram_bw_gb_s(self.interval)
        else:
            time.sleep(self.interval)
            dram_bw = float("nan")

        # ── RAPL e1 + cpu_percent 완료 ────────────────────────────────────────
        e1_pkg  = [_rapl_uj(p) for p in _RAPL_PKG]
        e1_dram = [_rapl_uj(p) for p in _RAPL_DRAM]
        per_cpu = psutil.cpu_percent(percpu=True)
        dt      = time.time() - t0             # 실제 샘플 간격

        # ── 전력 계산 ─────────────────────────────────────────────────────────
        cpu_pwr   = _rapl_diff_w(e0_pkg,  e1_pkg,  dt)   # package only
        dram_pwr  = _rapl_diff_w(e0_dram, e1_dram, dt)   # dram only

        # total_rapl = package + dram (dram 도메인 없으면 package만)
        import math
        if not math.isnan(dram_pwr):
            total_rapl = (cpu_pwr if not math.isnan(cpu_pwr) else 0) + dram_pwr
        else:
            total_rapl = cpu_pwr

        # ── CPU busy (할당 코어 기준) ─────────────────────────────────────────
        if self._core_list:
            valid    = [per_cpu[i] for i in self._core_list if i < len(per_cpu)]
            cpu_busy = sum(valid) / len(valid) if valid else float("nan")
        else:
            cpu_busy = sum(per_cpu) / len(per_cpu) if per_cpu else float("nan")

        # ── CPU 주파수 ────────────────────────────────────────────────────────
        cpu_freq = float("nan")
        try:
            freqs = psutil.cpu_freq(percpu=True)
            if freqs:
                if self._core_list:
                    fv = [freqs[i].current for i in self._core_list
                          if i < len(freqs)]
                    cpu_freq = sum(fv) / len(fv) if fv else float("nan")
                else:
                    cpu_freq = freqs[0].current
            else:
                f = psutil.cpu_freq()
                cpu_freq = f.current if f else float("nan")
        except Exception:
            pass

        # ── GPU 지표 ─────────────────────────────────────────────────────────
        gpu      = read_gpu_metrics()
        cpu_temp = read_cpu_temp()

        def _fmt(v):
            if isinstance(v, float) and v != v:   # nan
                return ""
            if isinstance(v, float):
                return round(v, 4)
            return v

        return {
            "timestamp"          : time.strftime("%Y-%m-%dT%H:%M:%S"),
            "elapsed_s"          : round(now - self._start_time, 3),
            "workload"           : self.workload,
            "load_level"         : self.load_level,
            "repeat"             : self.repeat,
            "assigned_cores"     : self.assigned_cores,
            "sample_dt_s"        : round(dt, 4),          # ← 신규
            "cpu_power_w"        : _fmt(cpu_pwr),          # package only
            "total_rapl_power_w" : _fmt(total_rapl),       # package + dram
            "gpu_power_w"        : _fmt(gpu["power_w"]),
            "cpu_busy_pct"       : _fmt(cpu_busy),
            "cpu_freq_mhz"       : _fmt(cpu_freq),
            "gpu_util_pct"       : _fmt(gpu["util_pct"]),
            "gpu_mem_util_pct"   : _fmt(gpu["mem_util_pct"]),
            "gpu_mem_used_mb"    : _fmt(gpu["mem_used_mb"]),
            "gpu_mem_bw_gb_s"    : _fmt(gpu["mem_bw_gb_s"]),
            "gpu_sm_clock_mhz"   : _fmt(gpu["sm_clock_mhz"]),
            "gpu_mem_clock_mhz"  : _fmt(gpu["mem_clock_mhz"]),
            "dram_bw_gb_s"       : _fmt(dram_bw),
            "cpu_temp_c"         : _fmt(cpu_temp) if cpu_temp is not None else "",
            "gpu_temp_c"         : _fmt(gpu["temp_c"]),
        }

    # ── 수집 루프 ─────────────────────────────────────────────────────────────
    def _run(self):
        Path(self.output_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            while not self._stop_evt.is_set():
                try:
                    row = self._sample()
                    writer.writerow(row)
                    f.flush()
                    self._rows.append(row)
                except Exception as e:
                    import logging
                    logging.getLogger(__name__).warning(f"샘플 수집 실패: {e}")

    def start(self):
        self._start_time = time.time()
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> list:
        self._stop_evt.set()
        if self._thread:
            self._thread.join(timeout=10)
        return self._rows