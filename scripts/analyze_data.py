#!/usr/bin/env python3
"""
analyze_data.py  (버그 수정판)
실험 결과 데이터 분석 및 시각화

수정 사항:
  1. agg_per_run 컬럼명 불일치 수정
     - named aggregation 결과 컬럼명을 명시적으로 통일
  2. energy_j 계산: 고정 0.1s → 실제 sample_dt_s 사용
     (구버전 CSV 호환: sample_dt_s 없으면 0.1s fallback)
  3. agg_mean_std에서 cpu_busy_pct 집계 경로 수정

사용법:
  python3 analyze_data.py --results /home/optimus/siyeon/exp_v3/results --out ./figures
  python3 analyze_data.py --dummy
"""

import argparse
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import LeaveOneGroupOut

warnings.filterwarnings("ignore")

# ────────────────────────────────────────────────
# 상수 / 메타
# ────────────────────────────────────────────────
CATEGORY_ORDER  = ["GPU-centric", "CPU-centric", "cross"]
CATEGORY_LABELS = {
    "GPU-centric": "GPU-centric",
    "CPU-centric": "CPU-centric",
    "cross"      : "교차 (GEMM)",
}

WORKLOAD_META = {
    "resnet"  : {"category": "GPU-centric"},
    "gpt2"    : {"category": "GPU-centric"},
    "matmul"  : {"category": "CPU-centric"},
    "gromacs" : {"category": "CPU-centric"},
    "gemm_cpu": {"category": "cross"},
    "gemm_gpu": {"category": "cross"},
}

WL_COLORS = {
    "resnet"  : "#e41a1c",
    "gpt2"    : "#377eb8",
    "matmul"  : "#4daf4a",
    "gromacs" : "#984ea3",
    "gemm_cpu": "#ff7f00",
    "gemm_gpu": "#a65628",
}
WL_MARKERS = {
    "resnet"  : "o",
    "gpt2"    : "s",
    "matmul"  : "^",
    "gromacs" : "D",
    "gemm_cpu": "P",
    "gemm_gpu": "X",
}

FIG_DPI = 150
_RE_LOAD = re.compile(r"load(\d+)\.csv$", re.IGNORECASE)


# ────────────────────────────────────────────────
# 데이터 로드
# ────────────────────────────────────────────────
def load_results(results_dir: Path) -> pd.DataFrame:
    frames = []
    for csv_path in sorted(results_dir.glob("*/rep*/load*.csv")):
        parts = csv_path.parts
        try:
            workload_from_path = parts[-3]
            repeat_from_path   = int(re.sub(r"\D", "", parts[-2]))
        except (IndexError, ValueError):
            print(f"[WARN] 경로 파싱 실패, 스킵: {csv_path}")
            continue

        m = _RE_LOAD.search(csv_path.name)
        if not m:
            continue
        load_from_path = int(m.group(1))

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"[WARN] CSV 읽기 실패: {csv_path}  {e}")
            continue

        if df.empty:
            continue

        if "workload"   not in df.columns: df["workload"]   = workload_from_path
        if "load_level" not in df.columns: df["load_level"] = load_from_path
        if "repeat"     not in df.columns: df["repeat"]     = repeat_from_path

        df["load_level"] = (pd.to_numeric(df["load_level"], errors="coerce")
                              .fillna(load_from_path).astype(int))
        df["repeat"]     = (pd.to_numeric(df["repeat"],     errors="coerce")
                              .fillna(repeat_from_path).astype(int))

        frames.append(df)

    if not frames:
        raise FileNotFoundError(
            f"CSV 파일 없음 (패턴: {results_dir}/*/rep*/load*.csv)\n"
            "  --dummy 옵션으로 형식 검증 가능"
        )

    df = pd.concat(frames, ignore_index=True)

    # 메타 병합
    df["category"] = df["workload"].map(
        lambda w: WORKLOAD_META.get(w, {}).get("category", "unknown")
    )

    # 컬럼 보완 (구버전 CSV 호환)
    for col in ("cpu_power_w", "total_rapl_power_w", "gpu_power_w",
                "cpu_busy_pct", "gpu_util_pct", "gpu_mem_util_pct",
                "cpu_temp_c", "gpu_temp_c", "gpu_sm_clock_mhz"):
        if col not in df.columns:
            df[col] = np.nan

    # 수정: sample_dt_s 없으면 0.1s fallback (구버전 CSV 호환)
    if "sample_dt_s" not in df.columns:
        df["sample_dt_s"] = 0.1
    else:
        df["sample_dt_s"] = pd.to_numeric(df["sample_dt_s"], errors="coerce").fillna(0.1)

    # total_power: RAPL 패키지(+dram) + GPU
    df["total_power_w"] = (
        df["total_rapl_power_w"].fillna(0) + df["gpu_power_w"].fillna(0)
    )

    # 수정: 에너지 = 전력 × 실제 샘플 간격 (고정 0.1 제거)
    df["energy_j"] = df["total_power_w"] * df["sample_dt_s"]

    print(f"[DATA] 로드 완료: {len(df):,}행")
    print(f"       워크로드  : {sorted(df['workload'].unique())}")
    print(f"       부하 레벨 : {sorted(df['load_level'].unique())}")
    print(f"       반복      : {sorted(df['repeat'].unique())}")
    return df


# ────────────────────────────────────────────────
# 집계
# ────────────────────────────────────────────────
def agg_per_run(df: pd.DataFrame) -> pd.DataFrame:
    """
    workload × load_level × repeat 별 평균 집계.

    수정: named aggregation 결과 컬럼명을 agg_mean_std와 일치시킴.
    """
    return df.groupby(
        ["workload", "load_level", "repeat", "category"]
    ).agg(
        cpu_power_w    = ("cpu_power_w",      "mean"),
        gpu_power_w    = ("gpu_power_w",      "mean"),
        total_power_w  = ("total_power_w",    "mean"),
        cpu_busy_pct   = ("cpu_busy_pct",     "mean"),   # ← 수정: 컬럼명 통일
        gpu_util_pct   = ("gpu_util_pct",     "mean"),
        gpu_mem_util   = ("gpu_mem_util_pct", "mean"),
        cpu_temp_c     = ("cpu_temp_c",       "mean"),
        gpu_temp_c     = ("gpu_temp_c",       "mean"),
        energy_j       = ("energy_j",         "sum"),
    ).reset_index()


def agg_mean_std(run: pd.DataFrame) -> pd.DataFrame:
    """
    workload × load_level 별 반복 간 평균±표준편차.

    수정: cpu_busy_pct 집계 소스 컬럼을 agg_per_run 출력과 일치시킴.
    """
    return run.groupby(
        ["workload", "load_level", "category"]
    ).agg(
        cpu_power_mean   = ("cpu_power_w",   "mean"),
        cpu_power_std    = ("cpu_power_w",   "std"),
        gpu_power_mean   = ("gpu_power_w",   "mean"),
        gpu_power_std    = ("gpu_power_w",   "std"),
        total_power_mean = ("total_power_w", "mean"),
        total_power_std  = ("total_power_w", "std"),
        cpu_busy_mean    = ("cpu_busy_pct",  "mean"),   # ← 수정: 소스 컬럼명 일치
        cpu_busy_std     = ("cpu_busy_pct",  "std"),
        gpu_util_mean    = ("gpu_util_pct",  "mean"),
        gpu_mem_mean     = ("gpu_mem_util",  "mean"),
        cpu_temp_mean    = ("cpu_temp_c",    "mean"),
        gpu_temp_mean    = ("gpu_temp_c",    "mean"),
        energy_mean      = ("energy_j",      "mean"),
        energy_std       = ("energy_j",      "std"),
    ).reset_index()


# ────────────────────────────────────────────────
# 회귀 헬퍼
# ────────────────────────────────────────────────
def fit_linear(x, y):
    m = make_pipeline(LinearRegression())
    m.fit(x.reshape(-1, 1), y)
    p    = m.predict(x.reshape(-1, 1))
    r2   = 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)
    rmse = np.sqrt(mean_squared_error(y, p))
    return m, r2, rmse


def fit_poly2(x, y):
    m = make_pipeline(PolynomialFeatures(2, include_bias=False), LinearRegression())
    m.fit(x.reshape(-1, 1), y)
    p    = m.predict(x.reshape(-1, 1))
    r2   = 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)
    rmse = np.sqrt(mean_squared_error(y, p))
    return m, r2, rmse


def regression_overlay(ax, x, y):
    valid = np.isfinite(x) & np.isfinite(y)
    x, y  = x[valid], y[valid]
    if len(x) < 3 or np.std(x) < 1e-9:
        return None
    try:
        xr = np.linspace(x.min(), x.max(), 200).reshape(-1, 1)
        lm, lr2, lrmse = fit_linear(x, y)
        pm, pr2, prmse = fit_poly2(x, y)
        ax.plot(xr, lm.predict(xr), "--", color="gray",  lw=1.5,
                label=f"Linear  R²={lr2:.3f}  RMSE={lrmse:.1f}W")
        ax.plot(xr, pm.predict(xr), "-",  color="black", lw=2.0,
                label=f"Poly-2  R²={pr2:.3f}  RMSE={prmse:.1f}W")
        return dict(linear_r2=lr2, linear_rmse=lrmse, poly2_r2=pr2, poly2_rmse=prmse)
    except np.linalg.LinAlgError:
        return None


def _scatter(ax, df, xcol, ycol):
    for wl, g in df.groupby("workload"):
        ax.scatter(g[xcol], g[ycol],
                   color=WL_COLORS.get(wl, "gray"),
                   marker=WL_MARKERS.get(wl, "o"),
                   label=wl, s=65, alpha=0.85, zorder=3)


# ────────────────────────────────────────────────
# [그래프 1] CPU Busy → CPU 소모 전력
# ────────────────────────────────────────────────
def plot_graph1(agg: pd.DataFrame, out_dir: Path):
    fig = plt.figure(figsize=(16, 9))
    gs  = GridSpec(2, 3, figure=fig, hspace=0.48, wspace=0.32)
    ax_all  = fig.add_subplot(gs[0, :])
    ax_cats = [fig.add_subplot(gs[1, i]) for i in range(3)]

    rows = []

    _scatter(ax_all, agg, "cpu_busy_mean", "cpu_power_mean")
    s = regression_overlay(ax_all,
                           agg["cpu_busy_mean"].values,
                           agg["cpu_power_mean"].values)
    if s: rows.append({"구간": "전체", **s})
    ax_all.set_xlabel("CPU Busy (%)", fontsize=11)
    ax_all.set_ylabel("CPU 소모 전력 (W)", fontsize=11)
    ax_all.set_title("[그래프 1] CPU Busy – CPU 소모 전력 회귀 분석 (전체)",
                     fontsize=13, fontweight="bold")
    ax_all.legend(fontsize=8, ncol=4, loc="upper left")
    ax_all.grid(True, alpha=0.3)

    for ax_c, cat in zip(ax_cats, CATEGORY_ORDER):
        sub = agg[agg["category"] == cat]
        if sub.empty:
            ax_c.set_visible(False)
            continue
        _scatter(ax_c, sub, "cpu_busy_mean", "cpu_power_mean")
        s = regression_overlay(ax_c,
                               sub["cpu_busy_mean"].values,
                               sub["cpu_power_mean"].values)
        if s: rows.append({"구간": CATEGORY_LABELS[cat], **s})
        ax_c.set_title(CATEGORY_LABELS[cat], fontsize=10)
        ax_c.set_xlabel("CPU Busy (%)", fontsize=9)
        ax_c.set_ylabel("CPU 전력 (W)", fontsize=9)
        ax_c.legend(fontsize=7)
        ax_c.grid(True, alpha=0.3)

    _print_table("[그래프 1] 회귀 지표", rows)
    _save(fig, out_dir, "graph1_cpu_busy_power.png")


# ────────────────────────────────────────────────
# [그래프 2] GPU Util → GPU 소모 전력
# ────────────────────────────────────────────────
def plot_graph2(agg: pd.DataFrame, out_dir: Path):
    gpu_df = agg[
        (agg["category"].isin(["GPU-centric", "cross"])) &
        (agg["workload"] != "gemm_cpu")
    ].copy()

    if gpu_df.empty:
        print("[WARN] 그래프 2: GPU 데이터 없음")
        return

    fig = plt.figure(figsize=(16, 9))
    gs  = GridSpec(2, 2, figure=fig, hspace=0.48, wspace=0.32)
    ax_all   = fig.add_subplot(gs[0, :])
    ax_gpu   = fig.add_subplot(gs[1, 0])
    ax_cross = fig.add_subplot(gs[1, 1])

    rows = []

    _scatter(ax_all, gpu_df, "gpu_util_mean", "gpu_power_mean")
    s = regression_overlay(ax_all,
                           gpu_df["gpu_util_mean"].values,
                           gpu_df["gpu_power_mean"].values)
    if s: rows.append({"구간": "전체(GPU)", **s})
    ax_all.set_xlabel("GPU Utilization (%)", fontsize=11)
    ax_all.set_ylabel("GPU 소모 전력 (W)", fontsize=11)
    ax_all.set_title("[그래프 2] GPU Utilization – GPU 소모 전력 회귀 분석 (전체)",
                     fontsize=13, fontweight="bold")
    ax_all.legend(fontsize=8, ncol=3)
    ax_all.grid(True, alpha=0.3)

    for cat, ax_c, label in [("GPU-centric", ax_gpu,   "GPU-centric"),
                              ("cross",       ax_cross, "교차 (gemm_gpu)")]:
        sub = gpu_df[gpu_df["category"] == cat]
        if sub.empty:
            ax_c.set_visible(False)
            continue
        _scatter(ax_c, sub, "gpu_util_mean", "gpu_power_mean")
        s = regression_overlay(ax_c,
                               sub["gpu_util_mean"].values,
                               sub["gpu_power_mean"].values)
        if s: rows.append({"구간": label, **s})
        ax_c.set_title(label, fontsize=10)
        ax_c.set_xlabel("GPU Util (%)", fontsize=9)
        ax_c.set_ylabel("GPU 전력 (W)", fontsize=9)
        ax_c.legend(fontsize=7)
        ax_c.grid(True, alpha=0.3)

    _print_table("[그래프 2] 회귀 지표", rows)
    _save(fig, out_dir, "graph2_gpu_util_power.png")


# ────────────────────────────────────────────────
# [그래프 3] 부하율별 CPU / GPU 전력 추이
# ────────────────────────────────────────────────
def plot_graph3(agg: pd.DataFrame, out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    specs = [
        (axes[0], "cpu_power_mean", "cpu_power_std", "CPU 소모 전력 (W)", "CPU 전력"),
        (axes[1], "gpu_power_mean", "gpu_power_std", "GPU 소모 전력 (W)", "GPU 전력"),
    ]

    for ax, ycol, ystd, ylabel, title_suffix in specs:
        for wl, grp in agg.groupby("workload"):
            g    = grp.sort_values("load_level")
            yerr = g[ystd].fillna(0).values if ystd in g.columns else None
            ax.errorbar(
                g["load_level"], g[ycol].values, yerr=yerr,
                fmt=f"{WL_MARKERS.get(wl,'o')}-",
                color=WL_COLORS.get(wl, "gray"),
                markersize=8, capsize=4, linewidth=1.8, label=wl,
            )
        ax.set_xticks([20, 50, 100])
        ax.set_xticklabels(["20%", "50%", "100%"])
        ax.set_xlabel("부하율", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f"[그래프 3] 부하율별 {title_suffix} 추이",
                     fontsize=12, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    _save(fig, out_dir, "graph3_power_vs_load.png")


# ────────────────────────────────────────────────
# [그래프 4] 전력 구성 비율 (100% 부하)
# ────────────────────────────────────────────────
def plot_graph4(agg: pd.DataFrame, out_dir: Path):
    sub = agg[agg["load_level"] == 100].copy()
    if sub.empty:
        print("[WARN] 그래프 4: 100% 부하 데이터 없음")
        return

    order_map = {c: i for i, c in enumerate(CATEGORY_ORDER)}
    sub["_ord"] = sub["category"].map(order_map)
    sub = sub.sort_values(["_ord", "workload"])
    wls = list(sub["workload"].unique())

    cpu_v = sub.groupby("workload")["cpu_power_mean"].mean().reindex(wls).fillna(0)
    gpu_v = sub.groupby("workload")["gpu_power_mean"].mean().reindex(wls).fillna(0)
    tot_v = sub.groupby("workload")["total_power_mean"].mean().reindex(wls).fillna(0)
    other = (tot_v - cpu_v - gpu_v).clip(lower=0)

    x = np.arange(len(wls))
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    ax = axes[0]
    ax.bar(x, cpu_v,  label="CPU 전력", color="#1f77b4", zorder=3)
    ax.bar(x, gpu_v,  label="GPU 전력", color="#ff7f0e",
           bottom=cpu_v.values, zorder=3)
    ax.bar(x, other,  label="기타",     color="#2ca02c",
           bottom=(cpu_v + gpu_v).values, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(wls, rotation=20, ha="right")
    ax.set_ylabel("전력 (W)"); ax.set_title("절대값 (W)", fontsize=11)
    ax.legend(); ax.grid(axis="y", alpha=0.3)

    ax2 = axes[1]
    safe    = tot_v.replace(0, np.nan)
    cpu_r   = (cpu_v  / safe * 100).fillna(0)
    gpu_r   = (gpu_v  / safe * 100).fillna(0)
    other_r = (other  / safe * 100).fillna(0)
    ax2.bar(x, cpu_r,   label="CPU",  color="#1f77b4", zorder=3)
    ax2.bar(x, gpu_r,   label="GPU",  color="#ff7f0e",
            bottom=cpu_r.values, zorder=3)
    ax2.bar(x, other_r, label="기타", color="#2ca02c",
            bottom=(cpu_r + gpu_r).values, zorder=3)
    ax2.set_xticks(x); ax2.set_xticklabels(wls, rotation=20, ha="right")
    ax2.set_ylabel("비율 (%)"); ax2.set_title("구성 비율 (%)", fontsize=11)
    ax2.set_ylim(0, 105); ax2.legend(); ax2.grid(axis="y", alpha=0.3)

    for ax_ in axes:
        prev_cat = None
        for idx, wl in enumerate(wls):
            cat = sub[sub["workload"] == wl]["category"].iloc[0]
            if prev_cat and cat != prev_cat:
                ax_.axvline(idx - 0.5, color="black", ls="--", lw=1.2, alpha=0.5)
            prev_cat = cat

    fig.suptitle("[그래프 4] 워크로드별 전력 구성 비율 (100% 부하)",
                 fontsize=13, fontweight="bold")
    _save(fig, out_dir, "graph4_power_composition.png")


# ────────────────────────────────────────────────
# [그래프 5] 교차 실험: GEMM CPU vs GPU 비교
# ────────────────────────────────────────────────
def plot_graph5(agg: pd.DataFrame, out_dir: Path):
    cross = agg[agg["category"] == "cross"].copy()
    if cross.empty:
        print("[WARN] 그래프 5: 교차 실험 데이터 없음")
        return

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    specs = [
        (axes[0], "total_power_mean", "total_power_std", "전체 소모 전력 (W)", "전체 전력"),
        (axes[1], "energy_mean",      "energy_std",      "에너지 소비 (J)",    "에너지"),
    ]

    for ax, ycol, ystd, ylabel, title in specs:
        for wl, fmt, color in [
            ("gemm_cpu", f"{WL_MARKERS['gemm_cpu']}-", WL_COLORS["gemm_cpu"]),
            ("gemm_gpu", f"{WL_MARKERS['gemm_gpu']}-", WL_COLORS["gemm_gpu"]),
        ]:
            sub = cross[cross["workload"] == wl].sort_values("load_level")
            if sub.empty:
                continue
            yerr = sub[ystd].fillna(0).values if ystd in sub.columns else None
            ax.errorbar(sub["load_level"], sub[ycol],
                        yerr=yerr, fmt=fmt, color=color,
                        markersize=10, capsize=5, linewidth=2,
                        label=wl, zorder=3)
            x_ = sub["load_level"].values.astype(float)
            y_ = sub[ycol].values
            valid = np.isfinite(x_) & np.isfinite(y_)
            if valid.sum() >= 3:
                try:
                    z  = np.polyfit(x_[valid], y_[valid], 2)
                    xx = np.linspace(x_[valid].min(), x_[valid].max(), 100)
                    ax.plot(xx, np.polyval(z, xx), "--", color=color, alpha=0.5, lw=1.5)
                except np.linalg.LinAlgError:
                    pass

        for ll in [20, 50, 100]:
            r_cpu = cross[(cross["workload"] == "gemm_cpu") & (cross["load_level"] == ll)]
            r_gpu = cross[(cross["workload"] == "gemm_gpu") & (cross["load_level"] == ll)]
            if not r_cpu.empty and not r_gpu.empty:
                vc = r_cpu[ycol].values[0]
                vg = r_gpu[ycol].values[0]
                ratio = vc / (vg + 1e-9)
                ypos  = max(vc, vg) * 1.06
                ax.annotate(f"CPU/GPU={ratio:.2f}x",
                            xy=(ll, ypos), ha="center",
                            fontsize=7.5, color="dimgray")

        ax.set_xticks([20, 50, 100])
        ax.set_xticklabels(["20%", "50%", "100%"])
        ax.set_xlabel("부하율", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle("[그래프 5] 교차 실험: GEMM CPU 실행 vs GPU 실행",
                 fontsize=13, fontweight="bold")
    _save(fig, out_dir, "graph5_cross_comparison.png")


# ────────────────────────────────────────────────
# [그래프 6] 온도 - 전력 상관관계
# ────────────────────────────────────────────────
def plot_graph6(agg: pd.DataFrame, out_dir: Path):
    load_sizes = {20: 50, 50: 110, 100: 200}

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    pairs = [
        (axes[0], "cpu_temp_mean", "cpu_power_mean",
         "CPU 온도 (°C)", "CPU 소모 전력 (W)", "CPU 온도 – CPU 전력"),
        (axes[1], "gpu_temp_mean", "gpu_power_mean",
         "GPU 온도 (°C)", "GPU 소모 전력 (W)", "GPU 온도 – GPU 전력"),
    ]

    for ax, xcol, ycol, xlabel, ylabel, title in pairs:
        seen_wl = {}
        for _, row in agg.iterrows():
            wl = row["workload"]
            ll = row["load_level"]
            xv = row[xcol]; yv = row[ycol]
            if not (np.isfinite(xv) and np.isfinite(yv)):
                continue
            sc = ax.scatter(xv, yv,
                            color=WL_COLORS.get(wl, "gray"),
                            s=load_sizes.get(ll, 80),
                            marker=WL_MARKERS.get(wl, "o"),
                            alpha=0.8, zorder=3)
            if wl not in seen_wl:
                seen_wl[wl] = sc

        xv_ = agg[xcol].values; yv_ = agg[ycol].values
        valid = np.isfinite(xv_) & np.isfinite(yv_)
        if valid.sum() > 3 and np.std(xv_[valid]) > 1e-9:
            try:
                z  = np.polyfit(xv_[valid], yv_[valid], 1)
                xx = np.linspace(xv_[valid].min(), xv_[valid].max(), 100)
                ax.plot(xx, np.polyval(z, xx), "--", color="gray",
                        lw=1.5, label="추세선", zorder=2)
            except np.linalg.LinAlgError:
                pass

        wl_handles = [
            Line2D([0],[0], marker=WL_MARKERS.get(w,"o"), color="w",
                   markerfacecolor=WL_COLORS.get(w,"gray"), markersize=9, label=w)
            for w in seen_wl
        ]
        sz_handles = [
            Line2D([0],[0], marker="o", color="w", markerfacecolor="gray",
                   markersize=np.sqrt(s) * 0.55, label=f"{ll}%")
            for ll, s in load_sizes.items()
        ]
        ax.legend(handles=wl_handles + sz_handles, fontsize=7.5,
                  ncol=2, loc="upper left")
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)

    fig.suptitle("[그래프 6] 온도 – 전력 상관관계",
                 fontsize=13, fontweight="bold")
    _save(fig, out_dir, "graph6_temp_power.png")


# ────────────────────────────────────────────────
# 모델링 파이프라인
# ────────────────────────────────────────────────
def modeling_pipeline(agg: pd.DataFrame):
    SEP = "=" * 68
    print(f"\n{SEP}\n  모델링 결과\n{SEP}")

    print("\n[Step 1] Baseline 선형 모델  P(u) = α·u + β  (u = 부하율 %)")
    for wl, g in agg.groupby("workload"):
        x = g["load_level"].values.astype(float)
        y = g["total_power_mean"].values
        if len(x) < 2: continue
        m, r2, rmse = fit_linear(x, y)
        lr = m.named_steps["linearregression"]
        print(f"  {wl:<12}: P = {lr.coef_[0]:.3f}·u + {lr.intercept_:.2f}"
              f"   R²={r2:.4f}  RMSE={rmse:.3f}W")

    print("\n[Step 2] 2차 다항 모델  P(u) = a·u² + b·u + c")
    poly_rows = []
    for wl, g in agg.groupby("workload"):
        x = g["load_level"].values.astype(float)
        y = g["total_power_mean"].values
        if len(x) < 3: continue
        m, r2, rmse = fit_poly2(x, y)
        lr  = m.named_steps["linearregression"]
        cat = g["category"].iloc[0]
        print(f"  {wl:<12} [{cat}]: R²={r2:.4f}  RMSE={rmse:.3f}W  "
              f"coef=[{lr.coef_[0]:.5f}, {lr.coef_[1]:.4f}]  b={lr.intercept_:.3f}")
        poly_rows.append({"workload": wl, "category": cat, "r2": r2, "rmse": rmse})

    print("\n[Step 3] Leave-one-workload-out Cross-Validation")
    x_all  = agg["load_level"].values.astype(float)
    y_all  = agg["total_power_mean"].values
    groups = agg["workload"].values

    logo = LeaveOneGroupOut()
    le, pe, lm_, pm_ = [], [], [], []
    for tr, te in logo.split(x_all.reshape(-1, 1), y_all, groups):
        Xtr, ytr = x_all[tr].reshape(-1, 1), y_all[tr]
        Xte, yte = x_all[te].reshape(-1, 1), y_all[te]

        lreg = LinearRegression().fit(Xtr, ytr)
        pl   = lreg.predict(Xte)
        le.append(np.sqrt(mean_squared_error(yte, pl)))
        lm_.append(mean_absolute_percentage_error(yte, pl) * 100)

        preg = make_pipeline(PolynomialFeatures(2, include_bias=False), LinearRegression())
        preg.fit(Xtr, ytr); pp = preg.predict(Xte)
        pe.append(np.sqrt(mean_squared_error(yte, pp)))
        pm_.append(mean_absolute_percentage_error(yte, pp) * 100)

    print(f"  Linear  RMSE={np.mean(le):.3f}±{np.std(le):.3f}W  MAPE={np.mean(lm_):.2f}%")
    print(f"  Poly-2  RMSE={np.mean(pe):.3f}±{np.std(pe):.3f}W  MAPE={np.mean(pm_):.2f}%")
    print(f"  → Poly-2 개선폭: ΔRMSE={np.mean(le)-np.mean(pe):.3f}W")

    print("\n[Step 4] 워크로드별 Poly-2 모델 요약")
    print(f"  {'워크로드':<12} {'분류':<14} {'R²':>8} {'RMSE(W)':>10}")
    print("  " + "-" * 46)
    for r in sorted(poly_rows, key=lambda x: x["r2"], reverse=True):
        print(f"  {r['workload']:<12} {r['category']:<14} {r['r2']:>8.4f} {r['rmse']:>10.3f}")

    def mean_r2_by_cat(cat):
        vals = [r["r2"] for r in poly_rows if r["category"] == cat]
        return np.mean(vals) if vals else float("nan")

    print(f"\n  분류별 평균 Poly-2 R²:")
    for cat in CATEGORY_ORDER:
        print(f"    {CATEGORY_LABELS[cat]:<16}: {mean_r2_by_cat(cat):.4f}")


# ────────────────────────────────────────────────
# 유틸
# ────────────────────────────────────────────────
def _print_table(title: str, rows: list):
    if not rows:
        return
    print(f"\n{title}")
    print(f"  {'구간':<20} {'Linear R²':>10} {'RMSE':>8} {'Poly2 R²':>10} {'RMSE':>8}")
    for r in rows:
        print(f"  {r['구간']:<20} {r['linear_r2']:>10.4f} {r['linear_rmse']:>8.3f}"
              f" {r['poly2_r2']:>10.4f} {r['poly2_rmse']:>8.3f}")


def _save(fig, out_dir: Path, fname: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / fname
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  저장: {path}")


# ────────────────────────────────────────────────
# 더미 데이터
# ────────────────────────────────────────────────
def generate_dummy_data() -> pd.DataFrame:
    np.random.seed(42)
    rows = []
    for wl, meta in WORKLOAD_META.items():
        cat = meta["category"]
        for load in [20, 50, 100]:
            for rep in range(1, 4):
                u        = float(load)
                base_cpu = 35 + u*0.5 + u**2*0.003 + np.random.randn()*3
                base_gpu = (
                    20 + u*1.2 + u**2*0.006 + np.random.randn()*4
                    if cat != "CPU-centric" else
                    8  + np.random.randn()*1.5
                )
                rows.append({
                    "workload"        : wl,
                    "load_level"      : load,
                    "repeat"          : rep,
                    "category"        : cat,
                    "cpu_power_mean"  : max(base_cpu, 5),
                    "cpu_power_std"   : abs(np.random.randn()*1.5),
                    "gpu_power_mean"  : max(base_gpu, 0),
                    "gpu_power_std"   : abs(np.random.randn()*2),
                    "total_power_mean": max(base_cpu+base_gpu, 10),
                    "total_power_std" : abs(np.random.randn()*2.5),
                    "cpu_busy_mean"   : min(u + np.random.randn()*4, 100),
                    "cpu_busy_std"    : abs(np.random.randn()*2),
                    "gpu_util_mean"   : (min(u*0.9+np.random.randn()*5, 100)
                                         if cat != "CPU-centric" else np.random.randn()*2),
                    "gpu_mem_mean"    : (u*0.4+np.random.randn()*3 if cat != "CPU-centric" else 0),
                    "cpu_temp_mean"   : 38 + u*0.15 + np.random.randn()*1.5,
                    "gpu_temp_mean"   : (45 + u*0.2 + np.random.randn()*2
                                         if cat != "CPU-centric" else 35+np.random.randn()),
                    "energy_mean"     : (base_cpu+base_gpu)*60,
                    "energy_std"      : abs(np.random.randn()*5),
                })
    return pd.DataFrame(rows)


# ────────────────────────────────────────────────
# main
# ────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="실험 결과 분석 및 시각화")
    parser.add_argument("--results", default="/home/optimus/siyeon/exp_v3/results")
    parser.add_argument("--out",     default="./figures")
    parser.add_argument("--dummy",   action="store_true", help="더미 데이터로 형식 검증")
    args = parser.parse_args()

    out_dir = Path(args.out)

    if args.dummy:
        print("[MODE] 더미 데이터")
        agg = generate_dummy_data()
    else:
        df_raw = load_results(Path(args.results))
        df_run = agg_per_run(df_raw)
        agg    = agg_mean_std(df_run)

    print(f"\n집계 데이터 미리보기:\n{agg.head()}\n")

    print("그래프 생성 중...")
    plot_graph1(agg, out_dir)
    plot_graph2(agg, out_dir)
    plot_graph3(agg, out_dir)
    plot_graph4(agg, out_dir)
    plot_graph5(agg, out_dir)
    plot_graph6(agg, out_dir)

    modeling_pipeline(agg)

    print(f"\n완료 → {out_dir}")