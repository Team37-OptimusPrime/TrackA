#!/usr/bin/env python3
"""
CPU-centric Workload: GROMACS Molecular Dynamics (Mixed-bound)
실험계획서 §2 - CPU-centric, Mixed-bound, 과학 HPC 대표 워크로드

부하 제어: --threads 인수로 OMP 스레드 수 조절
  (run_experiment.sh에서 load_pct에 따라 코어 수를 계산하여 전달)

실행 모드 (자동 선택):
  1. GROMACS(gmx) 설치됨 + tpr 파일 존재 → gmx mdrun 실행
  2. 위 조건 불충족             → 순수 Python LJ-MD 시뮬레이터 (fallback)

사용법:
    python gromacs.py --threads 4 --loop
    python gromacs.py --threads 10 --tpr ./gromacs_input/topol.tpr --loop
    python gromacs.py --threads 20 --force-python --loop
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


SCRIPT_DIR     = Path(__file__).parent
DEFAULT_TPR    = SCRIPT_DIR / "gromacs_input" / "topol.tpr"


# ─── GROMACS 탐색 ─────────────────────────────────────────────────────────────

def find_gmx() -> str | None:
    for name in ("gmx", "gmx_mpi", "gmx_d"):
        p = shutil.which(name)
        if p:
            return p
    return None


# ─── 모드 1: 실제 gmx mdrun ───────────────────────────────────────────────────

def run_gmx(tpr: Path, threads: int, n_steps: int, loop: bool) -> None:
    """
    gmx mdrun 을 threads 개의 OpenMP 스레드로 실행.
    run_experiment.sh 의 taskset/numactl 래핑과 함께 동작.
    """
    gmx = find_gmx()
    assert gmx, "gmx not found"

    print(f"[INFO] GROMACS mdrun | threads={threads} | tpr={tpr}")
    iteration = 0

    with tempfile.TemporaryDirectory(prefix="gmx_md_") as tmp:
        while True:
            cmd = [
                gmx, "mdrun",
                "-ntmpi", "1",
                "-ntomp",  str(threads),
                "-s",      str(tpr),
                "-nsteps", str(n_steps),
                "-noconfout",
                "-deffnm",  f"run{iteration}",
            ]
            result = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"[WARN] mdrun 오류 (iter {iteration}):\n{result.stderr[:300]}",
                      file=sys.stderr)
            else:
                print(f"[INFO] mdrun iter {iteration + 1} 완료")

            iteration += 1
            if not loop:
                break

    print(f"[INFO] GROMACS 완료 ({iteration} runs)")


# ─── 모드 2: 순수 Python LJ-MD (fallback) ────────────────────────────────────

def run_python_md(threads: int, n_atoms: int, n_steps: int, loop: bool) -> None:
    """
    GROMACS 없는 환경용 순수 numpy MD 시뮬레이터.

    연산 구성 (Mixed-bound 특성 재현):
      - Lennard-Jones 쌍별 힘 계산  O(N²/2) → Compute + Memory 혼합
      - Velocity Verlet 적분
      - PBC (최소 이미지 규약)
      - V-rescale 온도 조절
    """
    # OMP 스레드 수 → numpy/BLAS 코어 점유율 제어
    os.environ["OMP_NUM_THREADS"]      = str(threads)
    os.environ["MKL_NUM_THREADS"]      = str(threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads)

    try:
        import numpy as np
    except ImportError:
        print("[ERROR] numpy 미설치. pip install numpy", file=sys.stderr)
        sys.exit(1)

    # ── 시스템 파라미터 ──────────────────────────────────────────────────
    np.random.seed(42)
    dt      = 0.002          # 무차원 시간 스텝
    box     = 2.0            # nm (PBC 박스 크기)
    rc      = 1.0            # LJ 차단 반경 nm
    mass    = 1.0
    eps     = 0.65           # LJ ε
    sig     = 0.316          # LJ σ nm
    kT      = 2.494          # 300 K × k_B (무차원)

    pos = np.random.uniform(0, box, (n_atoms, 3))
    vel = np.random.normal(0, (kT / mass) ** 0.5, (n_atoms, 3))

    # ── 내부 함수 ────────────────────────────────────────────────────────

    def lj_forces(pos):
        """Lennard-Jones 힘 계산 (벡터화, O(N²/2))"""
        N = len(pos)
        F = np.zeros_like(pos)
        pe = 0.0
        ii, jj = np.triu_indices(N, k=1)

        rij = pos[ii] - pos[jj]
        rij -= box * np.round(rij / box)          # PBC 최소 이미지
        r2  = np.einsum("ij,ij->i", rij, rij)

        mask  = r2 < rc ** 2
        r2m   = np.maximum(r2[mask], 1e-8)        # 겹침 방지
        rijm  = rij[mask]
        im, jm = ii[mask], jj[mask]

        sr2   = (sig ** 2) / r2m
        sr6   = sr2 ** 3
        sr12  = sr6 ** 2
        pe   += np.sum(4 * eps * (sr12 - sr6))

        fmag  = np.nan_to_num((24 * eps / r2m) * (2 * sr12 - sr6))
        fvec  = fmag[:, None] * rijm
        np.add.at(F, im,  fvec)
        np.add.at(F, jm, -fvec)
        return F, pe

    def v_rescale(vel, kT):
        """간단한 속도 스케일링 온도 조절"""
        ke = 0.5 * mass * np.sum(vel ** 2)
        dof = 3 * len(vel) - 3
        T = 2 * ke / dof if dof > 0 else 1.0
        if T > 0:
            vel *= (kT / (mass * T)) ** 0.5
        return vel

    # ── 메인 MD 루프 ─────────────────────────────────────────────────────
    import time

    print(f"[INFO] Python LJ-MD 시작 | atoms={n_atoms} | threads={threads} | steps/iter={n_steps}")

    F, _ = lj_forces(pos)
    iteration = 0

    while True:
        t0 = time.perf_counter()
        pe_acc = ke_acc = 0.0

        for step in range(n_steps):
            # Velocity Verlet
            vel_h = vel + 0.5 * dt * F / mass
            pos   = (pos + dt * vel_h) % box
            F, pe = lj_forces(pos)
            vel   = vel_h + 0.5 * dt * F / mass
            pe_acc += pe

            if step % 100 == 0:
                vel    = v_rescale(vel, kT)
                ke_acc += 0.5 * mass * np.sum(vel ** 2)

        elapsed = time.perf_counter() - t0
        T_eff   = ke_acc / (n_steps / 100) / (1.5 * n_atoms)
        print(f"  [iter {iteration + 1}] steps={n_steps} | "
              f"elapsed={elapsed:.2f}s | T_eff={T_eff:.1f} | "
              f"<PE>={pe_acc / n_steps:.1f}")

        iteration += 1
        if not loop:
            break

    print(f"[INFO] Python MD 완료 ({iteration} iter × {n_steps} steps)")


# ─── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="GROMACS MD 워크로드 (스레드 수로 부하 제어)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--threads",      type=int, default=8,
                        help="OMP 스레드 수 (부하 제어)")
    parser.add_argument("--tpr",          type=str, default=str(DEFAULT_TPR),
                        help=".tpr 파일 경로 (gmx mdrun용)")
    parser.add_argument("--n-steps",      type=int, default=500000,
                        help="MD 스텝 수")
    parser.add_argument("--n-atoms",      type=int, default=648,
                        help="Python fallback 원자 수 (216 water × 3)")
    parser.add_argument("--loop",         action="store_true",
                        help="종료 신호까지 반복")
    parser.add_argument("--force-python", action="store_true",
                        help="Python fallback 강제 사용")
    args = parser.parse_args()

    tpr = Path(args.tpr)
    use_gmx = (not args.force_python) and (find_gmx() is not None) and tpr.exists()

    if use_gmx:
        run_gmx(tpr, args.threads, args.n_steps, args.loop)
    else:
        if not args.force_python:
            reason = "gmx 없음" if find_gmx() is None else f"tpr 없음 ({tpr})"
            print(f"[INFO] Python fallback 사용 ({reason})")
        run_python_md(args.threads, args.n_atoms, args.n_steps // 50, args.loop)


if __name__ == "__main__":
    main()
