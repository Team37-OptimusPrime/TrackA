#!/usr/bin/env python3
"""
GROMACS 입력 파일 준비 도구
실험계획서에서 요구하는 topol.tpr을 자동으로 생성합니다.

세 가지 시스템 중 선택:
  1. water      : SPC/E Water box (기본, 빠른 준비)
  2. alanine    : Alanine dipeptide in water (소형, 검증용)
  3. protein    : Villin headpiece (HP36) in water (중형, 실제 HPC 시나리오)

사용법:
    python prepare_gromacs_input.py --system water --n-molecules 1000
    python prepare_gromacs_input.py --system alanine
    python prepare_gromacs_input.py --system protein --pdb-id 1VII
    python prepare_gromacs_input.py --check   # 환경 점검만
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

SCRIPT_DIR  = Path(__file__).parent
OUT_DIR     = SCRIPT_DIR / "gromacs_input"


# ─── 유틸 ──────────────────────────────────────────────────────────────────────

def run_cmd(cmd: list[str], cwd: Path, label: str) -> subprocess.CompletedProcess:
    print(f"  [CMD] {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [WARN] {label} 실패:\n{result.stderr[:600]}", file=sys.stderr)
    else:
        print(f"  [OK]  {label}")
    return result


def find_gmx() -> str | None:
    for name in ("gmx", "gmx_mpi", "gmx_d"):
        p = shutil.which(name)
        if p:
            return p
    return None


def check_environment():
    print("=" * 55)
    print("GROMACS 환경 점검")
    print("=" * 55)

    gmx = find_gmx()
    if gmx:
        result = subprocess.run([gmx, "--version"], capture_output=True, text=True)
        ver_line = next((l for l in result.stdout.splitlines() if "GROMACS version" in l), "")
        print(f"  ✓ GROMACS 발견: {gmx}")
        print(f"    {ver_line.strip()}")
    else:
        print("  ✗ GROMACS 미발견")
        print("    Ubuntu:  sudo apt install gromacs")
        print("    Conda:   conda install -c bioconda gromacs")

    # Force field 확인
    for ff_name in ("oplsaa", "amber99sb-ildn", "charmm36"):
        ff_path = Path(f"/usr/share/gromacs/top/{ff_name}.ff")
        if ff_path.exists():
            print(f"  ✓ Force field: {ff_name}")
        else:
            ff_path2 = Path(f"/usr/local/share/gromacs/top/{ff_name}.ff")
            if ff_path2.exists():
                print(f"  ✓ Force field: {ff_name} (local)")
            else:
                print(f"  - Force field: {ff_name} (없음)")

    # Python 의존성
    for pkg in ("numpy", "scipy"):
        try:
            __import__(pkg)
            print(f"  ✓ Python: {pkg}")
        except ImportError:
            print(f"  ✗ Python: {pkg} 미설치")

    print()


# ─── MDP 파일 생성 ────────────────────────────────────────────────────────────

def write_em_mdp(path: Path):
    """에너지 최소화 MDP"""
    path.write_text(textwrap.dedent("""\
        ; Energy minimization
        integrator  = steep
        emtol       = 1000.0
        emstep      = 0.01
        nsteps      = 50000

        ; Neighbor searching
        cutoff-scheme   = Verlet
        ns_type         = grid
        nstlist         = 1
        rcoulomb        = 1.0
        rvdw            = 1.0

        ; Electrostatics
        coulombtype     = PME
        pme_order       = 4
        fourierspacing  = 0.16

        ; Constraints
        constraints     = none
    """))


def write_nvt_mdp(path: Path, nsteps: int = 50000):
    """NVT 평형화 MDP"""
    path.write_text(textwrap.dedent(f"""\
        ; NVT Equilibration
        define          = -DPOSRES
        integrator      = md
        nsteps          = {nsteps}
        dt              = 0.002

        nstxout         = 0
        nstvout         = 0
        nstfout         = 0
        nstlog          = 500
        nstenergy       = 500

        ; Neighbor searching
        cutoff-scheme   = Verlet
        ns_type         = grid
        nstlist         = 10
        rcoulomb        = 1.0
        rvdw            = 1.0

        ; Electrostatics
        coulombtype     = PME
        pme_order       = 4
        fourierspacing  = 0.16

        ; Temperature coupling
        tcoupl          = V-rescale
        tc-grps         = System
        tau_t           = 0.1
        ref_t           = 300

        ; Pressure coupling off (NVT)
        pcoupl          = no

        ; Initial velocities
        gen_vel         = yes
        gen_temp        = 300
        gen_seed        = 42

        ; Constraints
        constraints     = h-bonds
        constraint_algorithm = LINCS
    """))


def write_npt_mdp(path: Path, nsteps: int = 50000):
    """NPT 평형화 MDP"""
    path.write_text(textwrap.dedent(f"""\
        ; NPT Equilibration
        define          = -DPOSRES
        integrator      = md
        nsteps          = {nsteps}
        dt              = 0.002

        nstxout         = 0
        nstvout         = 0
        nstfout         = 0
        nstlog          = 500
        nstenergy       = 500

        ; Neighbor searching
        cutoff-scheme   = Verlet
        ns_type         = grid
        nstlist         = 10
        rcoulomb        = 1.0
        rvdw            = 1.0

        ; Electrostatics
        coulombtype     = PME
        pme_order       = 4
        fourierspacing  = 0.16

        ; Temperature coupling
        tcoupl          = V-rescale
        tc-grps         = System
        tau_t           = 0.1
        ref_t           = 300

        ; Pressure coupling
        pcoupl          = Parrinello-Rahman
        pcoupltype      = isotropic
        tau_p           = 2.0
        ref_p           = 1.0
        compressibility = 4.5e-5

        ; Constraints
        constraints     = h-bonds
        constraint_algorithm = LINCS
        continuation    = yes
        gen_vel         = no
    """))


def write_md_mdp(path: Path, nsteps: int = 500000):
    """본 실험용 MD MDP (전력 측정 시 사용)"""
    path.write_text(textwrap.dedent(f"""\
        ; Production MD for power benchmark
        integrator      = md
        nsteps          = {nsteps}
        dt              = 0.002

        nstxout         = 0
        nstvout         = 0
        nstfout         = 0
        nstlog          = 1000
        nstenergy       = 1000
        nstxout-compressed = 5000

        ; Neighbor searching
        cutoff-scheme   = Verlet
        ns_type         = grid
        nstlist         = 10
        rcoulomb        = 1.0
        rvdw            = 1.0

        ; Electrostatics
        coulombtype     = PME
        pme_order       = 4
        fourierspacing  = 0.16

        ; Temperature coupling
        tcoupl          = V-rescale
        tc-grps         = System
        tau_t           = 0.1
        ref_t           = 300

        ; Pressure coupling
        pcoupl          = Parrinello-Rahman
        pcoupltype      = isotropic
        tau_p           = 2.0
        ref_p           = 1.0
        compressibility = 4.5e-5

        ; Constraints
        constraints     = h-bonds
        constraint_algorithm = LINCS
        continuation    = yes
        gen_vel         = no
    """))


# ─── 시스템 1: SPC/E Water Box ────────────────────────────────────────────────

def prepare_water_system(out_dir: Path, n_molecules: int, gmx: str) -> bool:
    """
    SPC/E Water box 준비
    gmx solvate → gmx grompp (EM) → gmx mdrun (EM) → gmx grompp (NVT) → topol.tpr
    """
    print(f"\n[STEP] Water box 준비 ({n_molecules} molecules)...")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── topology ─────────────────────────────────────────────────────────
    top_path = out_dir / "topol.top"
    top_path.write_text(textwrap.dedent(f"""\
        ; SPC/E Water topology
        #include "oplsaa.ff/forcefield.itp"
        #include "oplsaa.ff/spce.itp"

        [ system ]
        SPC/E Water Benchmark

        [ molecules ]
        SOL     {n_molecules}
    """))

    # ── 빈 박스 생성 ─────────────────────────────────────────────────────
    import math
    # 물 분자당 약 0.030 nm³ → 박스 크기 계산
    vol_nm3  = n_molecules * 0.0306
    box_nm   = round(vol_nm3 ** (1/3) + 0.2, 2)
    empty_gro = out_dir / "empty.gro"
    empty_gro.write_text(textwrap.dedent(f"""\
        Empty box
         0
           {box_nm:.5f}   {box_nm:.5f}   {box_nm:.5f}
    """))

    # ── 용매 채우기 ───────────────────────────────────────────────────────
    solvated_gro = out_dir / "solvated.gro"
    r = run_cmd([gmx, "solvate",
                 "-cs", "spc216.gro",
                 "-o", str(solvated_gro),
                 "-p", str(top_path),
                 "-box", str(box_nm), str(box_nm), str(box_nm)],
                cwd=out_dir, label="solvate")
    if not solvated_gro.exists():
        return False

    # ── EM grompp ─────────────────────────────────────────────────────────
    em_mdp = out_dir / "em.mdp"
    write_em_mdp(em_mdp)
    em_tpr = out_dir / "em.tpr"
    r = run_cmd([gmx, "grompp",
                 "-f", str(em_mdp),
                 "-c", str(solvated_gro),
                 "-p", str(top_path),
                 "-o", str(em_tpr),
                 "-maxwarn", "2"],
                cwd=out_dir, label="grompp EM")
    if not em_tpr.exists():
        return False

    # ── EM mdrun ──────────────────────────────────────────────────────────
    run_cmd([gmx, "mdrun", "-v",
             "-s", str(em_tpr),
             "-deffnm", "em",
             "-ntmpi", "1", "-ntomp", "4"],
            cwd=out_dir, label="mdrun EM")
    em_gro = out_dir / "em.gro"
    if not em_gro.exists():
        print("[WARN] EM 실패. solvated.gro로 진행합니다.", file=sys.stderr)
        em_gro = solvated_gro

    # ── 본 실험 tpr 생성 (NVT MD) ─────────────────────────────────────────
    md_mdp = out_dir / "md.mdp"
    write_nvt_mdp(md_mdp, nsteps=500000)
    tpr_path = out_dir / "topol.tpr"
    run_cmd([gmx, "grompp",
             "-f", str(md_mdp),
             "-c", str(em_gro),
             "-p", str(top_path),
             "-o", str(tpr_path),
             "-maxwarn", "2"],
            cwd=out_dir, label="grompp MD production")

    if tpr_path.exists():
        print(f"\n[OK] topol.tpr 생성 완료: {tpr_path}")
        return True
    return False


# ─── 시스템 2: Alanine Dipeptide ─────────────────────────────────────────────

def prepare_alanine_system(out_dir: Path, gmx: str) -> bool:
    """
    Alanine dipeptide (Ace-Ala-NMe) in water
    pdb2gmx → editconf → solvate → grompp/mdrun EM → NVT tpr
    """
    print("\n[STEP] Alanine dipeptide 시스템 준비...")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── ala_dipeptide.pdb (간소화) ─────────────────────────────────────────
    pdb_path = out_dir / "ala_dipeptide.pdb"
    pdb_path.write_text(textwrap.dedent("""\
        ATOM      1  CH3 ACE A   1       1.522  -0.000   0.000  1.00  0.00           C
        ATOM      2  C   ACE A   1       0.000   0.000   0.000  1.00  0.00           C
        ATOM      3  O   ACE A   1      -0.667   1.066   0.000  1.00  0.00           O
        ATOM      4  N   ALA A   2      -0.667  -1.066   0.000  1.00  0.00           N
        ATOM      5  HN  ALA A   2      -0.155  -1.942   0.000  1.00  0.00           H
        ATOM      6  CA  ALA A   2      -2.133  -1.066   0.000  1.00  0.00           C
        ATOM      7  HA  ALA A   2      -2.516  -0.053   0.000  1.00  0.00           H
        ATOM      8  CB  ALA A   2      -2.667  -1.799  -1.232  1.00  0.00           C
        ATOM      9  C   ALA A   2      -2.667  -1.799   1.232  1.00  0.00           C
        ATOM     10  O   ALA A   2      -2.000  -2.865   1.232  1.00  0.00           O
        ATOM     11  N   NME A   3      -3.800  -1.306   1.723  1.00  0.00           N
        ATOM     12  CH3 NME A   3      -4.444  -1.950   2.866  1.00  0.00           C
        END
    """))

    # pdb2gmx
    gro_path = out_dir / "ala.gro"
    top_path = out_dir / "topol.top"
    r = run_cmd([gmx, "pdb2gmx",
                 "-f", str(pdb_path),
                 "-o", str(gro_path),
                 "-p", str(top_path),
                 "-ff", "oplsaa",
                 "-water", "spce",
                 "-ignh"],
                cwd=out_dir, label="pdb2gmx")

    if not gro_path.exists():
        print("[WARN] pdb2gmx 실패. water fallback으로 진행.", file=sys.stderr)
        return prepare_water_system(out_dir, 500, gmx)

    # editconf (박스 설정)
    box_gro = out_dir / "box.gro"
    run_cmd([gmx, "editconf",
             "-f", str(gro_path),
             "-o", str(box_gro),
             "-c", "-d", "1.2", "-bt", "cubic"],
            cwd=out_dir, label="editconf")

    # solvate
    solv_gro = out_dir / "solv.gro"
    run_cmd([gmx, "solvate",
             "-cp", str(box_gro),
             "-cs", "spc216.gro",
             "-o", str(solv_gro),
             "-p", str(top_path)],
            cwd=out_dir, label="solvate")

    if not solv_gro.exists():
        solv_gro = box_gro

    # EM → MD tpr
    em_mdp  = out_dir / "em.mdp";  write_em_mdp(em_mdp)
    em_tpr  = out_dir / "em.tpr"
    run_cmd([gmx, "grompp",
             "-f", str(em_mdp), "-c", str(solv_gro),
             "-p", str(top_path), "-o", str(em_tpr), "-maxwarn", "3"],
            cwd=out_dir, label="grompp EM")
    if em_tpr.exists():
        run_cmd([gmx, "mdrun", "-s", str(em_tpr), "-deffnm", "em",
                 "-ntmpi", "1", "-ntomp", "4"],
                cwd=out_dir, label="mdrun EM")

    em_gro = out_dir / "em.gro"
    start_gro = em_gro if em_gro.exists() else solv_gro

    md_mdp  = out_dir / "md.mdp";  write_nvt_mdp(md_mdp, nsteps=500000)
    tpr_path = out_dir / "topol.tpr"
    run_cmd([gmx, "grompp",
             "-f", str(md_mdp), "-c", str(start_gro),
             "-p", str(top_path), "-o", str(tpr_path), "-maxwarn", "3"],
            cwd=out_dir, label="grompp MD production")

    if tpr_path.exists():
        print(f"\n[OK] Alanine dipeptide tpr 완료: {tpr_path}")
        return True
    return False


# ─── 시스템 3: Villin Headpiece (HP36) ───────────────────────────────────────

def prepare_protein_system(out_dir: Path, gmx: str, pdb_id: str = "1VII") -> bool:
    """
    HP36 단백질 in water (실제 HPC 시나리오)
    PDB 다운로드 → pdb2gmx → solvate → ions → EM → NVT → NPT → MD tpr
    """
    print(f"\n[STEP] Protein ({pdb_id}) 시스템 준비...")
    out_dir.mkdir(parents=True, exist_ok=True)

    # PDB 다운로드
    pdb_path = out_dir / f"{pdb_id}.pdb"
    if not pdb_path.exists():
        try:
            import urllib.request
            url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
            print(f"  [DOWNLOAD] {url}")
            urllib.request.urlretrieve(url, str(pdb_path))
            print(f"  [OK] PDB 다운로드 완료")
        except Exception as e:
            print(f"  [WARN] PDB 다운로드 실패: {e}. Water fallback 사용.", file=sys.stderr)
            return prepare_water_system(out_dir, 2000, gmx)

    # HETATM 제거
    clean_pdb = out_dir / "clean.pdb"
    with open(pdb_path) as f:
        lines = [l for l in f if l.startswith(("ATOM", "TER", "END"))]
    clean_pdb.write_text("".join(lines))

    # pdb2gmx
    gro_path = out_dir / "protein.gro"
    top_path = out_dir / "topol.top"
    run_cmd([gmx, "pdb2gmx",
             "-f", str(clean_pdb),
             "-o", str(gro_path),
             "-p", str(top_path),
             "-ff", "amber99sb-ildn",
             "-water", "tip3p",
             "-ignh"],
            cwd=out_dir, label="pdb2gmx")

    if not gro_path.exists():
        print("[WARN] pdb2gmx 실패. Water fallback.", file=sys.stderr)
        return prepare_water_system(out_dir, 2000, gmx)

    # editconf
    box_gro = out_dir / "box.gro"
    run_cmd([gmx, "editconf", "-f", str(gro_path), "-o", str(box_gro),
             "-c", "-d", "1.2", "-bt", "dodecahedron"],
            cwd=out_dir, label="editconf")

    # solvate
    solv_gro = out_dir / "solv.gro"
    run_cmd([gmx, "solvate",
             "-cp", str(box_gro), "-cs", "tip3p.gro",
             "-o", str(solv_gro), "-p", str(top_path)],
            cwd=out_dir, label="solvate")

    # ions
    ions_mdp = out_dir / "ions.mdp"
    ions_mdp.write_text("integrator=steep\nnsteps=0\n")
    ions_tpr = out_dir / "ions.tpr"
    run_cmd([gmx, "grompp", "-f", str(ions_mdp),
             "-c", str(solv_gro if solv_gro.exists() else box_gro),
             "-p", str(top_path), "-o", str(ions_tpr), "-maxwarn", "2"],
            cwd=out_dir, label="grompp ions")

    if ions_tpr.exists():
        ions_gro = out_dir / "ions.gro"
        p = subprocess.Popen(
            [gmx, "genion", "-s", str(ions_tpr),
             "-o", str(ions_gro), "-p", str(top_path),
             "-pname", "NA", "-nname", "CL", "-neutral"],
            stdin=subprocess.PIPE, capture_output=True, text=True, cwd=str(out_dir)
        )
        p.communicate(input="SOL\n")

    start_struct = out_dir / "ions.gro"
    if not start_struct.exists():
        start_struct = solv_gro if solv_gro.exists() else box_gro

    # EM
    em_mdp = out_dir / "em.mdp"; write_em_mdp(em_mdp)
    em_tpr = out_dir / "em.tpr"
    run_cmd([gmx, "grompp", "-f", str(em_mdp), "-c", str(start_struct),
             "-p", str(top_path), "-o", str(em_tpr), "-maxwarn", "3"],
            cwd=out_dir, label="grompp EM")
    if em_tpr.exists():
        run_cmd([gmx, "mdrun", "-s", str(em_tpr), "-deffnm", "em",
                 "-ntmpi", "1", "-ntomp", "4"],
                cwd=out_dir, label="mdrun EM")

    em_gro = out_dir / "em.gro"

    # NVT
    nvt_mdp = out_dir / "nvt.mdp"; write_nvt_mdp(nvt_mdp, nsteps=50000)
    nvt_tpr = out_dir / "nvt.tpr"
    run_cmd([gmx, "grompp", "-f", str(nvt_mdp),
             "-c", str(em_gro if em_gro.exists() else start_struct),
             "-p", str(top_path), "-o", str(nvt_tpr), "-maxwarn", "3"],
            cwd=out_dir, label="grompp NVT")
    if nvt_tpr.exists():
        run_cmd([gmx, "mdrun", "-s", str(nvt_tpr), "-deffnm", "nvt",
                 "-ntmpi", "1", "-ntomp", "4"],
                cwd=out_dir, label="mdrun NVT")

    nvt_gro = out_dir / "nvt.gro"

    # NPT
    npt_mdp = out_dir / "npt.mdp"; write_npt_mdp(npt_mdp, nsteps=50000)
    npt_tpr = out_dir / "npt.tpr"
    run_cmd([gmx, "grompp", "-f", str(npt_mdp),
             "-c", str(nvt_gro if nvt_gro.exists() else em_gro if em_gro.exists() else start_struct),
             "-t", str(out_dir / "nvt.cpt") if (out_dir / "nvt.cpt").exists() else str(out_dir / "em.gro"),
             "-p", str(top_path), "-o", str(npt_tpr), "-maxwarn", "3"],
            cwd=out_dir, label="grompp NPT")
    if npt_tpr.exists():
        run_cmd([gmx, "mdrun", "-s", str(npt_tpr), "-deffnm", "npt",
                 "-ntmpi", "1", "-ntomp", "4"],
                cwd=out_dir, label="mdrun NPT")

    npt_gro = out_dir / "npt.gro"

    # 최종 MD tpr
    md_mdp = out_dir / "md.mdp"; write_md_mdp(md_mdp, nsteps=500000)
    tpr_path = out_dir / "topol.tpr"
    best_start = next((p for p in [npt_gro, nvt_gro, em_gro] if p.exists()), start_struct)
    best_cpt   = next((p for p in [out_dir / "npt.cpt", out_dir / "nvt.cpt"] if p.exists()), None)

    cmd = [gmx, "grompp", "-f", str(md_mdp),
           "-c", str(best_start),
           "-p", str(top_path),
           "-o", str(tpr_path),
           "-maxwarn", "3"]
    if best_cpt:
        cmd += ["-t", str(best_cpt)]
    run_cmd(cmd, cwd=out_dir, label="grompp MD production")

    if tpr_path.exists():
        print(f"\n[OK] Protein ({pdb_id}) tpr 완료: {tpr_path}")
        return True
    return False


# ─── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="GROMACS 실험 입력 파일 준비",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--system", choices=["water", "alanine", "protein"],
                        default="water", help="시뮬레이션 시스템 종류")
    parser.add_argument("--n-molecules", type=int, default=1000,
                        help="water 시스템: 물 분자 수")
    parser.add_argument("--pdb-id", default="1VII",
                        help="protein 시스템: RCSB PDB ID")
    parser.add_argument("--out-dir", default=str(OUT_DIR),
                        help="출력 디렉토리")
    parser.add_argument("--check", action="store_true",
                        help="환경 점검만 수행하고 종료")
    args = parser.parse_args()

    check_environment()
    if args.check:
        return

    gmx = find_gmx()
    if not gmx:
        print("[ERROR] GROMACS를 찾을 수 없습니다.", file=sys.stderr)
        print("        설치 후 재실행하거나, gromacs_runner.py --use-python-fallback 사용", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out_dir)

    success = False
    if args.system == "water":
        success = prepare_water_system(out_dir, args.n_molecules, gmx)
    elif args.system == "alanine":
        success = prepare_alanine_system(out_dir, gmx)
    elif args.system == "protein":
        success = prepare_protein_system(out_dir, gmx, args.pdb_id)

    if success:
        print("\n" + "=" * 55)
        print("입력 파일 준비 완료!")
        print(f"tpr 경로: {out_dir / 'topol.tpr'}")
        print("\n다음 명령으로 실험 실행:")
        print(f"  python gromacs_runner.py --load 100 --total-cores 20 \\")
        print(f"    --tpr {out_dir / 'topol.tpr'} --loop")
        print("=" * 55)
    else:
        print("\n[WARN] tpr 생성 실패. Python fallback 사용:")
        print("  python gromacs_runner.py --load 100 --total-cores 20 --use-python-fallback")


if __name__ == "__main__":
    main()
