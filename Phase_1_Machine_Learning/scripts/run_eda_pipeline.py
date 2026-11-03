#!/usr/bin/env python3
"""Runs all EDA notebooks sequentially in the local virtual environment."""

import os
import sys
import time
import glob
import subprocess

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NB_DIR = os.path.join(PROJECT_ROOT, "notebooks", "01.EDA")
OUT_DIR = os.path.join(PROJECT_ROOT, "data", "GeneratedOutput", "01.EDA")
REQ_FILE = os.path.join(PROJECT_ROOT, "requirements.txt")
SETUP_SCRIPT = os.path.join(PROJECT_ROOT, "scripts", "setup_environment.sh")


def ensure_venv_python():
    """Ensures execution occurs within the project virtual environment."""
    in_venv = (sys.prefix != sys.base_prefix) and os.path.exists(os.path.join(sys.prefix, "pyvenv.cfg"))
    if os.name == "nt":
        venv_python = os.path.join(PROJECT_ROOT, ".venv", "Scripts", "python.exe")
    else:
        venv_python = os.path.join(PROJECT_ROOT, ".venv", "bin", "python")

    if not in_venv:
        if os.path.exists(SETUP_SCRIPT) and os.name != "nt":
            subprocess.run(["bash", SETUP_SCRIPT], check=True)
        elif not os.path.exists(venv_python):
            subprocess.run([sys.executable, "-m", "venv", os.path.join(PROJECT_ROOT, ".venv")], check=True)
            subprocess.run([venv_python, "-m", "pip", "install", "--upgrade", "pip", "--quiet"], check=True)
            subprocess.run([venv_python, "-m", "pip", "install", "-r", REQ_FILE, "--quiet"], check=True)

        if os.path.exists(venv_python):
            os.execv(venv_python, [venv_python] + sys.argv)
        else:
            sys.exit("Error: Could not locate .venv python")


ensure_venv_python()

import nbformat
from nbclient import NotebookClient

try:
    import pyarrow.parquet as pq
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False


def execute_notebook(nb_path: str, index: int, total: int):
    nb_name = os.path.basename(nb_path)
    print(f"[{index}/{total}] Running: {nb_name} ...")
    t0 = time.time()

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    client = NotebookClient(
        nb,
        timeout=1200,
        kernel_name="python3",
        resources={"metadata": {"path": PROJECT_ROOT}},
    )

    try:
        client.execute()
        elapsed = time.time() - t0
        with open(nb_path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)
        print(f"  Passed in {elapsed:.1f}s")
        return True, elapsed, None
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  Failed in {elapsed:.1f}s - {e}")
        return False, elapsed, str(e)


def verify_parquet_outputs():
    out_files = sorted(glob.glob(os.path.join(OUT_DIR, "*.parquet")))
    if not out_files:
        print("Warning: No parquet files found in output directory.")
        return False

    print(f"\n{'Filename':45} | {'Rows':>9} | {'Cols':>6} | {'Size':>9}")
    print("-" * 75)
    for p in out_files:
        name = os.path.basename(p)
        size_kb = os.path.getsize(p) / 1024
        rows_str, cols_str = "N/A", "N/A"
        if HAS_PYARROW:
            try:
                table = pq.read_table(p)
                rows_str = f"{table.num_rows:,}"
                cols_str = str(table.num_columns)
            except Exception:
                pass
        print(f"{name:45} | {rows_str:>9} | {cols_str:>6} | {size_kb:>8.1f} KB")
    print("-" * 75)
    return True


def main():
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Notebooks dir: {NB_DIR}")

    nb_files = sorted(glob.glob(os.path.join(NB_DIR, "0[1-6]_*.ipynb")))
    if not nb_files:
        nb_files = sorted(glob.glob(os.path.join(NB_DIR, "*.ipynb")))

    if not nb_files:
        sys.exit(f"No notebooks found in {NB_DIR}")

    print(f"Found {len(nb_files)} notebooks to run:")
    for i, f in enumerate(nb_files, 1):
        print(f"  {i}. {os.path.basename(f)}")

    results = []
    t_start = time.time()
    for i, nb_path in enumerate(nb_files, 1):
        success, duration, error = execute_notebook(nb_path, i, len(nb_files))
        results.append({
            "notebook": os.path.basename(nb_path),
            "status": "PASSED" if success else "FAILED",
            "time": f"{duration:.1f}s",
            "error": error,
        })

    verify_parquet_outputs()

    print(f"\nCompleted in {time.time() - t_start:.1f}s")
    for r in results:
        mark = "✓" if r["status"] == "PASSED" else "✗"
        print(f" {mark} {r['notebook']:<45} {r['status']:<8} {r['time']}")


if __name__ == "__main__":
    main()
