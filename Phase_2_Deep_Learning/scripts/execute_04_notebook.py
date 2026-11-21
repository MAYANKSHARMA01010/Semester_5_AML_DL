import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
from pathlib import Path
import nbformat
from nbclient import NotebookClient

NOTEBOOK_PATH = Path("Phase_2_Deep_Learning/notebooks/04_hybrid_model.ipynb")

def main():
    if not NOTEBOOK_PATH.exists():
        raise FileNotFoundError(f"Notebook not found: {NOTEBOOK_PATH}")
    
    print(f"Loading notebook: {NOTEBOOK_PATH}")
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    client = NotebookClient(nb, timeout=1800, kernel_name="python3")
    print("Executing notebook top-to-bottom...")
    client.execute()

    print(f"Saving executed notebook: {NOTEBOOK_PATH}")
    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    print("✓ Notebook executed and updated successfully.")

if __name__ == "__main__":
    main()
