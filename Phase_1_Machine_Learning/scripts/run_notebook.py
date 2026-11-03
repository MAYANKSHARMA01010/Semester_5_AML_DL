import sys
import os
import time
import nbformat
from nbclient import NotebookClient

def run_nb(nb_path):
    print(f"Executing: {nb_path} ...")
    t0 = time.time()
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    client = NotebookClient(
        nb,
        timeout=1800,
        kernel_name="python3",
        resources={"metadata": {"path": os.path.dirname(os.path.abspath(nb_path))}}
    )
    client.execute()

    with open(nb_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.2f}s: {nb_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_notebook.py <path_to_notebook>")
        sys.exit(1)
    run_nb(sys.argv[1])
