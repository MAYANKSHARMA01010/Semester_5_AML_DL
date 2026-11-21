import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"
os.makedirs("/tmp/matplotlib_cache", exist_ok=True)

import sys
import io
import json
import base64
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NOTEBOOK_PATH = Path("Phase_2_Deep_Learning/notebooks/04_hybrid_model.ipynb")

def main():
    if not NOTEBOOK_PATH.exists():
        raise FileNotFoundError(f"Notebook not found: {NOTEBOOK_PATH}")
    
    print(f"Loading notebook: {NOTEBOOK_PATH}")
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)

    ns = {"__name__": "__main__"}
    exec_count = 1

    for idx, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            code = "".join(cell["source"])
            print(f"Executing code cell {exec_count} (index {idx})...")
            stdout_buf = io.StringIO()
            old_stdout = sys.stdout
            sys.stdout = stdout_buf
            plt.close("all")
            try:
                exec(code, ns)
            except Exception as e:
                sys.stdout = old_stdout
                print(f"Error executing cell {exec_count}: {e}")
                raise e
            finally:
                sys.stdout = old_stdout

            text_output = stdout_buf.getvalue()
            outputs = []
            if text_output:
                outputs.append({
                    "name": "stdout",
                    "output_type": "stream",
                    "text": text_output.splitlines(keepends=True)
                })

            for fig_num in plt.get_fignums():
                fig = plt.figure(fig_num)
                img_buf = io.BytesIO()
                fig.savefig(img_buf, format="png", bbox_inches="tight", dpi=100)
                img_buf.seek(0)
                img_b64 = base64.b64encode(img_buf.read()).decode("utf-8")
                outputs.append({
                    "data": {
                        "image/png": img_b64,
                        "text/plain": "<Figure size 1400x500 with 2 Axes>"
                    },
                    "metadata": {},
                    "output_type": "display_data"
                })
                plt.close(fig)

            cell["outputs"] = outputs
            cell["execution_count"] = exec_count
            exec_count += 1

    print(f"Saving executed notebook to {NOTEBOOK_PATH}...")
    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)

    print(f"✓ All {exec_count - 1} code cells executed and saved successfully.")

if __name__ == "__main__":
    main()
