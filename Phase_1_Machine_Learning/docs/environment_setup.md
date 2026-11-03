# Environment Setup & Virtual Environment Guide

This guide explains how to set up a clean Python virtual environment (`.venv`) to run all data processing, EDA notebooks, machine learning models, and deep learning pipelines in this project.

---

## 1. Prerequisites

- **Python 3.10+** (Python 3.10, 3.11, or 3.12 recommended)
- Terminal access (macOS/Linux bash/zsh, or Windows PowerShell)

---

## 2. Quick Setup (One-Liner)

Open your terminal in the project root directory (`Semester_5_AML_DL`) and run:

### macOS / Linux:
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Windows (PowerShell):
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. Step-by-Step Breakdown

### Step 1: Create the Virtual Environment
Create an isolated environment named `.venv` in the project root:
```bash
python3 -m venv .venv
```
*(This folder is already ignored by `.gitignore` so your installed libraries will not be committed to GitHub).*

### Step 2: Activate the Virtual Environment
- **macOS / Linux**:
  ```bash
  source .venv/bin/activate
  ```
- **Windows (Command Prompt)**:
  ```cmd
  .venv\Scripts\activate.bat
  ```
- **Windows (PowerShell)**:
  ```powershell
  .venv\Scripts\Activate.ps1
  ```
*(Once activated, you will see `(.venv)` in front of your terminal prompt).*

### Step 3: Install Project Dependencies
Install all required libraries from the root `requirements.txt` file:
```bash
pip install -r requirements.txt
```

### Step 4: Register Kernel for Jupyter Notebooks & VS Code
To ensure Jupyter notebooks in `notebooks/` use this virtual environment:
```bash
python -m ipykernel install --user --name=aml-dl-venv --display-name="Python (.venv - AML/DL)"
```

---

## 4. Selecting the Kernel in VS Code

1. Open any notebook in `notebooks/01.EDA/` (e.g. `01_data_understanding.ipynb`).
2. In the top-right corner of the editor, click **Select Kernel**.
3. Choose **Python Environments...** &rarr; select **`.venv` (Python 3.x.x)** or **Python (.venv - AML/DL)**.
4. You are ready to run the cells seamlessly!

---

## 5. Deactivating the Environment
When you are done working:
```bash
deactivate
```
