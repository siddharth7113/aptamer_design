## 🗂️ Step 7.1: Directory Structure

Organize files like this:

```
aptamer_design/
├── __init__.py
├── data.py
├── models.py
├── pipeline.py
├── evaluation.py
├── cli.py
tests/
├── test_data.py
├── test_models.py
├── test_evaluation.py
examples/
├── aptamer_clean_no_binding.csv
├── aptamer_extracted.csv
├── sample_mlp.pth
results/
├── cli_output.csv
README.md
requirements.txt
setup.py  (optional)
```

---

## 📄 Step 7.2: `README.md`

Start with this template:

````markdown
# Aptamer Binding Prediction Platform (PoC)

A modular, inference-ready tool for predicting aptamer binding affinity using classical ML models.

## 🔍 What It Does

- Loads real aptamer datasets (CSV, Excel, TSV)
- Supports k-mer MLP and CNN inference
- Calculates regression metrics (MSE, Spearman, Pearson)
- Command-line interface with YAML config support

## 🧪 Getting Started

### Installation

```bash
pip install -r requirements.txt
```
````

### Run CLI

```bash
python -m aptamer_design.cli \
  --model mlp \
  --data examples/aptamer_extracted.csv \
  --output results/output.csv \
  --eval
```

Or with YAML config:

```bash
python -m aptamer_design.cli --config config.yaml
```

## ⚙️ Model Options

| Model | Description                   |
| ----- | ----------------------------- |
| `mlp` | k-mer frequency vector → MLP  |
| `cnn` | 1D convolution on one-hot DNA |

## 📁 Folder Overview

- `aptamer_design/`: Core logic
- `examples/`: Sample datasets
- `results/`: Saved outputs
- `tests/`: Unit tests

## ✅ Current Status

- [x] Inference pipeline
- [x] CLI tool
- [x] Test coverage
- [ ] Pretrained model integration (planned)

---

## 📚 Citation

UTexas Aptamer Database: [https://sites.utexas.edu/aptamerdatabase/](https://sites.utexas.edu/aptamerdatabase/)

````

---

## 📦 Step 7.3: `requirements.txt`

```txt
pandas
numpy
torch
scipy
pyyaml
````

---

## 🧪 Step 7.4: Validation

Before you wrap:

- [x] Run `pytest`
- [x] Run CLI once on sample file
- [x] Open `cli_output.csv` and confirm predictions
- [ ] Optionally upload code to GitHub with `README.md` as entry

---
