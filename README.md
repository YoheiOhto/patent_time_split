# patent_time_split

**Time-aware Toxicity Prediction & Temporal Analysis on SureChEMBL Compounds**

---

## Overview

`patent_time_split` provides tools and experiments for **time-split toxicity prediction** using chemical compound datasets such as SureChEMBL.  
The repository focuses on how temporal dynamics influence predictive performance and how structural features of molecules evolve over time in relation to toxicity.

---

## Publication

Not yet published (work in progress).

---

## Motivation

- Traditional models assume data is static, but in real-world settings (e.g., patents, drug discovery), data evolves over time.  
- This project investigates **temporal generalization** by splitting datasets chronologically and testing predictive performance on future data.  
- Provides insights into:
  - How predictive models behave when trained on past vs. future chemical data.
  - Structural and toxicological trends across different time periods.
  - Comparison of multiple datasets (SureChEMBL, DrugBank, DSSTox, MoleculeNet, TDC).

---

## Repository Structure
```
  patent_time_split/
  ├── data/
  │ ├── raw/ # Raw datasets (SureChEMBL, DrugBank, DSSTox, etc.)
  │ ├── defreezed/ # Pre-processed intermediate data
  │ ├── processed/ # Final processed datasets for modeling
  │ └── result/ # Experimental results (p-values, scores, visualizations)
  ├── notebook/
  │ ├── preprocess/ # Preprocessing notebooks
  │ └── experiment/ # Experiment & analysis notebooks
  ├── src/
  │ ├── chemo_process.py # Chemical feature engineering
  │ └── util.py # Utility functions
  └── README.md
```

---

## Authors

Yohei Ohto — Main implementation
Tadahaya Mizuno — Supervision & collaboration

---

## Contact:

oy826c60[at]gmail.com
tadahaya[at]gmail.com, tadahaya[at]mol.f.u-tokyo.ac.jp