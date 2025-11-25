# How much do Europeans cycle? Cycling mode share in 864 European cities

Machine learning approach to predict cycling mode share for European cities using open data on urban characteristics, infrastructure, climate, and demographics.

## Overview

This repository accompanies the paper and contains code to:
- Collect and process city-level data from multiple open sources
- Train XGBoost classifiers to predict cycling mode share categories (low <5%, moderate 5-15%, high >15%)
- Generate predictions with uncertainty quantification for cities without survey data
- Reproduce all analyses and figures from the paper

## Requirements

Python 3.8+ with dependencies listed in `requirements.txt`

## Reproduction

Run scripts in order:
```bash
# 1. Collect data
python src/data/attributes.py
python src/data/eurostat.py
python src/data/urban_audit.py

# 2. Build and transform features
python src/features/build.py
python src/features/transform.py
python src/features/select.py

# 3. Train models
python src/model/classification.py
python src/model/conformal_prediction.py
python src/model/final_prediction.py

# 4. Run robustness checks
python src/model/rob_*.py

# 5. Generate figures 
python src/report/*.py
```

## Citation

[Add]

