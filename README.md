# 🦷 NHANES Oral Health Predictor

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![Framework](https://img.shields.io/badge/Framework-CatBoost%2FXGBoost-orange.svg)
![Status](https://img.shields.io/badge/Status-Active-green.svg)

**Predicting dental visit frequency from NHANES demographics and oral health data**

[🎯 Overview](#-project-overview) • [📊 Results](#-results) • [🚀 Quick Start](#-quick-start) • [📝 Notebook](#-notebook)

</div>

> Building a reproducible baseline comparing CatBoost vs XGBoost for predicting "visited a dentist in the last 12 months" using NHANES data. Focus on tabular ML best practices: PR-AUC, threshold policy, and SHAP explanations.

---

## 👨‍💻 Author

<div align="center">

**Francisco Teixeira Barbosa**

[![GitHub](https://img.shields.io/badge/GitHub-Tuminha-black?style=flat&logo=github)](https://github.com/Tuminha)
[![Kaggle](https://img.shields.io/badge/Kaggle-Profile-20BEFF?style=flat&logo=kaggle&logoColor=white)](https://www.kaggle.com/franciscotbarbosa)
[![Email](https://img.shields.io/badge/Email-cisco%40periospot.com-blue?style=flat&logo=gmail)](mailto:cisco@periospot.com)
[![Twitter](https://img.shields.io/badge/Twitter-cisco__research-1DA1F2?style=flat&logo=twitter)](https://twitter.com/cisco_research)

*Learning Machine Learning through hands-on projects • Building AI solutions for healthcare*

</div>

---

## 🎯 Project Overview

### What
This project predicts whether an individual visited a dentist in the last 12 months using demographic and oral health questionnaire data from the National Health and Nutrition Examination Survey (NHANES).

### Why
- Learn tabular ML best practices with real public health data
- Compare gradient boosting algorithms (CatBoost vs XGBoost vs LightGBM)
- Practice PR-AUC evaluation and threshold policy selection
- Generate interpretable models with SHAP explanations
- Maintain brand-consistent visualizations using Periospot palette

### Expected Outcome
A single, end-to-end notebook that demonstrates:
- Clean target definition from survey data
- CatBoost's native categorical handling
- PR-AUC vs ROC-AUC trade-offs
- Policy-driven threshold selection
- SHAP-based model interpretability

---

## 🎓 Learning Objectives

- [x] Working with NHANES SAS XPT files
- [x] Handling missing data in survey datasets
- [x] Comparing multiple gradient boosting frameworks
- [x] Evaluating with PR-AUC (average precision)
- [x] Selecting optimal classification thresholds
- [x] Generating SHAP explanations for interpretability
- [x] Creating brand-consistent visualizations

---

## 📊 Dataset / Domain

- **Source:** NHANES (CDC National Health and Nutrition Examination Survey)
- **Cycle:** 2017-2018 (adjustable)
- **Components:**
  - **DEMO:** Demographics (age, sex, race/ethnicity, education, income-to-poverty ratio)
  - **OHQ:** Oral Health Questionnaire (last dental visit question)
- **Target:** Binary classification - visited dentist ≤ 12 months (derived from `OHQ030`)
- **Features:** Categorical (sex, race/ethnicity, education) + Numeric (age, income ratio)

---

## 🚀 Quick Start

### Prerequisites

```bash
pip install catboost xgboost lightgbm shap pandas numpy matplotlib seaborn scikit-learn joblib tabulate pyyaml
```

Or install from requirements (if provided):
```bash
pip install -r requirements.txt
```

### Setup

1. **Clone or download this repository**
   ```bash
   cd NHANES-oral-health-Predictor
   ```

2. **Download NHANES data**
   - Option A: Use Kaggle CLI
     ```bash
     kaggle datasets download -d cdc/national-health-and-nutrition-examination-survey -p data/raw
     unzip -o data/raw/national-health-and-nutrition-examination-survey.zip -d data/raw
     ```
   - Option B: Download directly from [CDC NHANES website](https://www.cdc.gov/nchs/nhanes/index.htm)

3. **Open the notebook**
   ```bash
   jupyter notebook notebooks/nhanes_dental_visits_one_notebook.ipynb
   ```

4. **Follow the TODO instructions** in each code cell to complete the implementation

---

## 📝 Notebook

The main notebook (`notebooks/nhanes_dental_visits_one_notebook.ipynb`) is structured with:

- **70% Markdown** - Explanations, instructions, and context
- **30% Code cells** - TODOs with hints for you to implement

### Sections:

1. **Setup and brand style** - Load Periospot palette, configure matplotlib
2. **Get the data** - Load NHANES DEMO and OHQ XPT files
3. **Clean and EDA** - Handle missingness, basic visualizations
4. **Train/test split** - Stratified 80/20 split
5. **CatBoost baseline** - Native categorical handling
6. **XGBoost baseline** - One-hot encoding pipeline
7. **LightGBM baseline** (optional) - Third comparison
8. **Threshold policy** - Find optimal threshold for policy decisions
9. **SHAP explanations** - Interpretability with TreeExplainer
10. **Compare and save** - Model comparison, artifact saving
11. **Model card** - Documentation template

---

## 🏆 Results

*Results will appear here after running the notebook.*

Final test metrics will include:
- PR-AUC (average precision)
- ROC-AUC
- Precision and Recall at optimal threshold
- SHAP feature importance plots

---

## 🛠 Technical Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Data Processing | Pandas, NumPy | Loading SAS XPT, feature engineering |
| Visualization | Matplotlib, Seaborn | EDA and result plots |
| ML Frameworks | CatBoost, XGBoost, LightGBM | Gradient boosting models |
| Interpretability | SHAP | Model explanations |
| Evaluation | Scikit-learn | Metrics and curves |
| Pipeline | Scikit-learn Pipeline | Preprocessing + modeling |

---

## 🎨 Brand Palette

This project uses the Periospot brand palette defined in `brand_palette.json`:

- **periospot_blue:** `#15365a`
- **mystic_blue:** `#003049`
- **periospot_red:** `#6c1410`
- **crimson_blaze:** `#a92a2a`
- **vanilla_cream:** `#f7f0da`

All plots and visualizations follow this color scheme for brand consistency.

---

## 📝 Learning Journey

- Tabular ML with gradient boosting frameworks
- Handling categorical features natively vs one-hot encoding
- PR-AUC evaluation for imbalanced classification
- Threshold selection for policy-driven decisions
- Model interpretability with SHAP
- Brand-consistent data visualization

---

## 🚀 Next Steps

- [ ] Complete all TODO sections in the notebook
- [ ] Experiment with feature engineering
- [ ] Try hyperparameter tuning with Optuna
- [ ] Add calibration curves
- [ ] Explore demographic bias analysis
- [ ] Consider survey weights in modeling
- [ ] Extend to multiple NHANES cycles

---

## ⚠️ Caveats

- Survey design weights are ignored in this baseline
- Not intended as clinical advice
- Potential demographic bias in predictions
- Missing data imputation assumptions (median/mode)
- Model trained on single NHANES cycle; may not generalize

---

## 📄 License

MIT License (see [LICENSE](LICENSE) if provided)

---

<div align="center">

**⭐ Star this repo if you found it helpful! ⭐**  
*Building AI solutions for healthcare, one dataset at a time* 🚀

</div>

