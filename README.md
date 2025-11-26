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
- **Dataset Format:** CSV files from Kaggle (easier to work with than SAS XPT)
- **Files Used:**
  - `demographic.csv` (10,175 rows, 47 columns) - Demographics
  - `questionnaire.csv` (10,175 rows, 953 columns) - All questionnaire responses
- **Target:** Binary classification - visited dentist ≤ 12 months (derived from `OHQ030`)
  - `OHQ030` codes: 1 = <6 months, 2 = 6-12 months → **target = 1**
  - All other valid responses → **target = 0**
- **Features Selected:**
  - Categorical: `RIAGENDR` (sex), `RIDRETH3` (race/ethnicity), `DMDEDUC2` (education)
  - Numeric: `RIDAGEYR` (age), `INDFMPIR` (income-to-poverty ratio)

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

1. ✅ **Setup and brand style** - Load Periospot palette, configure matplotlib
2. ✅ **Get the data** - Load NHANES CSV files (demographic.csv + questionnaire.csv)
3. ✅ **Clean and EDA** - Handle missingness, basic visualizations
4. ⏳ **Train/test split** - Stratified 80/20 split
5. ⏳ **CatBoost baseline** - Native categorical handling
6. ⏳ **XGBoost baseline** - One-hot encoding pipeline
7. ⏳ **LightGBM baseline** (optional) - Third comparison
8. ⏳ **Threshold policy** - Find optimal threshold for policy decisions
9. ⏳ **SHAP explanations** - Interpretability with TreeExplainer
10. ⏳ **Compare and save** - Model comparison, artifact saving
11. ⏳ **Model card** - Documentation template

**Legend:** ✅ Completed | 🚧 In Progress | ⏳ Pending

---

## 📊 Exploratory Data Analysis

### Data Distribution Overview

After loading and merging the NHANES datasets, we performed initial exploratory analysis to understand the target distribution and key demographic patterns. Here are the key findings:

#### 1. Class Balance

The dataset shows a moderate class imbalance:

<div align="center">

<img src="images/class_balance.png" alt="Class Balance - Distribution of target variable" width="680" />

</div>

**Key Findings:**
- **No Visit (>12 months):** ~6,100 instances (~63%)
- **Visited (≤12 months):** ~3,600 instances (~37%)
- **Imbalance Ratio:** ~1.7:1

This moderate imbalance suggests we should consider:
- Class weights during model training
- PR-AUC as a primary metric (better for imbalanced data than ROC-AUC)
- Threshold tuning to optimize recall/precision trade-off

#### 2. Target Rate by Age Group

Age is a strong predictor of dental visit frequency:

<div align="center">

<img src="images/target_rate_by_age.png" alt="Target Rate by Age Group" width="680" />

</div>

**Key Findings:**
- **<18 years:** Highest target rate at ~73% - Young adults/teens most likely to visit
- **18-35 years:** Lowest target rate at ~51% - Drop likely due to life transitions, less parental oversight
- **35-50 years:** ~56% - Gradual recovery as people establish routines
- **50-65 years:** ~57% - Continued improvement
- **65+ years:** ~59% - Highest among adults, likely due to insurance and health awareness

**Insights:**
- Strong age effect - model should capture this well
- Non-linear relationship suggests tree-based models (CatBoost/XGBoost) will excel
- Younger cohort (<18) may have different visit patterns due to family dynamics

#### 3. Target Rate by Sex

Sex shows a slight but noticeable difference:

<div align="center">

<img src="images/target_rate_by_sex.png" alt="Target Rate by Sex" width="680" />

</div>

**Key Findings:**
- **Female:** ~64-65% target rate
- **Male:** ~60% target rate
- **Difference:** ~4-5 percentage points

**Insights:**
- Females slightly more likely to visit dentists regularly
- This pattern aligns with general healthcare-seeking behavior differences
- Sex should be included as a categorical feature

### Data Cleaning Summary

- **Missing Value Handling:**
  - Numeric features (`RIDAGEYR`, `INDFMPIR`): Imputed with median
  - Categorical features (`RIAGENDR`, `RIDRETH3`, `DMDEDUC2`): Imputed with most frequent mode
- **Target Filtering:**
  - Excluded invalid responses (refused: 77, don't know: 99, missing: NaN)
  - Kept only valid responses (codes 1-7)
- **Final Dataset:**
  - Ready for train/test split and modeling

---

## 🏆 Results

*Results will appear here after running the notebook.*

Final test metrics will include:
- PR-AUC (average precision)
- ROC-AUC
- Precision and Recall at optimal threshold
- SHAP feature importance plots

---

## 📈 Project Progress

### ✅ Completed

- [x] **Project structure** - Folder organization, brand palette JSON, README
- [x] **Section 0: Setup** - Brand palette loading, matplotlib configuration, folder creation
- [x] **Section 1: Data Loading** - Successfully loaded NHANES data from CSV files
  - Fixed CSV loading (Kaggle dataset uses CSV, not SAS XPT)
  - Loaded `demographic.csv` (10,175 rows, 47 columns)
  - Loaded `questionnaire.csv` (10,175 rows, 953 columns)
  - Merged datasets on `SEQN` (participant ID)
  - Built binary target from `OHQ030` (visited dentist ≤ 12 months)
  - Target class balance: ~63% no visit, ~37% visited (moderate imbalance)
- [x] **Section 2: EDA and Cleaning** - Exploratory data analysis completed
  - Handled missing values (median for numeric, mode for categorical)
  - Created visualizations: class balance, target rate by age group, target rate by sex
  - Key insights: Strong age effect (youngest highest at 73%, 18-35 lowest at 51%)
  - Sex shows slight difference (females ~65%, males ~60%)
  - Identified moderate class imbalance requiring appropriate metrics

### 🚧 In Progress

- [ ] **Section 3: Train/Test Split** - Stratified 80/20 split for model training

### 📋 Upcoming

- [ ] **Section 3-6:** Model training (CatBoost, XGBoost, LightGBM)
- [ ] **Section 7-8:** Threshold policy and SHAP explanations
- [ ] **Section 9-10:** Model comparison and artifacts

### 📊 Current Dataset Stats

- **Total participants:** ~9,760 (after filtering invalid responses)
- **Features selected:**
  - Categorical: `RIAGENDR` (sex), `RIDRETH3` (race/ethnicity), `DMDEDUC2` (education)
  - Numeric: `RIDAGEYR` (age), `INDFMPIR` (income-to-poverty ratio)
- **Target:** Binary (visited dentist in last 12 months: Yes/No)
- **Target distribution:** 
  - No visit (>12 months): ~6,100 (63%)
  - Visited (≤12 months): ~3,600 (37%)
  - **Imbalance ratio:** 1.7:1
- **Key Patterns:**
  - Age is strongest predictor (73% for <18, 51% for 18-35)
  - Females slightly more likely to visit (65% vs 60%)

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

