# Applied Machine Learning for Healthcare: Diabetes Risk Classification Using the CDC BRFSS 2021 Dataset

**Author:** Leo Littlewood | lsjl2@cam.ac.uk  
**Institution:** University of Cambridge — Health Data Science MSt  
**Module:** Module 5: Applied Machine Learning in Healthcare  

---

## Abstract

Diabetes is one of the largest health burdens worldwide, with over 537 million adults currently affected, placing considerable strain on healthcare systems in terms of time, cost and capacity. Both prevention and earlier intervention could be meaningfully improved by risk stratification using accurate, interpretable machine learning models.

This study focuses on the CDC Behavioural Risk Factor Surveillance System (BRFSS) 2021 dataset, with 223,243 unique respondents self-reporting 21 health indicators, alongside 6 engineered features, applying supervised machine learning to classify diabetes status. Two classifiers, Logistic Regression (LR) and Random Forest (RF), were compared, with LR acting as an interpretable linear baseline and RF selected for its high capacity and ability to handle non-linear relationships. A stratified 5-fold cross-validation was used to evaluate the models, with grid and randomised search used for tuning and SMOTE used within the pipeline to mitigate the substantial class imbalance.

LR marginally outperformed RF in AUC-ROC (0.807 vs 0.802) on the test set, while their precision-recall profiles differed substantially. LR prioritised recall compared to RF (0.749 vs 0.253), but at the expense of diminished precision (0.316 vs 0.469). The strongest predictor in both RF importance and SHAP analyses was CardioRisk_Score, a composite engineered feature, followed by GeneralHealth and Cholesterol. Both models were exceptionally stable across 10 random seeds (SD < 0.003).

---

## Repository Contents

```
├── HDS_MM_LEOLITTLEWOOD_2604.ipynb   # Main analysis notebook
├── README.md                          # This file
```

---

## Dataset

The dataset used is the **CDC Behavioural Risk Factor Surveillance System (BRFSS) 2021**, accessed via Kaggle:

> Teboul, A. (2022). *Diabetes Health Indicators Dataset (CDC BRFSS 2021)*. Kaggle.  
> https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset

The dataset is loaded automatically within the notebook directly from the repository's hosted CSV file:

```
https://raw.githubusercontent.com/leosijunltw/HDS_MM_LEOLITTLEWOOD_2604/main/diabetes_binary_health_indicators_BRFSS2021.csv
```

No manual download is required — simply run the notebook from top to bottom.

---

## How to Reproduce the Results

### 1. Clone the repository

```bash
git clone https://github.com/leosijunltw/HDS_MM_LEOLITTLEWOOD_2604.git
cd HDS_MM_LEOLITTLEWOOD_2604
```

### 2. Install required libraries

All required packages are installed automatically in Cell 2 of the notebook. Alternatively, install manually:

```bash
pip install numpy matplotlib pandas scikit-learn imbalanced-learn shap seaborn
```

### 3. Run the notebook

Open the notebook in Jupyter or VS Code and run all cells from top to bottom:

```bash
jupyter notebook HDS_MM_LEOLITTLEWOOD_2604.ipynb
```

All outputs including plots, model metrics, and the comparison table will be generated automatically. Figures are saved to an `outputs/` folder created in the same directory.

### 4. Reproducibility

A global random seed (`RANDOM_SEED = 42`) is set at the top of the notebook and passed explicitly to all stochastic components (train/test split, SMOTE, KMeans, RandomForest, RandomizedSearchCV, and the stability analysis). All results are fully reproducible across runs.

---

## Library Versions

The notebook installs the latest versions of all packages automatically. The analysis was developed and tested with the following versions:

| Library | Version |
|---|---|
| Python | 3.12+ |
| numpy | 2.x |
| pandas | 2.x |
| scikit-learn | 1.x |
| imbalanced-learn | 0.12.x |
| shap | 0.46.x |
| matplotlib | 3.x |
| seaborn | 0.13.x |

---

## Analysis Pipeline

The notebook follows this pipeline:

1. **Data loading** — automatic download from GitHub URL
2. **Feature selection and engineering** — all 21 BRFSS variables retained plus 6 engineered features (BMI_x_Age, CardioRisk_Score, DietScore, LifestyleRisk_Score, PoorHealth_Days, BMI_Inactive)
3. **Preprocessing** — duplicate removal, SMOTE class balancing within CV folds, StandardScaler normalisation
4. **Exploratory analysis** — PCA biplot, k-means clustering (k=3), correlation heatmap, class distribution
5. **Model training** — Logistic Regression (GridSearchCV) and Random Forest (RandomizedSearchCV) with stratified 5-fold cross-validation
6. **Evaluation** — AUC-ROC, F1, precision, recall, accuracy, PR-AUC, confusion matrices, ROC and PR curves, calibration curves, classification report
7. **Stability analysis** — AUC-ROC across 10 random seeds
8. **Explainability** — RF feature importances, LR coefficients, SHAP beeswarm and summary bar plots
