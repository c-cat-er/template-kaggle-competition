# Technical Specification: Housing Price Prediction Model

> Specification Name: Housing Price Prediction Model  
> Document Version: 2  
> Kaggle Score: 13049.08243  
> Status: Approved for Pre-Production  
> Target Environment: Kubernetes / FastAPI Ingress  
> Author: https://www.kaggle.com/tmcater

---

## 1. Project Objective & Performance

- **Objective**: Build a high-performance ensemble regression framework using tree-based and linear algorithms via a 10-Fold Out-of-Fold (OOF) validation strategy.
- **Key Validation Metric**: Evaluated via Root Mean Squared Logarithmic Error (RMSLE).
- **Model Performance Results (Overall OOF RMSLE)**:
    - `XGBoost`: **0.12236**
    - `Lasso`: **0.13836**
    - `ElasticNet`: **0.13754**
    - `Ridge`: **0.13836**
    - **Final Blend (Ensemble)**: **0.12009** (Kaggle Score: **13,049.08243**)

---

## 2. Advanced Feature Engineering

In addition to foundational spatial transformations, v3.0 introduces interaction features targeting real estate valuation heuristics:

- `OverallQual_GrLivArea`: Interaction between subjective grade and ground living area.
- `OverallQual_TotalSF`: Interaction between quality and combined structural footprint.
- `GarageScore`: Joint scaling performance of vehicle capacity and total garage square footage (`GarageCars * GarageArea`).

---

## 3. Dual-Track Preprocessing Architecture

To satisfy the structural differences between tree boosters and linear models, data flows through two parallel `ColumnTransformer` tracks:

┌──► Tree Track ───► Median Imputer (No Scaling)  
│  
Raw Data ──► Feature Engineering ─────┼──► Categorical ──► Mode Imputer ──► OneHot/Ordinal  
│  
└──► Linear Track ──► Median Imputer ──► StandardScaler

- **Tree Preprocessing Pipeline**: Skips computational scaling to preserve tree splitting speed. Continuous features undergo median imputation only.
- **Linear Preprocessing Pipeline**: Enforces `StandardScaler()` following median imputation to ensure gradient descent stability for Lasso, Ridge, and ElasticNet.
- **Shared Categorical Configuration**:
    - _Ordinal_: 9 quality metrics mapped to hierarchy `["None", "Po", "Fa", "TA", "Gd", "Ex"]`.
    - _Nominal_: 35 columns processed via `OneHotEncoder(handle_unknown="ignore")`.

---

## 4. 10-Fold Out-of-Fold (OOF) Cross-Validation

To completely eliminate data leakage during model blending, predictions are gathered out-of-fold using a strict 10-Split cross-validation architecture:

- **Splits**: `N_SPLITS = 10`, shuffled with a static random seed (`42`).
- **Tree Execution**: `XGBRegressor` tuned with early evaluation markers (`n_estimators: 1500`, `learning_rate: 0.05`, `subsample: 0.8`).
- **Linear Execution**: Regularization penalties optimized via Lasso ($\alpha=0.0005$), ElasticNet ($\alpha=0.0005$, $l1\_ratio=0.5$), and Ridge ($\alpha=12$).

---

## 5. Optimal Ensemble Blending Structure

Final test inferences are compiled via a weighted linear combination of log-price outputs before being mapped back to the nominal scale via `np.expm1()`:

$$\text{Final Preds (Log)} = (0.70 \times \text{XGBoost}) + (0.15 \times \text{Lasso}) + (0.15 \times \text{Ridge})$$

- **Output Sanity Check**: Post-inference pipeline runs full `NaN` rejection assertions before building the standard `submission.csv` layout.
