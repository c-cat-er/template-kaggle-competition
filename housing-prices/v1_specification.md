# Housing Price Prediction Model

> **Document Version:** 1  
> **Kaggle Score**: 14931.67157  
> **Status:** Approved for Pre-Production  
> **Target Environment:** Kubernetes / FastAPI Ingress  
> **Author:** [tmcater](https://www.kaggle.com/tmcater)

---

## 1. System Overview & Business Objective

### 1.1 Objective

The system deploys a high-precision, multi-feature tabular regression pipeline designed to predict residential property values (`SalePrice`) based on structural, spatial, and asset-specific physical features.

### 1.2 Performance Metrics & SLA

- **Offline Validation Metrics**:
    - Baseline Random Forest Validation MAE: **17,350.54**
    - Tuned XGBoost 5-Fold Cross-Validation (Log-scale MAE): **0.09**
    - Final Selected Model Validation MAE: **17,328.13**
- **Production SLA Target**:
    - P95 Inference Latency: **< 15 ms** per single-record request.
    - Peak Throughput Capability: **200 QPS** (Queries Per Second).

---

## 2. Data Engineering & Feature Ingestion Pipeline

### 2.1 Upstream Ingestion & Target Engineering

- **Inference Constraints**: Raw input targets are often right-skewed. The system applies a log-normal transformation during training to stabilize booster gradients.
    - _Training Target transformation_: $$y_{transformed} = \ln(x + 1)$$ via `np.log1p()`.
    - _Runtime Inference Reversion_: $$x_{original} = e^{y} - 1$$ via `np.expm1()`.
- **Data Dropping Boundary**: Row elimination is strictly constrained to target omissions. Input instances lacking `SalePrice` in training data are completely purged (`dropna(subset=["SalePrice"])`).

### 2.2 Domain-Specific Imputation Rules

To preserve actual physical meaning, stochastic missing values (`NaN`) are decoupled from standard statistical imputation and handled via explicit conditional semantics:

1.  **Categorical Facility Absence**: Columns indicating absence of structures (`PoolQC`, `GarageType`, `Alley`, etc.) are mapped to a strict string literal `'None'`.
2.  **Numerical Asset Deficit**: Fields tracking non-existent structural units (`GarageArea`, `GarageCars`, `BsmtFullBath`, etc.) are explicitly assigned a quantitative value of `0`.

### 2.3 Synthesized Feature Engineering

Four primary deterministic business metrics are calculated programmatically inside `add_custom_features(df)` before data ingestion into transformers:

- **Total Living Space Area (`TotalSF`)**:
  $$\text{TotalSF} = \text{TotalBsmtSF} + \text{1stFlrSF} + \text{2ndFlrSF}$$
- **Aggregate Fractional Bathroom Metric (`TotalBaths`)**:
  $$\text{TotalBaths} = \text{FullBath} + (0.5 \times \text{HalfBath}) + \text{BsmtFullBath} + (0.5 \times \text{BsmtHalfBath})$$
- **Property Depreciation Profiles (`HouseAge`, `RemodAge`)**:
  $$\text{HouseAge} = \text{YrSold} - \text{YearBuilt}$$
  $$\text{RemodAge} = \text{YrSold} - \text{YearRemodAdd}$$
- **Consolidated Outdoor Footprint (`TotalOutsideSF`)**:
  $$\text{TotalOutsideSF} = \text{WoodDeckSF} + \text{OpenPorchSF} + \text{EnclosedPorch} + \text{3SsnPorch} + \text{ScreenPorch}$$

---

## 3. Preprocessing & Sklearn Pipeline Architecture

To guarantee maximum isolation and completely eradicate data leakage, raw inputs are split into three structured parallel tracks via an immutable scikit-learn `ColumnTransformer`:

┌──► Numerical (40 features) ──► Median Imputer ──► StandardScaler ──┐  
│  
Raw Features (X) ─────┼──► Ordinal (9 features) ─────► Mode Imputer ────► OrdinalEncoder ──┼──► Concatenated Array  
│  
└──► Nominal (35 features) ────► Mode Imputer ──► OneHotEncoder ──┘

### 3.1 Pipeline Configurations

- **Numerical Pipeline (40 Features)**: Implements median tracking fallback for stochastic test missing values (`SimpleImputer(strategy="median")`) followed by variance scaling (`StandardScaler()`).
- **Ordinal Pipeline (9 Features)**: Enforces specific physical quality orders:
    - _Monitored Columns_: `ExterQual`, `ExterCond`, `BsmtQual`, `BsmtCond`, `HeatingQC`, `KitchenQual`, `FireplaceQu`, `GarageQual`, `GarageCond`.
    - _Predefined Order Map_: `["None", "Po", "Fa", "TA", "Gd", "Ex"]`.
    - _Leakage Safeguard_: Out-of-vocabulary test ratings fallback to encoded value `-1` via `handle_unknown="use_encoded_value"`.
- **Nominal Pipeline (35 Features)**: Converts non-ordered text descriptors utilizing standard `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`.

---

## 4. Hyperparameter Search & Production Retraining

### 4.1 Cross-Validation Strategy

The search architecture optimizes tree booster depth and allocation limits using a static 5-Fold Grid Search Cross-Validation (`cross_val_score`) targeting logarithmic Mean Absolute Error.

### 4.2 Candidate Tree Sweep Results

n_estimators=100 -> CV MAE: 0.09n_estimators=250 -> CV MAE: 0.09n_estimators=500 -> CV MAE: 0.09n_estimators=750 -> CV MAE: 0.09 <-- [Selected Optimal Configuration]n_estimators=1000 -> CV MAE: 0.09\* **Optimal Hyperparameters**: `XGBRegressor(n_estimators=750, learning_rate=0.05, max_depth=6)`.

- **Full-Data Convergence**: Following optimization parameter localization, the pipeline runs an un-split, full-dataset fit execution on $X_{all}$ and $y_{log}$ before final export.
