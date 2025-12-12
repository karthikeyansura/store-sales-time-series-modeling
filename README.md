# Store Sales - Time Series Forecasting

This project addresses the **Store Sales - Time Series Forecasting** challenge hosted on Kaggle. The primary objective is to predict unit sales for thousands of product families at distinct Corporación Favorita store locations in Ecuador. By leveraging historical sales data, store metadata, and macroeconomic indicators, this study employs advanced machine learning techniques to generate accurate forecasts for a 16-day future horizon.

**Competition Link:** [Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)

## Methodology

The project pipeline follows a structured data science approach, emphasizing robust feature engineering and gradient boosting algorithms.

### 1. Data Preprocessing
The dataset consists of relational tables including training data, test data, store metadata, daily oil prices, and holiday events. Key preprocessing steps included:
* **Temporal Integration:** Concatenation of training and test datasets prior to feature generation to ensure continuity in lag-based features.
* **Imputation:** Application of forward-fill and backward-fill strategies for missing oil prices; zero-imputation for future transactions to prevent data leakage.
* **Anomaly Handling:** Creation of binary indicators for the April 16, 2016, earthquake to isolate outlier events.

### 2. Feature Engineering
To capture temporal dependencies and seasonality, the following features were engineered:
* **Lag Features:** Historical sales values at intervals of 1, 7, 14, and 28 days to capture autocorrelation.
* **Rolling Statistics:** Moving averages and standard deviations over 7, 14, and 28-day windows to identify trends and volatility.
* **Calendar Attributes:** Extraction of cyclical date components (day of week, month, payday indicators) and specific holiday flags utilizing the `holidays_events.csv` metadata.

### 3. Model Architecture
Three gradient boosting frameworks were implemented and evaluated to handle the tabular nature of the data:
* **LightGBM:** Utilized for its training efficiency and ability to handle large-scale datasets.
* **XGBoost:** Employed as a robust baseline for gradient boosted decision trees.
* **CatBoost:** Selected for its effective handling of categorical variables without extensive preprocessing.

### 4. Hyperparameter Optimization
Model performance was optimized using both Randomized Search and Bayesian Optimization (via `scikit-optimize`) to minimize the validation error.

## Evaluation

The models were evaluated using **Root Mean Squared Logarithmic Error (RMSLE)**. This metric was chosen to penalize relative errors rather than absolute errors, aligning with the competition's scoring criteria and the nature of retail sales distributions.

## Repository Structure

* `store_sales_time_series_modeling.ipynb`: Primary notebook containing the end-to-end pipeline (Data Loading, EDA, Feature Engineering, Modeling, and Evaluation).
* `dataset/`: Directory containing the competition datasets (e.g., `train.csv`, `test.csv`, `stores.csv`).
* `submission.csv`: Final model predictions formatted for Kaggle submission.

## Usage

To reproduce the results, ensure the following dependencies are installed:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn lightgbm xgboost catboost scikit-optimize tensorflow
