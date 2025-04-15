# Optiver Realized Volatility Prediction: Model Comparison

## Overview

Volatility, the degree of price variation over time, is a cornerstone metric in quantitative finance, essential for risk management, option pricing, and trading strategy development. This project tackles the prediction of short-term (10-minute) realized volatility for various stocks using high-frequency order book and trade data provided by the [Optiver Realized Volatility Prediction Kaggle competition](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/data).

The analysis implements comprehensive feature engineering and compares the predictive performance of three distinct machine learning methodologies:

1.  **TabNet + Support Vector Machine (SVM)**: A hybrid leveraging TabNet's attention mechanism for tabular data with SVM's strength in regression tasks.
2.  **XGBoost with GridSearchCV**: An optimized gradient boosting model, tuned for performance via systematic hyperparameter search.
3.  **LightGBM**: An efficient gradient boosting framework employing techniques like GOSS and EFB for speed and scalability, particularly relevant for large financial datasets.

The ultimate aim is to determine the most accurate approach for forecasting volatility, primarily evaluated using Root Mean Squared Percentage Error (RMSPE).

## Project Goal

*   **Predict Target Volatility:** Forecast the realized volatility (`target`) for 10-minute buckets associated with specific `stock_id` and `time_id` combinations.
*   **Feature Engineering:** Construct relevant features from granular book and trade data, capturing market microstructure dynamics. Key derived features include:
    *   Weighted Average Prices (WAP 1/2/3/4)
    *   Log Returns based on WAPs
    *   Realized Volatility of Log Returns over the 10-minute bucket and sub-intervals (e.g., last 100, 200, 300, 400, 500 seconds)
    *   Price and Volume Spreads/Imbalances
*   **Model Comparison:** Systematically implement, train (using 5-fold cross-validation), and evaluate TabNet+SVM, XGBoost+GridSearch, and LightGBM.
*   **Evaluation Metric:** Quantify prediction error using Root Mean Squared Percentage Error (RMSPE), calculated as:
    `RMSPE = sqrt(mean(((y_true - y_pred) / y_true)^2))` where `y_true` is the actual target volatility and `y_pred` is the model's prediction.

## Data

1.  **Source & Link:** Data originates from the **Optiver Realized Volatility Prediction competition on Kaggle**: [https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/data](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/data)
2.  **Raw Data Files:**
    *   `train.csv`: Contains `stock_id`, `time_id`, and the target `volatility` for training. (428,932 instances)
    *   `test.csv`: Provides `stock_id` and `time_id` for submission prediction (target is hidden).
    *   `book_[train/test].parquet/`: Order book snapshots (bid/ask prices/sizes) for each stock-time window.
    *   `trade_[train/test].parquet/`: Executed trades data (price, size, order count) for each stock-time window.
3.  **Data Relationships & Key Calculations:**
    *   `train.csv` maps `stock_id` and `time_id` pairs to the `target` volatility.
    *   Order book data (`book_*.parquet`) provides snapshots used to calculate features like Weighted Average Price (WAP):
        `WAP = (BidPrice1*AskSize1 + AskPrice1*BidSize1) / (BidSize1 + AskSize1)` (and variations for level 2, or using bid/ask sizes respectively)
    *   Log returns are derived from sequential WAP values: `LogReturn = log(WAP_t / WAP_{t-1})`
    *   Realized volatility (the core concept related to the `target`) is calculated from log returns within a time window: `RealizedVol = sqrt(sum(LogReturn^2))`
    *   Trade data (`trade_*.parquet`) provides features based on actual transaction volume and counts.
4.  **Data Placement:** Raw files should be downloaded from Kaggle and placed in the `data/original/` directory. Update paths within notebooks if placed elsewhere.
5.  **Processed Data:** The notebook `notebooks/1_Data_Preprocessing.ipynb` aggregates features from raw book/trade data, creating `new_train.pkl` and `new_test.pkl` (typically placed in `data/processed/`). **These large files are gitignored and must be generated locally by running Notebook 1.**
6.  **Evaluation Note:** As Kaggle hides the true test set book/trade data, model performance is evaluated using 5-fold cross-validation on the provided `train.csv` and its associated book/trade data.
7.  **Data Visualization:** Initial analysis (`0_Data_Description.ipynb`) reveals the target volatility distribution is positively skewed with significant kurtosis, suggesting potential challenges for models assuming normality and highlighting the utility of robust methods or appropriate transformations/weighting.

## Directory Structure


.
├── .gitignore # Ignores large data files (.pkl, .parquet, .csv), temp files, etc.
├── README.md # This project overview file
├── requirements.txt # Python dependencies (Must be generated)
│
├── data/ # Root directory for project data (content mostly gitignored)
│ ├── README_DATA.md # Create this: Detailed data instructions (Kaggle download, placement)
│ ├── original/ # Raw data from Kaggle (content gitignored)
│ └── processed/ # Processed data (e.g., .pkl files - content gitignored)
│
├── notebooks/ # Jupyter notebooks for analysis workflow
│ ├── 0_Data_Description.ipynb # Target variable exploration
│ ├── 1_Data_Preprocessing.ipynb # Feature engineering from book/trade data (generates .pkl files)
│ ├── 2_Method_TabNet_SVM.ipynb # Implementation/evaluation of TabNet + SVM
│ ├── 3_Method_XGBoost.ipynb # Implementation/evaluation of XGBoost + GridSearch
│ ├── 4_Method_LightGBM.ipynb # Implementation/evaluation of LightGBM
│ └── utils/ # Supplementary/experimental notebooks
│ └── LightGBM_Trial.ipynb # Initial LightGBM experimentation
│
├── scripts/ # (Optional) Placeholder for reusable utility scripts
│
└── reports/ # (Optional) Placeholder for generated reports/figures (typically gitignored)

## Methodology Explored

*   **TabNet:** Chosen for its effectiveness on tabular data, utilizing a sequential attention mechanism for feature selection and built-in categorical embedding support (`stock_id`). Trained optimizing directly for RMSPE using weighted loss (inverse square of target).
*   **SVM (Support Vector Machine):** Used in ensemble with TabNet. A classical ML model capable of capturing non-linear relationships via kernels. Tuned using GridSearchCV (`C`, `kernel`, `epsilon`).
*   **XGBoost:** A performance-focused gradient boosting implementation using regularization (L1/Lasso tested via `alpha`) and efficient tree construction. Feature selection based on LightGBM importance was applied before training. Hyperparameters (`learning_rate`, `max_depth`, `alpha`) optimized using GridSearchCV within each cross-validation fold.
*   **LightGBM:** An efficient gradient boosting framework using Gradient-based One-Side Sampling (GOSS) and Exclusive Feature Bundling (EFB), designed for speed and scalability on large datasets common in finance. Feature importance assessed using both 'split' and 'gain' metrics. Trained optimizing for RMSE but evaluated using a custom RMSPE metric (`feval_RMSPE`).

## Setup Instructions

1.  **Clone Repository:**
    ```bash
    git clone https://github.com/trenton-lau/optiver-volatility-prediction.git
    cd optiver-volatility-prediction
    ```
2.  **Create Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    # venv\Scripts\activate  # Windows
    ```
3.  **Install Dependencies:** (Generate `requirements.txt` via `pip freeze > requirements.txt` first!)
    ```bash
    pip install -r requirements.txt
    ```
    *(Ensure libraries like `pandas`, `numpy`, `scikit-learn`, `pytorch-tabnet`, `torch`, `xgboost`, `lightgbm`, `category_encoders`, `pyarrow`/`fastparquet`, `seaborn`, `matplotlib`, `joblib`, `tqdm` are included).*
4.  **Download Raw Data:** Obtain data from Kaggle (see **Data** section link) and place it in `data/original/`.
5.  **Run Data Preprocessing:** Execute notebook `notebooks/1_Data_Preprocessing.ipynb`. This is essential and may take significant time/memory. Ensure sufficient resources.

## Usage Instructions

1.  Follow **Setup Instructions** completely.
2.  Run the notebooks in the `notebooks/` folder, preferably in numerical order (0-4):
    *   Notebook 0 provides data context.
    *   Notebook 1 is required to generate processed data for notebooks 2, 3, and 4.
    *   Notebooks 2, 3, and 4 implement and evaluate the respective models.
3.  Notebooks in `notebooks/utils/` are for reference or supplementary analysis.

## Results Summary

*   **Cross-Validation Performance (Mean RMSPE):**
    *   TabNet + SVM: ~0.2298 (Best fold: ~0.2236)
    *   XGBoost + GridSearch: ~0.2836 (Best fold: ~0.2796)
    *   LightGBM: ~0.2350 (Best fold: ~0.2320)
*   **Best Performing Model:** Based on the average 5-fold cross-validation RMSPE, the **TabNet + SVM** ensemble yielded the lowest prediction error (~0.2298).
*   **Feature Importance Highlights:**
    *   *LightGBM (Gain):* Features related to realized volatility (`log_return_realized_volatility`) over different time windows were highly influential.
    *   *LightGBM (Split):* `stock_id` and `trade_seconds_in_bucket` appeared frequently in splits, though their contribution to gain was lower.
    *   *TabNet:* *(Report top features like `log_return2_realized_volatility_150_mean_stock` if applicable based on the generated plot).*
*   **Hyperparameter Tuning:** *(Briefly mention any key findings from GridSearchCV for XGBoost or insights from TabNet/LightGBM parameter choices, e.g., optimal learning rate, depth, regularization found).*

## Discussion & Conclusion

The analysis indicates that the **TabNet + SVM** ensemble demonstrated the most robust performance for predicting short-term realized volatility on this dataset, achieving the lowest average RMSPE. This likely stems from its ability to combine TabNet's capacity for capturing complex, non-linear feature interactions via its attention mechanism, with SVM's proficiency in handling regression tasks, potentially capturing simpler underlying relationships missed by TabNet alone.

LightGBM also performed competitively, outperforming the tuned XGBoost model in this instance. Its efficiency optimizations (GOSS, EFB) are valuable for this type of high-frequency data. Feature importance analyses suggest that lagged realized volatility metrics are highly predictive, aligning with financial theory (volatility clustering). The importance of `stock_id` also highlights idiosyncratic stock behavior.

XGBoost, despite hyperparameter tuning via GridSearch, lagged behind the other two primary methods based on the reported RMSPE, suggesting that either the chosen parameter grid wasn't optimal or that the other architectures were inherently better suited for this specific data structure and task.

**Potential Improvements & Future Work:**

1.  **Expanded Datasets:** Incorporating data from different market regimes or longer time periods could enhance model generalizability.
2.  **Advanced Feature Engineering:** Exploring more sophisticated microstructure features (e.g., order flow imbalance metrics, interaction terms, different volatility estimators) might yield further performance gains.
3.  **Refined Ensemble Methods:** Experimenting with different ensemble weighting schemes (instead of simple averaging) or stacking models could improve the TabNet+SVM results or combine predictions from all three models.
4.  **Deeper Hyperparameter Optimization:** Employing more advanced techniques like Bayesian optimization instead of GridSearch for XGBoost/LightGBM might uncover better parameter combinations.
5.  **Alternative Architectures:** Investigating time-series specific deep learning models like LSTMs or Transformers, adapted for tabular features, could be explored.

In conclusion, while all models provided predictive capability, the hybrid TabNet+SVM approach showed the most promise on this specific dataset and evaluation metric. Further refinements in feature engineering and ensemble techniques hold potential for pushing performance boundaries.

## References

*   **Data Source:** Optiver Realized Volatility Prediction - Kaggle Competition: [https://www.kaggle.com/competitions/optiver-realized-volatility-prediction](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction)
*   **LightGBM Paper:** Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *Advances in Neural Information Processing Systems (NIPS)*.
*   **TabNet Paper:** Arik, S. O., & Pfister, T. (2019). TabNet: Attentive Interpretable Tabular Learning. arXiv preprint arXiv:1908.07442.)*
