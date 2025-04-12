# Apartment Rental Price Prediction

## Overview

This project focuses on predicting apartment rental prices across the continental United States using a dataset sourced from Kaggle. The primary goal is to apply and evaluate various regression models to determine the most effective approach for this prediction task. The project demonstrates a comprehensive data science workflow, from data acquisition and cleaning to feature engineering, model training, hyperparameter tuning, and results analysis.

**Dataset:** [Apartment Rent Data on Kaggle](https://www.kaggle.com/datasets/shashanks1202/apartment-rent-data/data)

## Key Data Science Concepts Demonstrated

*   **Data Acquisition & Exploration:** Loading data, initial inspection (`info`, `describe`, `head`), understanding data structure and types.
*   **Data Cleaning & Preprocessing:**
    *   Handling Missing Values (NaNs): Dropping columns with high NaN percentages, imputing missing `bathrooms` and `bedrooms` based on `square_feet` averages from similar listings.
    *   Duplicate Removal.
    *   Outlier Detection & Handling: Identifying and removing outliers in `price` and `square_feet` based on the 99th percentile.
*   **Feature Engineering:**
    *   Dropping irrelevant or redundant features (e.g., `ID`, `title`, `body`, `currency`).
    *   Encoding Categorical Features:
        *   One-Hot Encoding (`has_photo`).
        *   Target Encoding (`cityname`, `state`) based on the target variable (`price`) to handle high cardinality.
*   **Feature Selection:**
    *   Initial pruning based on relevance and variance (dropping `fee`, `price_type`).
    *   Quantitative selection using statistical methods (Gain Ratio, Pearson Correlation, F-value) to identify the most predictive features for the regression task. The top 5 features selected for modeling were: `bedrooms`, `bathrooms`, `square_feet`, `cityname_encoded`, `state_encoded`.
*   **Data Scaling:** Applying `StandardScaler` to numerical features before model training (although only the selected top 5 features were ultimately used in the models).
*   **Regression Modeling:** Implementing and comparing multiple regression algorithms:
    *   Polynomial Regression (using PyTorch with CUDA acceleration due to scikit-learn performance limitations on higher degrees).
    *   Ridge Regression (L2 Regularization).
    *   Elastic Net Regression (L1 & L2 Regularization).
    *   Random Forest Regressor (Ensemble Method).
    *   XGBoost Regressor (Gradient Boosting - Bonus).
*   **Hyperparameter Tuning:**
    *   Manual iteration for Polynomial Regression degree.
    *   Manual iteration for Ridge Regression `alpha`.
    *   Using `GridSearchCV` for optimizing Elastic Net, Random Forest, and XGBoost parameters.
*   **Model Evaluation:** Assessing model performance using standard regression metrics: Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and Root Mean Squared Percentage Error (RMSPE) on both training and testing sets.
*   **Results Analysis:** Comparing model performance, discussing overfitting vs. generalization, and explaining why certain models performed better based on their characteristics and the nature of the data (e.g., handling non-linearity).

## Workflow / Methodology

1.  **Data Acquisition:** The dataset containing ~100k apartment listings from the continental US was loaded from a CSV file.
2.  **Data Cleaning:**
    *   Initial assessment revealed significant missing values in `amenities`, `pets_allowed`, and `address`; these columns were dropped.
    *   Missing `bathrooms` and `bedrooms` (less than 1%) were imputed using a custom function that averaged values from listings with similar square footage (±300 sqft).
    *   Duplicate rows were removed.
    *   Remaining rows with any NaN values (a very small fraction) were dropped.
    *   Outliers in `price` and `square_feet` exceeding the 99th percentile were removed to create a more robust dataset for modeling.
3.  **Feature Engineering & Selection:**
    *   Columns deemed irrelevant for price prediction (like `ID`, `title`, `body`, `currency`, `source`, `time`) were dropped.
    *   Low-variance categorical features (`fee`, `price_type`) were identified and removed.
    *   `has_photo` was one-hot encoded.
    *   High-cardinality features `cityname` and `state` were target encoded using the average `price` for each category to convert them into meaningful numerical features.
    *   Statistical tests (Gain Ratio, Correlation, F-value) confirmed the importance of `cityname_encoded`, `state_encoded`, `square_feet`, `bathrooms`, and `bedrooms` relative to the `price`. These top 5 features were selected for model training. The analysis also indicated non-linear relationships between some features and the target.
4.  **Data Scaling:** `StandardScaler` was applied to the selected numerical features to standardize their range.
5.  **Model Training & Evaluation:**
    *   The cleaned, processed data was split into training (80%) and testing (20%) sets.
    *   Five different regression models were trained:
        *   **Polynomial Regression:** Implemented using PyTorch and CUDA for degrees 1-9 to find the optimal complexity. Degree 5 yielded the best results.
        *   **Ridge Regression:** Tested with various `alpha` values for L2 regularization.
        *   **Elastic Net:** `GridSearchCV` was used to find the best `alpha` and `l1_ratio` for combined L1/L2 regularization.
        *   **Random Forest:** `GridSearchCV` identified the optimal hyperparameters (estimators, depth, features, etc.).
        *   **XGBoost:** `GridSearchCV` determined the best hyperparameters for this gradient boosting model.
    *   Each model's performance was evaluated using MAE, MSE, RMSE, and RMSPE on both training and test sets.

## Results

The performance of the models varied, highlighting the importance of choosing appropriate algorithms for the data's characteristics:

| Model                 | Best Test MAE | Best Test MSE | Best Test RMSE | Best Test RMSPE (%) | Notes                                                                 |
| :-------------------- | :------------ | :------------ | :------------- | :------------------ | :-------------------------------------------------------------------- |
| Polynomial (Deg 5)    | 248.87        | 119266.76     | 345.35         | 25.11               | Good at capturing non-linearity, but sensitive. PyTorch/CUDA used.    |
| Ridge                 | 264.23        | 132811.06     | 364.43         | 27.33               | Linear model with L2; performance stable across alphas.               |
| Elastic Net           | 264.23        | 132811.00     | 364.43         | 27.33               | Linear model with L1/L2; similar to Ridge, L1 had little impact.      |
| **Random Forest**     | **180.06**    | **76564.71**  | **276.70**     | **20.17**           | **Best test performance**, handles non-linearity well. Showed overfitting. |
| XGBoost (Bonus)       | 223.07        | 98095.17      | 313.20         | 22.60               | Strong performance, better generalization than Random Forest.           |

*   **Random Forest** achieved the lowest error metrics on the test set (e.g., ~20% RMSPE), indicating the best predictive accuracy for unseen data among the tested models. Its ensemble nature likely helped handle the non-linearities and feature interactions effectively. However, it exhibited the largest gap between training and test error, suggesting some overfitting.
*   **XGBoost** also performed very well, with slightly higher test error than Random Forest but significantly less overfitting (smaller train-test error gap), indicating better generalization.
*   **Polynomial Regression** (degree 5) captured some non-linearity but was outperformed by the tree-based ensembles.
*   **Ridge and Elastic Net**, being linear models, struggled the most with the inherent non-linear relationships in the dataset, resulting in the highest errors. The similarity in their results suggests L1 regularization (present in Elastic Net) offered minimal benefit over L2 alone for this feature set.

## Technologies Used

*   Python 3
*   Pandas
*   NumPy
*   Scikit-learn (for preprocessing, modeling, evaluation, hyperparameter tuning)
*   PyTorch (for Polynomial Regression implementation with CUDA)
*   Matplotlib (for plotting)
*   Category Encoders (for Target Encoding)
*   XGBoost

## Conclusion

This project successfully demonstrates a full data science pipeline for predicting apartment rental prices. It highlights the importance of thorough data cleaning, appropriate feature engineering (especially target encoding for high-cardinality categoricals), and feature selection. The comparison of multiple regression models clearly shows the superiority of ensemble tree methods (Random Forest, XGBoost) over linear models and standard Polynomial Regression for this specific dataset, likely due to their ability to capture complex, non-linear relationships between features like square footage, location, and price. Random Forest provided the best predictive accuracy on the test set, while XGBoost offered a strong balance between accuracy and generalization.
