# Practice-Regression-House-Prices

dataset used:
https://www.kaggle.com/datasets/yasserh/housing-prices-dataset?resource=download

🔎 Why scale features with StandardScaler?

Linear Regression coefficients depend on feature scale

Without scaling, features measured in larger units (e.g. area in square feet) can dominate features in smaller units (e.g. bathrooms count).

Scaling makes all features comparable, so regression weights reflect true importance rather than unit size.

Numerical stability

Linear Regression (especially when you check VIF for multicollinearity) involves matrix operations.

If feature values vary a lot in scale, the math can become unstable → large coefficient swings or warnings about ill-conditioned matrices.

Interpretability of VIF & coefficients

VIF measures correlations among predictors. Without scaling, a feature with huge magnitude can distort results.

Scaling makes VIF more meaningful and coefficients easier to compare.

General good practice in ML

Many algorithms (like KNN, SVM, PCA, Logistic Regression) need features on the same scale.

Even though plain Linear Regression can run without scaling, doing it usually improves stability and interpretability.  

📌 Rule of thumb for VIF (Variance Inflation Factor) interpretation:

VIF ≈ 1 → no multicollinearity.

VIF 1–5 → moderate correlation, usually okay.

VIF > 10 → serious multicollinearity, consider dropping/merging features.

🔎 Breakdown:

X → should be a DataFrame with only numeric features (after encoding/scaling).

variance_inflation_factor(X.values, i)

For each feature 𝑖, it runs an auxiliary regression: regress feature 𝑖 on all the other features.

📊 Model Performance

MSE = 1,754,318,687,330.67

Mean Squared Error: the average squared difference between predicted and actual house prices.

It’s in the squared units of your target (so here it looks big, but that’s normal for price data).

RMSE = 1,324,506.96

Root Mean Squared Error: easier to interpret because it’s in the same units as house price.

On average, your predictions are off by about 1.3 million.

If house prices in your dataset are in the range of millions, that might be acceptable; if they’re lower, the model is too rough.

R² = 0.6529

Coefficient of Determination: about 65% of the variance in house prices is explained by your features.

Not bad for a simple Linear Regression model 👌.

Insights:

Linear Regression actually outperforms Random Forest here — which is a bit unusual, since Random Forest often handles nonlinearities and interactions better.

Your R² is higher (0.65 vs 0.61).

Your errors (MSE, RMSE) are lower.

Possible reasons Random Forest didn’t shine:

Dataset might be relatively small, and Random Forest could be overfitting.

Features could have mostly linear relationships with the target.

Random Forest hyperparameters might need tuning (e.g., n_estimators, max_depth, min_samples_split).