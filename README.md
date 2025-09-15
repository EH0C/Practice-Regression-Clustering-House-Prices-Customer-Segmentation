# Practice-Regression-Clustering-House-Prices-Customer-Segmentation

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