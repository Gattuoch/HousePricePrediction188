# House Price Prediction AI/ML Project

This project involves building a machine learning model to predict house prices based on various features. The dataset used for this project is from  GBR Competition  "House Prices - Advanced Regression Techniques". The goal is to develop a model that accurately predicts house prices given a set of input features.

##  GBR Competition
- Dataset: [House Prices - Advanced Regression Techniques](https://github.com/Shreyas3108/house-price-prediction.git)
- Model Score: 87.16% (R-squared score)

## File Structure
- `house-price.ipynb` : Jupyter Notebook containing the code for data preprocessing, exploratory data analysis (EDA), feature engineering, model training, and prediction and many other.
- `dfcleaned.csv`: CSV file containing the predicted house prices for the test dataset.
- `model.pkl`: Pickle file containing the trained GradientBoostingRegressor model.

## Libraries Used
-Python version: 3.6.9
Matplotlib version: 3.1.3
Seaborn version: 0.9.0
Pandas version: 0.25.1
Numpy version: 1.16.5
Statsmodels version: 0.10.1
Scikit-learn version: 0.21.2
Folium version: 0.9.1
Geopandas version: 0.7.0
Geopy version: 1.21.0
Reverse_geocoder version: 1.5.1
Pickleshare version: 0.7.5
## Data Loading and Analysis
- The training and test datasets are loaded from CSV files.
- Exploratory data analysis is performed to understand the structure and characteristics of the data.
- Data visualization techniques such as histograms, box plots, and heatmaps are used to analyze the distribution of features and identify missing values.

## Data Preprocessing
- Missing values are handled using appropriate techniques such as imputation or dropping columns.
- Categorical variables are encoded using one-hot encoding.
- Numerical features are standardized to ensure uniformity and improve model performance.

## Model Selection and Training
- Several regression models are considered, including Linear Regression, Folium,  GridSearchCV, KNeighborsRegressor,Geopandas, Reverse_geocoder , and teLimiter.
- Cross-validation is used to evaluate each model's performance based on the R-squared score.
- The GradientBoostingRegressor model is selected based on its superior performance.

## Model Evaluation and Prediction
- The selected model is trained on the training dataset.
- The trained model is used to make predictions on the test dataset.
- The predictions are saved to a CSV file (`dfcleaned.csv`) for submission.


## Additional Notes
- The `dfcleaned.csv` file contains the predicted house prices for the test dataset.
- The trained model (`model.pkl`) is stored as a pickle file for future use or deployment.

## Final model details
# Model A

17 features
Adjusted R-squared of 0.701 (70% of variations explained by our model)
Uses zip code tiers instead of actual zipcodes
Better for generalising to other areas
RMSE of 132,444 (mean RMSE with 10-fold cross-validation)
# Model B

87 features
No interacting terms or polynomials
Adjusted R-squared of 0.832 (83% of variations explained by our model)
RMSE of 99,654 (mean RMSE with 10-fold cross-validation)
## Interpreting Model coefficients

A grade 12 house on average is worth USD 52,000 more than grade 11 (model A)
Being on the waterfront is valued at USD 277,442 (model A)
For every additional squarefoot of living space, the price inscreases by USD 123.5 (model B)