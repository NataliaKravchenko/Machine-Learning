# Test the efficiency of different linear models

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.datasets import load_boston

boston = load_boston()
col_names = boston.feature_names
y = boston.target
X = boston.data

# Split the data set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

# Train models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Lasso

# Print coefficients into a DataFrame
RMSE=[]
MAE=[]
R2=[]
models=[LinearRegression, GradientBoostingRegressor, ElasticNet, KNeighborsRegressor, Lasso]
for model in models:
    model = model()
    model.fit(X_train, y_train)
    y_predicted = model.predict(X_test)
    RMSE.append(np.sqrt(metrics.mean_squared_error(y_test, y_predicted)))
    MAE.append(metrics.mean_absolute_error(y_test, y_predicted))
    R2.append(metrics.r2_score(y_test, y_predicted))

statistics=np.array([RMSE,MAE,R2])
transposed=np.transpose(statistics)
results_df=pd.DataFrame(transposed)
results_df.columns = ['RMSE', 'MAE', 'R2']
results_df.index = ['LR', 'GBR', 'EN', 'KNR', 'L']
print(results_df)
print(f"Min RMSE: {results_df['RMSE'].min()}")
print(f"Min MAE: {results_df['MAE'].min()}")
print(f"Max R2: {results_df['R2'].max()}")
#or
print(results_df[['RMSE']].idxmin())
print(results_df[['MAE']].idxmin())
print(results_df[['R2']].idxmax())


