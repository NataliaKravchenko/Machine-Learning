# GBR regression
# Part 1 Preparing the data
# Import libraries
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import GradientBoostingRegressor

# Import of boston housing data set:
from sklearn.datasets import load_boston
boston = load_boston()
X=boston.data
y=boston.target

# Rebuild pandas DF
df = pd.DataFrame(X)
col_names=boston.feature_names
df['target_price'] = y

# Split test and train
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.33, random_state = 1)

# Create Regressor and train the model:
reg = GradientBoostingRegressor()
reg.fit(X_train, y_train)

# Predict the price using train data:
y_train_pred = reg.predict(X_train)
print('R^2:',metrics.r2_score(y_train, y_train_pred))
print('MAE:',metrics.mean_absolute_error(y_train, y_train_pred))
print('MSE:',metrics.mean_squared_error(y_train, y_train_pred))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_train, y_train_pred)))

# Part 2. Predict the price using test data:
y_test_pred = reg.predict(X_test)
print('R^2:', metrics.r2_score(y_test, y_test_pred))
print('MAE:',metrics.mean_absolute_error(y_test, y_test_pred))
print('MSE:',metrics.mean_squared_error(y_test, y_test_pred))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test, y_test_pred)))

# Save figures
#import os
#os.makedirs('project_plots', exist_ok=True)

# Import libraries
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(palette="inferno")

# Plot difference between actual and predicted values
sns.scatterplot(y_test, y_test_pred)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual price vs predicted')
plt.savefig('project_plots/gbr_act_vs_pred.png', dpi=300)
plt.close()

# Plot the residuals
residuals = y_test - y_test_pred
sns.scatterplot(y_test, residuals)
plt.xlabel('Predicted Price')
plt.ylabel('Residuals')
plt.title('Actual price vs residuals')
plt.savefig('project_plots/gbr_act_vs_resid.png', dpi=300)
plt.close()

