# Linear regression
#Part 1 Preparing the data
# Import libraries
import pandas as pd
import numpy as np

# Import of boston housing data set:
from sklearn.datasets import load_boston
boston = load_boston()
X=boston.data
y=boston.target


# Rebuild pandas DF
df = pd.DataFrame(X)
col_names=boston.feature_names
df['target_price'] = y

# Check for missing values
null_values=df.isnull().sum(axis=0)

# Split test and train
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.33, random_state = 1)

# Initialize regressor and train the model
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train, y_train)

# Linear regressin output
print(f"Intercept: {lm.intercept_}")
print(f"Named Coeficients: {pd.DataFrame(lm.coef_, col_names)}")
# The coefficient represents the mean change in the response value per one unit change in the predictor.
# When a coefficient is -0.099, the price decreases by -0.099 for every one unit of the predictor.


# Part 2. Predict the price using test data
y_predicted = lm.predict(X_test)

# Import libraries
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(palette="inferno")

# Save figures
import os
os.makedirs('project_plots', exist_ok=True)

# Plot difference between actual and predicted values
sns.scatterplot(y_test, y_predicted)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual price vs predicted')
plt.savefig('project_plots/lin_act_vs_pred.png', dpi=300)
plt.close()

# there is s a strong correlation between the predictions and actual values

# Plot the residuals
residuals = y_test - y_predicted
sns.scatterplot(y_test, residuals)
plt.xlabel('Predicted Price')
plt.ylabel('Residuals')
plt.title('Actual price vs residuals')
plt.savefig('project_plots/lin_act_vs_resid.png', dpi=300)
plt.close()

# y-axes displays the accuracy of the prediction.
# #The larger is the distance from 0 line, the more inaccurate the prediction was.
# Since residual = Actual – Predicted
#positive y-values for the residual mean the prediction was too low
# and negative y-values mean the prediction was too high

# Distribution of residuals
sns.distplot(residuals, bins=20, kde=False)
plt.title('Residuals Distribution')
plt.savefig('project_plots/lin_resid_distr.png', dpi=300)
plt.close()
#Linear regression requires normally distributed error terms, which is our case

# Evaluate model and calculate errors
from sklearn import metrics
print(f"MAE error(avg abs residual): {metrics.mean_absolute_error(y_test, y_predicted)}")
print(f"MSE error: {metrics.mean_squared_error(y_test, y_predicted)}")
print(f"RMSE error: {np.sqrt(metrics.mean_squared_error(y_test, y_predicted))}")
print(f"R2 Score: {metrics.r2_score(y_test, y_predicted)}")
