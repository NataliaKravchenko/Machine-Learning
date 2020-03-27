# Logistic Regression
#Import packages
import pandas as pd
from sklearn.datasets import load_boston
boston = load_boston()

D=boston.data
T=boston.target

# Rebuild pandas DF
col_names= boston.feature_names
df = pd.DataFrame(D,columns=col_names)
df["Price"]=T
print(df.head())

# Split train and test:
y = df['CHAS']
X = df.drop('CHAS',axis=1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35)

# Initialize regressor and train the model
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train)

# Output of the training:
print(f"Intercept per class: {lr.intercept_}\n")
print(f"Coeficients per class: {lr.coef_}\n")

print(f"Available classes: {lr.classes_}\n")
print(f"Named Coeficients for class 1: {pd.DataFrame(lr.coef_[0], col_names)}\n")
print(f"Named Coeficients for class 2: {pd.DataFrame(lr.coef_[-1], col_names)}\n")

print(f"Number of iterations generating model: {lr.n_iter_}")

# Predict results for test dataset
predicted_values = lr.predict(X_test)

# Evaluation
# Residuals
for (real, predicted) in list(zip(y_test, predicted_values)):
    print(f'Value: {real}, pred: {predicted} {"is different" if real != predicted else ""}')

# Print accuracy score(mean accuracy)
print(f'Accuracy score is {lr.score(X_test, y_test):.2f}/1 \n')

# Print classification report
from sklearn.metrics import classification_report, confusion_matrix, f1_score
print(f"Classification Report: {classification_report(y_test, predicted_values)}\n")

# Print confusion matrix
print(f"Confusion Matrix: {confusion_matrix(y_test, predicted_values)}\n")

# Print f1-score
print(f"Overall f1-score: {f1_score(y_test, predicted_values, average='macro')}\n")