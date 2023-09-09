##### Transformers
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler,MinMaxScaler

##### Estimators
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier

##### Utils
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np


# read the data and split it into a training and test set
url = "http://bit.ly/wine-quality-lwd"
wine = pd.read_csv(url)
print(wine.head())
X = wine.drop("quality", axis=1).copy()
y = wine["quality"].copy()
print(X)
print(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print(X_train)
print('*********************************************')

pipe_long = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",StandardScaler()),
    ("knn", KNeighborsRegressor())
])

for i in range(len(pipe_long.steps)):
    print(pipe_long.steps[i])
    print(pipe_long[i])
    print('*****************************')


# Apply all the transformation on the training set and train a knn model
pipe_long.fit(X_train, y_train)
# apply all the transformation on the test set and make predictions
y_pred = pipe_long.predict(X_test)
# measure the performance
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(rmse)
