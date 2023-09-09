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
# prepare the data for machine learning
# fill missing values with medians
imputer = SimpleImputer(strategy="median")
X_train_tr = imputer.fit_transform(X_train)
print(X_train_tr)
print('*********************************************')
# scale the data
scale = StandardScaler()
X_train_tr = scale.fit_transform(X_train_tr)
print(X_train_tr)
print('*********************************************')
# do the same for test data. But here we will not apply the
# fit method only the transform method because we
# do not want our model to learn anything from the test data
X_test_tr = imputer.transform(X_test)
print(X_test_tr)
print('*********************************************')
X_test_tr = scale.transform(X_test_tr)
print(X_test_tr)

# initiate the k-nearest neighbors regressor class
knn = KNeighborsRegressor()
# train the knn model on training data
knn.fit(X_train_tr, y_train)
# make predictions on test data
y_pred = knn.predict(X_test_tr)
# measure the performance of the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(rmse)


