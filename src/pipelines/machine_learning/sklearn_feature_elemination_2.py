from sklearn.datasets import make_friedman1
from sklearn.feature_selection import RFE
from sklearn.svm import SVR

# Friedman1 Regression dataset
X, y = make_friedman1(n_samples=50, n_features=10, random_state=0)

# Support Vector Regression Estimator
estimator = SVR(kernel="linear")

# Feature Selection for elemination
selector = RFE(estimator, n_features_to_select=5, step=1)
selector = selector.fit(X, y)
print(f'Selector Support:\n{selector.support_}')
print(f'Selector Ranking:\n{selector.ranking_}')

# https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html
# https://www.analyticsvidhya.com/blog/2023/05/recursive-feature-elimination/