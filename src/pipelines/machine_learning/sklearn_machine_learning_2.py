from sklearn.datasets import make_friedman1


X, y = make_friedman1(random_state=42)
print(X.shape)
print(y.shape)
print(list(y[:5]))