from sklearn.datasets import make_friedman2


X, y = make_friedman2(random_state=42)
print(X.shape)
print(y.shape)
print(list(y[:3]))