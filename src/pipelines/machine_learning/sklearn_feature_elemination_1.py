from sklearn.datasets import load_digits
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from matplotlib import pyplot as plt

# load digits dataset
digits=load_digits()
X=digits.images.reshape((len(digits.images), -1))
y=digits.target

# Create the RFE object and rank each pixel
svc=SVC(kernel="linear", C=1)
rfe=RFE(estimator=svc, n_features_to_select=1, step=1)
rfe.fit(X, y)
ranking=rfe.ranking_.reshape(digits.images[0].shape)

# Plot the pixel ranking
plt.matshow(ranking, cmap=plt.cm.Blues)
plt.title("Ranking of pixels with RFE")
plt.show()

