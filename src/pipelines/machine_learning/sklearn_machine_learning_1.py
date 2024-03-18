# The digits dataset consists of 8x8 pixel images of digits.
# The images attribute of the dataset stores 8x8 arrays of grayscale values for each image.
# We will use these arrays to visualize the first 4 images.
# The target attribute of the dataset stores the digit each image represents
# and this is included in the title of the 4 plots below.

# Note: if we were working from image files (e.g., ‘png’ files),
# we would load them using matplotlib.pyplot.imread.


from sklearn.datasets import load_digits
from sklearn import metrics, svm
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

digits=load_digits()

_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training: %i" % label)

# plt.show()

# Classification
# To apply a classifier on this data, we need to flatten the images,
# turning each 2-D array of grayscale values from shape (8, 8) into shape (64,).
# Subsequently, the entire dataset will be of shape (n_samples, n_features),
# where n_samples is the number of images and n_features is the total number of pixels in each image.
#
# We can then split the data into train and test subsets and
# fit a support vector classifier on the train samples.
# The fitted classifier can subsequently be used to
# predict the value of the digit for the samples in the test subset.

# flatten the images
# Get the number of samples: Rows of data
n_samples=len(digits.images)
print(f'No of samples:\n{n_samples}')
print(f'Shape of the dataset:\n{digits.data.shape}')
print(f'Digits Images type:\n{type(digits.images)}')
print(f'Digits Images Shape:\n{digits.images.shape}')
print(f'Digits Images data:\n{digits.images}')

# Reshape 2-D Image data from shape (8,8) into shape(64,)
image_Data=digits.images.reshape((n_samples, -1))
print(f'Modified Shape of the image data:{image_Data.shape}')

# Create a Classifier, a Support Vector Classifier
clf=svm.SVC(gamma=0.001) # What is gamma How ro set this with Grid Search and Cross Validation

# Split the data into 50% train and 50% test subsets
X_train, X_test, y_train, y_test=train_test_split(
    image_Data, digits.target, test_size=0.5, shuffle=False
)

# Learn the digits on the train subset
clf.fit(X_train, y_train)

# Predict the value of the digit on the test subset
predicted=clf.predict(X_test)

# clf.score()

# Below we visualize the first 4 test samples and show their predicted digit value in the title.

_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, prediction in zip(axes, X_test, predicted):
    ax.set_axis_off()
    print(f'Image Shape Current:{image.shape}')
    image = image.reshape(8, 8)
    print(f'Modified Image Shape Current:{image.shape}')
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title(f"Prediction: {prediction}")

plt.show()

# classification_report builds a text report showing the main classification metrics.

print(
    f"Classification report for classifier {clf}:\n"
    f"{metrics.classification_report(y_test, predicted)}\n"
)

# We can also plot a confusion matrix of the true digit values and the predicted digit values.
disp=metrics.ConfusionMatrixDisplay.from_predictions(y_test,predicted)
disp.figure_.suptitle('Confusion Matrix')
print(f'Confusion Matrix:\n{disp.confusion_matrix}')

plt.show()

# If the results from evaluating a classifier are stored in the form of a confusion matrix and not in terms of y_true and y_pred,
# one can still build a classification_report as follows:

# The ground truth and the predicted truth
y_true=[]
y_pred=[]
cm=disp.confusion_matrix
print(f'Length of the Confusion Matrix:\n{cm}')

for gt in range(len(cm)):
    for pred in range(len(cm)):
        y_true +=[gt]*cm[gt][pred]
        y_pred +=[pred]*cm[gt][pred]

print("Classification Report built from confusion matrix:\n"
      f"{metrics.classification_report(y_true, y_pred)}")
