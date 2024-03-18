from sklearn.datasets import load_digits
from sklearn import metrics, svm
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


digits=load_digits()
digit_data=digits.data
print(f'Digits Dataset Data:\n{digit_data}')
print(f'Digits  Data Shape:\n{digit_data.shape}')
print(f'Digits Dataset Features:\n{digits.feature_names}')
print(f'Total Digits Dataset Features:\n{len(digits.feature_names)}')
print(f'Digits Dataset Target Names:\n{digits.target_names}')
print(f'Digits Dataset Target:\n{digits.target}')

# Show Plot
plt.gray()
plt.matshow(digits.images[0])
plt.show()