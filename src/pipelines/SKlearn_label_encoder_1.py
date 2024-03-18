from sklearn.preprocessing import LabelEncoder
# Instantiate the encoder
le = LabelEncoder()
# Fit the encoder and transform the data
# encoded_data = le.fit_transform(['Red', 'Blue', 'Green', 'Red', 'Green'])

encoded_data = le.fit_transform(['setosa', 'versicolor', 'virginica'])

# Print the encoded data
print(encoded_data)


# ['setosa' 'versicolor' 'virginica']