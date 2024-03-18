import pandas as pd
from sklearn.preprocessing import LabelEncoder
# Instantiate the encoder
le = LabelEncoder()
# Fit the encoder and transform the data
# encoded_data = le.fit_transform(['Red', 'Blue', 'Green', 'Red', 'Green'])

# Create a DataFrame with categorical data
df = pd.DataFrame({'Color': ['Red', 'Blue', 'Green', 'Red', 'Green']})
# Instantiate the encoder
le = LabelEncoder()
# Fit the encoder and transform the data
df['Color'] = le.fit_transform(df['Color'] )

# Print the encoded data
print(df['Color'])


# ['setosa' 'versicolor' 'virginica']