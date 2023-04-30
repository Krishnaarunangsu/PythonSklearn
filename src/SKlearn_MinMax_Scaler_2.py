#Using pandas and numpy
import seaborn as sns
import pandas as pd
import numpy as np

data=sns.load_dataset('iris')
print('Original Dataset')
print(data.head())

#Min-max Normalization
# Drop the target column
df=data.drop('species', axis=1)
print(df.head())

df_norm=(df-df.min())/(df.max()-df.min())
df_norm=pd.concat((df_norm, data.species), axis=1)
print("Scaled Dataset Using Pandas")
print(df_norm.head())