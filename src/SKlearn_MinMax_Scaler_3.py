#Using pandas and numpy
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

data=sns.load_dataset('iris')
print('Original Dataset')
print(data.head())

df=data.drop('species', axis=1)
print(df.head())

def perform_min_max_scaling(df_original,scaler):
    """

    Args:
        df: Dataframe
        scaler: MinMaxScaler

    Returns:

    """
    df_scaled = scaler.fit_transform(df_original)
    print(df_scaled)
    # convert the numpy array `to dataframe
    df_scaled = pd.DataFrame(df_scaled, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_Width'])
    return df_scaled
# MinMAx Scaler
scaler_min_max=MinMaxScaler()
df_scaled_min_max=perform_min_max_scaling(df, scaler_min_max)
print('Scaled Dataset using MinMaxScaler')
print(df_scaled_min_max)


