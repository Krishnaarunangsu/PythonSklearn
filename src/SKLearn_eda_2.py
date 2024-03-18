# IMPORT THE PANDAS LIBRARY
# TO USE THE DATAFRAME TOOL
import pandas as pd
import numpy as np
import seaborn as sns

# IMPORT THE IRIS DATA FROM THE
# SKLEARN MODULE
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris

# LOAD THE IRIS DATASET BY CALLING
# THE FUNCTION
iris_data=load_iris()
# print(f'Data:\n{iris_data.data}')
# print(f'Features:\n{iris_data.feature_names}')
# print(f'Target:\n{iris_data.target}')
# print(f'Target Names:\n{iris_data.target_names}')

# PLACE THE IRIS DATA IN A PANDAS
# DATAFRAME
df_iris=pd.DataFrame(data= np.c_[iris_data['data'], iris_data['target']],
                  columns= iris_data['feature_names'] + ['Species'])

# DISPLAY FIRST 5 RECORDS OF THE
# DATAFRAME
print(df_iris.head())
print(df_iris.info())

#STATISTICS ABOUT DATASET
print(df_iris.describe())


