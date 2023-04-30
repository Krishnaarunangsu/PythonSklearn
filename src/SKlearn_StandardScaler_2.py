import pandas as pd
from sklearn.preprocessing import StandardScaler
import seaborn as sns

data = sns.load_dataset('iris')
print('Original Dataset')
data.head()

# Remove the Non-numeric Column
df=data.drop('species', axis=1)

std_scaler = StandardScaler()

df_scaled = std_scaler.fit_transform(df.to_numpy())
df_scaled = pd.DataFrame(df_scaled, columns=[
    'sepal_length', 'sepal_width', 'petal_length', 'petal_width'])

print("Scaled Dataset Using StandardScaler")
print(df_scaled.head())