from sklearn.preprocessing import StandardScaler
data=[[0,0],[0,0],[1,1],[1,1]]

# Instantiate Standard Scaler
scaler = StandardScaler()
print(scaler.fit(data))
StandardScaler()
print(scaler.mean_)
print(scaler.var_)
print(scaler.scale_)
print()
print(scaler.transform(data))
scaled_data=scaler.fit_transform(data)
print(scaled_data.mean(axis=0))
print(scaled_data.std(axis=0))
# print(scaler.transform([[2, 2]]))
# print(scaler.fit_transform([[2, 2]]))