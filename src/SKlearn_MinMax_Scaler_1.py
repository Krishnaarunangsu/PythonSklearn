from sklearn.preprocessing import MinMaxScaler
data_for_scaling=[[-1,2],[-0.5,6],[0,10],[1,18]]
data_for_scaling_2=[[-1,2,0],[-0.5,6,8],[0,10,5],[1,18,-1]]
#Instantiate a MinMax scalar object
scaler_min_max=MinMaxScaler()

# Fit the data: Compute the minimum and maximum to be used for later scaling.
print(scaler_min_max.fit(data_for_scaling))
print(scaler_min_max.get_feature_names_out())
print(scaler_min_max.get_params())
# Transform the data : Scale features of X according to feature_range
print(scaler_min_max.transform(data_for_scaling))
# Fit and Transform the data: Fit to data, then transform it.
print(scaler_min_max.fit_transform(data_for_scaling))
print(scaler_min_max.fit(data_for_scaling_2))
print(scaler_min_max.get_feature_names_out())
print(scaler_min_max.get_params())
