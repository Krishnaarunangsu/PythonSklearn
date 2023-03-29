# With random_state
import pandas as pd
from sklearn.model_selection import train_test_split
test_series = pd.Series(range(100))
size30split = train_test_split(test_series,random_state = 1,test_size = .3)
size25split = train_test_split(test_series,random_state = 1,test_size = .25)
common = [element for element in size25split[0] if element in size30split[0]]
print(len(common))