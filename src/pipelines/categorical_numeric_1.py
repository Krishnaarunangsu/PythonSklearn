# Method 1: Using the cat.codes Attribute
# The easiest way to convert categorical data to numerical data in Pandas is to use the cat.codes attribute. This attribute is available for categorical data types in Pandas and returns a numerical representation of each category.

# Here is an example of how to use the cat.codes attribute to convert categorical data to numerical data:

import pandas as pd

# Create a DataFrame with categorical data
df = pd.DataFrame({'Color': ['Red', 'Blue', 'Green', 'Red', 'Green']})

# Convert categorical data to numerical data using cat.codes
df['Color'] = df['Color'].astype('category')
df['Color_Codes'] = df['Color'].cat.codes

# View the converted DataFrame
print(df)