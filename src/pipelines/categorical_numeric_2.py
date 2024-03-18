# Method 1: Using the cat.codes Attribute
# TMethod 2: Using the replace() Method
# Another way to convert categorical data to numerical data in Pandas is to use the replace() method. This method replaces each category with a specified numerical value.
# Here is an example of how to use the replace() method to convert categorical data to numerical data:
import pandas as pd

# Create a DataFrame with categorical data
df = pd.DataFrame({'Color': ['Red', 'Blue', 'Green', 'Red', 'Green']})

# Convert categorical data to numerical data using replace
df['Color'] = df['Color'].replace({'Red': 0, 'Blue': 1, 'Green': 2})

# View the converted DataFrame
print(df)