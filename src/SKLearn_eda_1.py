# IMPORT THE PANDAS LIBRARY TO USE THE DATAFRAME TOOL
import pandas as pd
import numpy as np

import seaborn as sns
from matplotlib import pyplot as plt

# IMPORT THE IRIS DATA FROM THE SKLEARN MODULE
from sklearn.datasets import load_iris
class ScikitLearnEDA():
    """

    """
    def __init__(self):
        """
        Initialization
        """
        # LOAD THE IRIS DATASET BY CALLING THE FUNCTION
        self.iris_data=load_iris()
        self.df_iris=None
        print('******************************IRIS DATA********************')
        print(self.iris_data)

    def get_iris_data_details(self):
        """

        Returns: Features and Target

        """
        print('************************************************************************')
        print(f'Features:\n{self.iris_data.feature_names}')
        print('#################################################################################')
        print(f'Data-Feature Values:\n{self.iris_data.data}')
        print('************************************************************************')
        print(f'Target Names:\n{self.iris_data.target_names}')
        # print(f'Target Names:\n{self.iris_data["target_names"]}')
        print('#################################################################################')
        print(f'Target:\n{self.iris_data.target}')

        return self.iris_data.data, self.iris_data.target

    def convert_data_dataframe(self):
        """

        Returns: iris data as dataframe

        """
        # PLACE THE IRIS DATA IN A PANDAS
        # DATAFRAME
        self.df_iris = pd.DataFrame(data=np.c_[self.iris_data.data, self.iris_data.target],
                               columns=self.iris_data.feature_names + ['Species'])

        # DISPLAY FIRST 5 RECORDS OF THE DATAFRAME
        print(self.df_iris.head())
        print(self.df_iris.info())
        print(self.df_iris.nunique())

        # STATISTICS ABOUT DATASET
        print(self.df_iris.describe())

        # Checking the null values
        print(self.df_iris.isnull().sum())

        return self.df_iris

    def perform_univariate_analysis(self):
        """

        Returns:

        """
        self.df_iris=self.convert_data_dataframe()
        # univariate analysis
        self.df_iris.groupby('Species').agg([np.mean, np.median])

    def display_box_plot(self,x:int, y:int):
        """

        Args:
            x: int
            y: int

        Returns:

        """
        # Box Plot
        # plt.figure(figsize=(x, y))
        plt.figure(figsize=(x, y))
        self.df_iris = self.convert_data_dataframe()
        sns.boxplot(x='Species', y='sepal width (cm)', data=self.df_iris, palette='YlGnBu')
        plt.show()


if __name__=="__main__":
    scikitlearn_eda=ScikitLearnEDA()
    scikitlearn_eda.get_iris_data_details()
    scikitlearn_eda.convert_data_dataframe()
    scikitlearn_eda.display_box_plot(8, 4)









