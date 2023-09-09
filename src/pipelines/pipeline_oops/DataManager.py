##### Utils
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

class DataManager:
    "Class to Load and Manage Data"
    def __init__(self,url:str):
        """Constructor"""
        self.url=url
        self.wine_data=None
        self.X=None
        self.y=None
        self.X_train=None
        self.X_test=None
        self.y_train=None
        self.y_test=None

    def load_data(self):
        """
        Method to load the data
        Returns:
            wine_data

        """
        print(self.url)
        # self.url="http://bit.ly/wine-quality-lwd"
        self.wine_data = pd.read_csv(self.url)
        print(self.wine_data.head())
        return self.wine_data

    def split_data_train_test(self):
        """
        Method to split the data in train and test dataset
        Returns:
            X_train
            X_test
            y_train
            y_test

        """
        wine_dataset = self.load_data()
        self.X = wine_dataset.drop("quality", axis=1).copy()
        self.y = wine_dataset["quality"].copy()
        # print(self.y)
        #
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=42
        )
        print(self.X_train)
        print(self.y_train)
        return self.X_train, self.X_test, self.y_train, self.y_test




if __name__=="__main__":
    url=str(input())
    data_manager=DataManager(url)
    data_manager.load_data()
    X_train,X_test,y_train,y_test= data_manager.split_data_train_test()
    # print(X_train)
