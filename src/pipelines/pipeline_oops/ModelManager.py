from sklearn.metrics import mean_squared_error
import numpy as np

from src.pipelines.pipeline_oops.DataManager import  DataManager
from src.pipelines.pipeline_oops.PipelineManager import PipelineManager


class ModelManager:
    """
    Class for managing the Models
    """
    def __init__(self, url:str):
        """
        Initialization
        """
        self.url=url
        self.data_manager=None
        self.pipeline_manager=None
        self.X=None
        self.y=None
        self.X_train=None
        self.X_test=None
        self.y_train=None
        self.y_test=None
        self.pipe_long=None
        self.y_pred=None
        self.mse=None
        self.rmse=None

    def manage_model(self):
        """
        Manage The Model
        Returns:
            y_pred,
            mse,
            rmse

        """
        print(self.url)
        #c='http://bit.ly/wine-quality-lwd'
        self.data_manager=DataManager(self.url)
        self.pipeline_manager=PipelineManager()

        # Split the data in train and test data
        self.X_train, self.X_test, self.y_train, self.y_test = self.data_manager.split_data_train_test()
        self.pipe_long = self.pipeline_manager.build_pipeline()

        # Apply all the transformation on the training set and train a knn model
        self.pipe_long.fit(self.X_train, self.y_train)

        # Apply all the transformation on the test set and make predictions
        self.y_pred = self.pipe_long.predict(self.X_test)
        # measure the performance
        # 1. Mean Squared Error(MSE)
        self.mse = mean_squared_error(self.y_test, self.y_pred)

        # 2. Root-Mean Squared Error(RMSE)
        self.rmse = np.sqrt(self.mse)

        print(self.y_pred)
        print(self.mse)
        print(self.rmse)

        return self.y_pred, self.mse, self.rmse




if __name__=="__main__":

    url=input()
    model_manager=ModelManager(url)
    y_pred, mse, rmse=model_manager.manage_model()