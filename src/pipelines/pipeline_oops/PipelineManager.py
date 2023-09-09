# Transformers
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Estimators
from sklearn.neighbors import KNeighborsRegressor

# Utils
from sklearn.pipeline import Pipeline

class PipelineManager:
    """
    Class for Managing the Pipeline
    """
    def __init__(self):
        """
        Initializtion
        """
        self.pipe_long=None

    def build_pipeline(self):
        """

        Returns:
            pipe_long

        """
        self.pipe_long=Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",StandardScaler()),
    ("knn", KNeighborsRegressor())])
        for i in range(len(self.pipe_long.steps)):
            print(self.pipe_long.steps[i])
            print(self.pipe_long[i])
            print('*****************************')
        return self.pipe_long


if __name__=="__main__":

    pipeline_manager=PipelineManager()
    pipeline_manager.build_pipeline()
