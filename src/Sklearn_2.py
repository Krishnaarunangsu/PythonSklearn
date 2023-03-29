# Fitting and predicting: estimator basics
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

class RandomForestModelBasic:
    """
    Random Forest Modelling
    """
    prediction_classifier: object

    def __init__(self):
        """
        Initialization
        :return:
        """
        self.x=None
        self.y=None
        self.rf_classfier=RandomForestClassifier(random_state=0)
        self.prediction_classifier=None
        self.prediction=None

    def predict(self,data):
        """
        Predict the Classifier
        :return: prediction_classifier
        """
        # 2 samples, 3 features
        self.x=[[1,2, 3],
            [11, 12, 13]]
        self.y=[0,1]
        self.prediction_classifier = self.rf_classfier.fit(self.x, self.y)
        #Prediction
        print(f'Classes of Training Data:{self.rf_classfier.predict(self.x)}')
        self.prediction=self.rf_classfier.predict(data)
        return self.prediction_classifier,self.prediction


class StandardScalerBasic:
    """
    Standardscaler
    """
    def __init__(self):
        """
        Initialization
        """
        self.x=None
        self.transformed_data=None
    def transform(self):
        """
        Data Transformation

        :return:
        """
        self.x=[[0, 15],
                [1, -10]]
        self.transformed_data=StandardScaler().fit(self.x).transform(self.x)
        print(f'Transformed Data:{self.transformed_data}')



if __name__=="__main__":
    rf_prediction=RandomForestModelBasic()
    # 2 samples, 3 features
    x_data=[[4, 5, 6], [14, 15, 16]]
    print(f'The Classes of new data are:{rf_prediction.predict(x_data)}')

    sklearn_transformer=StandardScalerBasic()
    print(f'The transformer is:{sklearn_transformer.transform()}')