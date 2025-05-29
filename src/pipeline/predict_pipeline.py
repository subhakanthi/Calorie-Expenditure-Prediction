import os
import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_obj


class PredictPipeline:
    def __init__(self):
        self.model_path = '/home/kanthi/Documents/calorie_expenditure/artifacts/model.pkl'
        self.preprocessor_path = '/home/kanthi/Documents/calorie_expenditure/artifacts/preprocessor.pkl'
    
    def predict(self, features):
        try:
            model = load_obj(self.model_path)
            label_encoder = load_obj(self.preprocessor_path)  # Your saved LabelEncoder

            # Transform only the 'Sex' column (or whatever categorical column you have)
            features['Sex'] = label_encoder.transform(features['Sex'])

            # Now predict on the transformed features DataFrame
            predictions = model.predict(features)

            return predictions

        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self,
                id: int,
                Sex: str, 
                Age: int,
                Height: int,
                Weight: int,
                Duration:int,
                Heart_Rate:int,
                Body_Temp: int):
        self.id = id
        self.Sex = Sex
        self.Age = Age
        self.Height = Height
        self.Weight = Weight
        self.Duration = Duration
        self.Heart_Rate = Heart_Rate
        self.Body_Temp = Body_Temp

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "id": [self.id],
                "Sex": [self.Sex],
                "Age": [self.Age],
                "Height": [self.Height],
                "Weight": [self.Weight],
                "Duration": [self.Duration],
                "Heart_Rate": [self.Heart_Rate],
                "Body_Temp": [self.Body_Temp]
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)



        
