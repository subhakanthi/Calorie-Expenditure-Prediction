import sys
from dataclasses import dataclass
import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transormation_config= DataTransformationConfig()

    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info('Read Train and Test data')
            train_df.dropna(inplace=True)
            logging.info('Dropped missing values')

            label_encoder=LabelEncoder()
            train_df['Sex']=label_encoder.fit_transform(train_df['Sex'])
            test_df['Sex']=label_encoder.fit_transform(test_df['Sex'])

            target_column='Calories'
            X_train=train_df.drop(columns=[target_column],axis=1)
            y_train=train_df[target_column]

            X_test=test_df

            save_object(
                file_path=self.data_transormation_config.preprocessor_obj_file_path,
                obj=label_encoder
            )

            logging.info('Saved Label Encoder object successfully')

            return(
                X_train,
                y_train,
                X_test,
                self.data_transormation_config.preprocessor_obj_file_path
            )
        


        except Exception as e:
            raise CustomException(e,sys) from e




            
