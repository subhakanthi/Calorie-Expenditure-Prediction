import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainer
from src.components.model_trainer import ModelTrainerConfig

@dataclass
class DataIngestionConfig:
    train_data_path:str=os.path.join('artifacts','train.csv')
    test_data_path:str=os.path.join('artifacts','test.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Enter the data ingestion method or component")
        try:
            train_path=pd.read_csv('/home/kanthi/Documents/calorie_expenditure/data/train.csv')
            test_path=pd.read_csv('/home/kanthi/Documents/calorie_expenditure/data/test.csv')
            logging.info('Read the train and test data set')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            train_path.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_path.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info('Train and Test data saved to Artifacts')
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)
        

if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data = obj.initiate_data_ingestion()

    data_transformation=DataTransformation()
    X_train,y_train,X_test,_=data_transformation.initiate_data_transformation(train_data,test_data)

    X1_train, X1_val, y1_train, y1_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    logging.info("Train data split into training and validation sets")

    model_trainer=ModelTrainer()
    r2_train, r2_val = model_trainer.initiate_model_trainer(X1_train, y1_train, X1_val, y1_val)    


    
