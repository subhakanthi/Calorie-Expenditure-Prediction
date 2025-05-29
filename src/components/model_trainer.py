import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig:
    model_path= os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,X1_train, y1_train, X1_val, y1_val):
        try:

                model=CatBoostRegressor(verbose=0)
                model.fit(X1_train,y1_train)
                logging.info('Model training completed')

                train_preds = model.predict(X1_train)
                r2_train = evaluate_model(y1_train, train_preds)
                print(f"R^2 Score on Training data: {r2_train:.4f}")

                val_preds = model.predict(X1_val)
                r2_val = evaluate_model(y1_val, val_preds)
                print(f"R^2 Score on Validation data: {r2_val:.4f}")


                save_object(
                    file_path=self.model_trainer_config.model_path,
                    obj=model
                )

                logging.info('Training Completed')
                return r2_train, r2_val
        
        except Exception as e:
             raise CustomException(e,sys) from e
        


