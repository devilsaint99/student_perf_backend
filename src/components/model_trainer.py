from src.logger import logging
from src.exception import CustomException
import pandas as pd
import numpy as np
import os, sys
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from src.utils import model_evaluater, save_obj

from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, X_train, X_test, y_train, y_test):
        try:
            models = {
            "LinearRegression": LinearRegression(),
            "Ridge": Ridge(),
            "Lasso": Lasso(),
            "DecisionTreeRegressor": DecisionTreeRegressor(),
            "RandomForestRegressor": RandomForestRegressor()
            }

            params = {
            "LinearRegression": {},
            "Ridge": {
                'solver':['sag', 'saga']
            },
            "Lasso": {
                'selection':['cyclic', 'random']
            },
            "DecisionTreeRegressor": {
                'criterion':['squared_error', 'friedman_mse'],
                'max_depth':[2, 4, 6, 10, None]
            },
            "RandomForestRegressor": {
                'criterion':['squared_error', 'friedman_mse'],
                'max_depth':[2, 4, 6, 10, None]
            }
            }
        
            model_evaluate = model_evaluater(X_train, X_test, y_train, y_test, models, params)

            best_model_score = max(list(model_evaluate.values()))
            best_model_name = list(model_evaluate.keys())[list(model_evaluate.values()).index(best_model_score)]
            best_model = models[best_model_name]
            logging.info(f'Best performer is {best_model_name}. With these params {best_model.get_params()}')
            if best_model_score<0.6:
                logging.error("Your models aren't performing well!!!")
                raise Exception("Your models aren't performing well!!!")
            save_obj(self.model_trainer_config.trained_model_file_path, best_model)
        except Exception as e:
            logging.error(CustomException(e, sys))
            raise CustomException(e, sys)


