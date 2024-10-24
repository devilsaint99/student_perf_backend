import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_obj

class PredictPipeline:
    def predict_math_score(self, features):
        try:
            model_path = 'artifacts\\model.pkl'
            preprocessor_path = 'artifacts\\preprocessor.pkl'
            logging.info("Loading preprocessor pickle")
            preprocessor = load_obj(preprocessor_path)
            logging.info("Standardizing input values")
            data_scaled = preprocessor.transform(features)

            logging.info("Loading model pickle")
            model = load_obj(model_path)
            logging.info("Predicting maths scores")
            maths_score_pred = model.predict(data_scaled)
            return int(maths_score_pred[0])
        except Exception as e:
            logging.error(CustomException(e, sys))
            CustomException(e, sys)

class CustomData:
    def __init__(self, raw_data):
        self.raw_data = raw_data

    def convert_to_df(self):
        try:
            logging.info("Converting json recieved from api to dataframe")
            data_df = pd.DataFrame(self.raw_data,index=[0])
            print(data_df)
            return data_df
        except Exception as e:
            logging.error(CustomException(e, sys))
            CustomException(e, sys)
