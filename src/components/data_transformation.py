import sys
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from src.utils import split_io_data, get_feature_typewise, save_obj

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            input_data_df, _ = split_io_data(pd.read_csv('src\\notebook\\stud.csv'))
            num_features, cat_features = get_feature_typewise(input_data_df)
            num_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='median')),
                    ("scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='most_frequent')),
                    ("one_hot_encoder", OneHotEncoder(drop='first'))
                ]
            )

            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, num_features),
                    ('cat_pipeline', cat_pipeline, cat_features)
                ]
            )

            return preprocessor
        except Exception as e:
            logging.info(CustomException(e,sys))
            raise CustomException(e, sys)
    
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Reading of train and test data completed.")

            logging.info("Getting preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()
            input_feature_train_df, target_feature_train_df = split_io_data(train_df)
            input_feature_test_df, target_feature_test_df = split_io_data(test_df)

            logging.info("Applying preprocessing on train and test data")
            input_feature_train_df = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_df = preprocessing_obj.transform(input_feature_test_df)
            print(input_feature_test_df[0])
            
            save_obj(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                ml_obj = preprocessing_obj
            )

            return(
                input_feature_train_df, input_feature_test_df, target_feature_train_df, target_feature_test_df

            )
        except Exception as e:
            logging.info(CustomException(e, sys))
            raise CustomException(e, sys)