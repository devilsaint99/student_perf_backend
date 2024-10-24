from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
import sys
import os
import pandas as pd
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts', 'train.csv')
    test_data_path: str=os.path.join('artifacts', 'test.csv')
    raw_data_path: str=os.path.join('artifacts', 'raw.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion started")
        try:
            df = pd.read_csv('src\\notebook\\stud.csv')
            logging.info("Data read completed")
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train split inititaed")
            train_set, test_set = train_test_split(df, test_size=0.25, random_state=2)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of data completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)

if __name__=='__main__':
    data_ingestion_obj = DataIngestion()
    train_path, test_path = data_ingestion_obj.initiate_data_ingestion()
    data_transformation_obj = DataTransformation()
    X_train, X_test, y_train, y_test = data_transformation_obj.initiate_data_transformation(train_path, test_path)
    model_trainer = ModelTrainer()
    model_trainer.initiate_model_training(X_train, X_test, y_train, y_test)