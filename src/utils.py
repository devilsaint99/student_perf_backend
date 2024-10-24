import pandas as pd, os, sys
import dill
from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV


def split_io_data(raw_data_df:pd.DataFrame):
    input_data_df = raw_data_df.drop(columns=['math_score'])
    output_data_df = raw_data_df['math_score']
    return input_data_df, output_data_df

def get_feature_typewise(input_data_df):
    num_features = input_data_df.select_dtypes(exclude="O").columns
    cat_features = input_data_df.select_dtypes(include="O").columns
    return num_features, cat_features

def save_obj(file_path, ml_obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(ml_obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)
    
def load_obj(obj_path):
    try:
        dir_path = os.path.dirname(obj_path)
        with open(obj_path, "rb") as file_obj:
            return dill.load(file_obj)
        return 
    except Exception as e:
        raise CustomException(e, sys)
def model_evaluater(X_train, X_test, y_train, y_test, models, params):
    try:
        report = {}
        for k,v in models.items():
            model = v
            param = params[k]

            grid_search_cv = GridSearchCV(model, param, cv=3, n_jobs=-1, scoring='r2')
            grid_search_cv.fit(X_train,y_train)
            logging.info(f"Best params for {k} are {grid_search_cv.best_params_}")

            model.set_params(**grid_search_cv.best_params_)

            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            r2_score_train = r2_score(y_train, y_train_pred)
            r2_score_test= r2_score(y_test, y_test_pred)
            logging.info(f"R-squared values for {k} is {r2_score_test}")
            report[k] = r2_score_test
            models[k] = model
        return report
    except Exception as e:
        logging.error(CustomException(e, sys) )
        raise CustomException(e, sys) 