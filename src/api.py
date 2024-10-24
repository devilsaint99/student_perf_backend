from flask import Flask, request
from flask_restx import Api, Resource
from flask_cors import CORS
from marshmallow import Schema, fields
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
import math


app = Flask(__name__)
CORS(app)
api = Api(app)

class ModelInputSchema(Schema):
    gender = fields.Str(required=True)
    race_ethnicity = fields.Str(required=True)
    parental_level_of_education = fields.Str(required=True)
    lunch = fields.Str(required=True)
    test_preparation_course = fields.Str(required=True)
    reading_score = fields.Int(required=True)
    writing_score = fields.Int(required=True)

model_input_schema = ModelInputSchema()

@api.route('/prediction')
class Prediction(Resource):
    def post(self):
        body = model_input_schema.load(request.json['message'])
        print(body)
        custom_data = CustomData(body)
        features = custom_data.convert_to_df()
        predictor = PredictPipeline()
        maths_score_predict = predictor.predict_math_score(features)
        return math.floor(maths_score_predict)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)