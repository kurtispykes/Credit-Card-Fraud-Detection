from flask import Blueprint, request, jsonify
from random_forest_model.inference import predict

from api.config import get_logger

_logger = get_logger(logger_name=__name__)

prediction_app = Blueprint("prediction_app", __name__)

@prediction_app.route("/health", methods=["GET"])
def health():
    if request.method == "GET":
        _logger.info("Health status ok")
        return "Working Fine"

@prediction_app.route("/v1/inference/random_forest_model", methods=['POST'])
def inference():
    if request.method == 'POST':
        json_data = request.get_json()
        _logger.info(f'Inputs: {json_data}')

        result = predict(input_data=json_data)
        _logger.info(f'Outputs: {result}')

        predictions = result.get('predictions')[0]
        version = result.get('version')

        return jsonify({'predictions': predictions,
                        'version': version})