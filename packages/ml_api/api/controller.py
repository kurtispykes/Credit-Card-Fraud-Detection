from flask import Blueprint, request, jsonify

from random_forest_model.inference import predict
from random_forest_model import __version__ as _version
from api.config import get_logger
from api import __version__ as api_version

_logger = get_logger(logger_name=__name__)

prediction_app = Blueprint("prediction_app", __name__)

@prediction_app.route("/health", methods=["GET"])
def health():
    if request.method == "GET":
        _logger.info("Health status ok")
        return "Working Fine"

@prediction_app.route("/version", methods=["GET"])
def version():
    if request.method == "GET":
        return jsonify({"model_version": _version,
                        "api_version": api_version})

@prediction_app.route("/v1/inference/random_forest_model", methods=['POST'])
def inference():
    if request.method == 'POST':
        # Extract post data from request body as json
        json_data = request.get_json()
        _logger.info(f'Inputs: {json_data}')

        # Model prediction
        result = predict(input_data=json_data)
        _logger.info(f"Outputs: {result}")

        # Convert numpy ndarray to list
        predictions = result.get("predictions")[0]
        version = result.get("version")

        # Return the response as JSON
        return jsonify({"predictions": predictions,
                        "version": version})