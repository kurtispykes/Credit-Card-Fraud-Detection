import json

from random_forest_model import __version__ as _version
from random_forest_model.preprocessors import data_management as dm

def test_health_endpoint_returns_200(flask_test_client):
    # When
    response = flask_test_client.get('/health')

    # Then
    assert response.status_code == 200


def test_prediction_endpoint_returns_prediction(flask_test_client):
    # Given
    test_data = dm.read_test_data()
    post_json  = test_data[0:1].to_json(orient="records")

    # When
    response = flask_test_client.post("/v1/inference/random_forest_model",
                                      json=post_json)

    # Then
    assert response.status_code == 200
    response_json = json.loads(response.data)
    prediction = response_json["predictions"]
    response_version = response_json["version"]
    assert isinstance(prediction, float)
    assert prediction == 0.056
    assert response_version == _version