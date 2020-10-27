from random_forest_model import inference
from random_forest_model.preprocessors import data_management as dm

def test_make_single_prediction():
    # Given
    test_data = dm.read_test_data()
    single_test_json = test_data[0:1].to_json(orient="records")

    # When
    subject = inference.predict(input_data=single_test_json)

    # Then
    assert subject is not None
    assert isinstance(subject.get("predictions")[0], float)
    assert subject.get("predictions")[0] == 0.056

def test_make_multiple_predictions():
    # Given
    test_data = dm.read_test_data()
    original_data_length = len(test_data[:100])
    multiple_test_json = test_data[:100].to_json(orient="records")

    # When
    subject = inference.predict(input_data=multiple_test_json)

    # Then
    assert subject is not None
    assert len(subject.get("predictions")) == original_data_length

