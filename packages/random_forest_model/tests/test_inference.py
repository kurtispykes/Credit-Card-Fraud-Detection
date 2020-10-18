from random_forest_model import inference
from random_forest_model.preprocessors import data_management as dm

def test_make_single_prediction():
    # Given
    test_data = dm.read_test_data()
    single_test_json = test_data[0:1]

    # When
    subject = inference.predict(input_data=single_test_json)

    # Then
    assert subject is not None
    assert isinstance(subject["predictions"][0], float)
    assert subject["predictions"][0] == 0.092

def test_make_multiple_predictions():
    # Given
    test_data = dm.read_test_data()
    original_data_length = len(test_data)

    # When
    subject = inference.predict(input_data=test_data)

    # Then
    assert subject is not None
    assert len(subject["predictions"]) == original_data_length

