import pandas as pd

def validate_inputs(input_data:pd.DataFrame) -> pd.DataFrame:
    """Check model inputs for unprocessable values"""
    validated_data = input_data.copy()

    # Check for missing values in categorical features and fill them with NONE
    categorical_columns = input_data.select_dtypes(include=['object']).columns
    if input_data[categorical_columns].isna().any().any():
        validated_data[categorical_columns] = validated_data[categorical_columns].\
            fillna(value="NONE")

    # # Check for missing values in numerical features and fill them will 0
    # numeric_columns = input_data.select_dtypes(include=['number']).columns
    # if input_data[numeric_columns].isna().any().any():
    #      validated_data[numeric_columns] = validated_data[numeric_columns].\
    #          fillna(value=0)

    return validated_data



