'''Runs EFS time inference using tuned XGBoost regression model.'''

import pickle
import pandas as pd


def run(data_df:pd.DataFrame, assets_file:str) -> list:
    '''Main inference function.'''

    # Load model
    with open(assets_file, 'rb') as input_file:
        assets=pickle.load(input_file)

    # Unpack the assets
    scaler=assets['scaler']
    model=assets['model']

    # Scale the data
    data_df=scaler.transform(data_df)

    # Make predictions
    predictions=model.predict(data_df)

    return predictions