'''Data data processing pipeline. Handles data cleaning and encoding
and feature engineering for inference.'''
import pandas as pd
from data_pipeline import cleaning
from data_pipeline import encoding

####################################################
# Asset file paths #################################
####################################################

# Data
INPUT_DATA_FILE='./data/raw/train.csv'
FEATURE_TYPES_DICT_FILE='./data/processed/01.1-feature_type_dict.pkl'
FEATURE_LEVEL_DICTS_FILE='./data/processed/01.1-feature_value_translation_dicts.pkl'
NAN_DICTS_FILE='./data/processed/01.1-nan_placeholders_list.pkl'

# Models
TARGET_ENCODER_FILE='./models/01.2-continuous_target_encoder.pkl'
POWER_TRANSFORMER_FILE='./models/01.2-continuous_target_power_transformer.pkl'
KNN_IMPUTER_FILE='./models/01.2-numerical_imputer.pkl'

def run():
    '''Main function to run data pipeline.'''

    # Read data
    data_df=pd.read_csv(INPUT_DATA_FILE)

    # Drop unnecessary columns if present
    data_df.drop(['efs', 'efs_time', 'ID'], axis=1, inplace=True, errors='ignore')

    # Translate feature level values and replace string NAN
    # placeholders with actual np.nan.
    data_df=cleaning.run(
        data_df=data_df,
        feature_level_dicts_file=FEATURE_LEVEL_DICTS_FILE,
        nan_dicts_file=NAN_DICTS_FILE
    )

    # Encodes features with SciKit-learn continuous target encoder, then 
    # applies a power transform with standardization using the Yeo-Johnson 
    # method. Missing values filled with KNN imputation.
    data_df=encoding.run(
        data_df=data_df,
        feature_types_dict_file=FEATURE_TYPES_DICT_FILE,
        target_encoder_file=TARGET_ENCODER_FILE,
        power_transformer_file=POWER_TRANSFORMER_FILE,
        knn_imputer_file=KNN_IMPUTER_FILE
    )

    print(data_df.head().transpose())

    return data_df
