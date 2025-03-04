'''Data data processing pipeline. Handles data cleaning and encoding
and feature engineering for inference.'''
import pandas as pd
from data_pipeline import cleaning
from data_pipeline import encoding
from data_pipeline import feature_engineering

####################################################
# Asset file paths #################################
####################################################

# Data
INPUT_DATA_FILE='./data/raw/train.csv'
FEATURE_TYPES_DICT_FILE='./data/processed/01.1-feature_type_dict.pkl'
FEATURE_LEVEL_DICTS_FILE='./data/processed/01.1-feature_value_translation_dicts.pkl'
NAN_DICTS_FILE='./data/processed/01.1-nan_placeholders_list.pkl'
COXPH_FEATURES_FILE='./data/processed/02.1-coxPH_significant_features.pkl'
WAFT_FEATURES_FILE='./data/processed/02.2-weibullAFT_significant_features.pkl'

# Models
TARGET_ENCODER_FILE='./models/01.2-continuous_target_encoder.pkl'
POWER_TRANSFORMER_FILE='./models/01.2-continuous_target_power_transformer.pkl'
KNN_IMPUTER_FILE='./models/01.2-numerical_imputer.pkl'
COXPH_MODEL_FILE='./models/02.1-coxPH_model.pkl'
WAFT_MODEL_FILE='./models/02.2-weibullAFT_model.pkl'
KLD_MODELS_FILE='./models/02.3-KLD_models.pkl'
EFS_MODEL_FILE='./models/02.4-EFS_classifier_model.pkl'

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

    # Does feature engineering. Adds survival model features and their
    # their corresponding Kullback-Leibler divergence scores. Also adds
    # learned EFS probability
    data_df=feature_engineering.run(
        data_df=data_df,
        coxph_features_file=COXPH_FEATURES_FILE,
        waft_features_file=WAFT_FEATURES_FILE,
        coxph_model_file=COXPH_MODEL_FILE,
        waft_model_file=WAFT_MODEL_FILE,
        kld_models_file=KLD_MODELS_FILE,
        efs_model_file=EFS_MODEL_FILE
    )

    print(data_df.head().transpose())

    return data_df
