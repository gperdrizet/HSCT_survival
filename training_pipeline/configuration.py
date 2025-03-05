'''File paths and parameters for training and inference pipelines'''

# Input data
INPUT_DATA_FILE='./pipeline_assets/data/train.csv'


# Models and other assets for each pipeline step
models='./pipeline_assets/models'
DATA_CLEANING_ASSETS=f'{models}/01-data_cleaning.pkl'
DATA_ENCODING_ASSETS=f'{models}/02-data_encoding.pkl'
SURVIVAL_MODEL_ASSETS=f'{models}/03-survival.pkl'


# Target data files for each training pipeline step
data='./pipeline_assets/data'
DATA_CLEANING_RESULT=f'{data}/01-cleaned_data.pkl'
DATA_ENCODING_RESULT=f'{data}/02-encoded_data.pkl'
SURVIVAL_FEATURES_RESULTS=f'{data}/03-survival_features.pkl'