'''Globals for scripts & notebooks.'''

##########################################################
# Training and inference pipeline asset paths ############
##########################################################

INPUT_DATA_FILE='./pipeline_assets/data/train.csv'

# Models and other assets for each pipeline step
models='./pipeline_assets/models'
DATA_CLEANING_ASSETS=f'{models}/01-data_cleaning.pkl'
DATA_ENCODING_ASSETS=f'{models}/02-data_encoding.pkl'
SURVIVAL_MODEL_ASSETS=f'{models}/03-survival.pkl'
KLD_MODEL_ASSETS=f'{models}/04-kullback-leibler_divergence.pkl'
CLASSIFIER_MODEL_ASSETS=f'{models}/05-EFS_classifier.pkl'
REGRESSOR_MODEL_ASSETS=f'{models}/06-regressor.pkl'

# Target data files for each training pipeline step
data='./pipeline_assets/data'
DATA_CLEANING_RESULT=f'{data}/01-cleaned_data.pkl'
DATA_ENCODING_RESULT=f'{data}/02-encoded_data.pkl'
SURVIVAL_FEATURES_RESULT=f'{data}/03-survival_features.pkl'
KLD_FEATURES_RESULT=f'{data}/04-KLD_features.pkl'
EFS_FEATURE_RESULT=f'{data}/05-EFS_feature.pkl'
PREDICTIONS=f'{data}/06-predictions.pkl'

##########################################################
# Notebook asset paths ###################################
##########################################################

# Project paths
PROJECT_ROOT_PATH='.'
NOTEBOOKS_PATH=f'{PROJECT_ROOT_PATH}/notebooks'
MODELS_PATH=f'{PROJECT_ROOT_PATH}/models'
DATA_PATH=f'{PROJECT_ROOT_PATH}/data'
PROCESSED_DATA=f'{DATA_PATH}/processed'
PLOTS=f'{DATA_PATH}/results/plots'
RESULTS=f'{DATA_PATH}/results/data'


# Data files
DATASETS=f'{PROCESSED_DATA}/02.1-dataset_definitions.pkl'
COXPH_FEATURES=f'{PROCESSED_DATA}/02.1-coxPH_survival.pkl'
WEIBULLAFT_FEATURES=f'{PROCESSED_DATA}/02.2-weibullAFT_survival.pkl'
LEARNED_EFS_FEATURES=f'{PROCESSED_DATA}/02.4-learned_efs.pkl'