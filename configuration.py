'''Globals for scripts'''

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