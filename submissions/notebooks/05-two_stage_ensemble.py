{"metadata":{"kernelspec":{"language":"python","display_name":"Python 3","name":"python3"},"language_info":{"name":"python","version":"3.10.12","mimetype":"text/x-python","codemirror_mode":{"name":"ipython","version":3},"pygments_lexer":"ipython3","nbconvert_exporter":"python","file_extension":".py"},"kaggle":{"accelerator":"none","dataSources":[{"sourceId":70942,"databundleVersionId":10381525,"sourceType":"competition"},{"sourceId":10934306,"sourceType":"datasetVersion","datasetId":6799119}],"dockerImageVersionId":30918,"isInternetEnabled":false,"language":"python","sourceType":"notebook","isGpuEnabled":false}},"nbformat_minor":4,"nbformat":4,"cells":[{"cell_type":"code","source":"'''Data preprocessing and inference pipeline. Handles data cleaning and encoding\nand feature engineering for inference. Makes predictions.'''\n\nimport os\nimport time\nimport pickle\nimport multiprocessing as mp\nfrom typing import Callable\n\nimport numpy as np\nimport pandas as pd\n\n#######################################################\n# ASSET PATHS #########################################\n#######################################################\n\n# Figure out if we are in a Kaggle notebook or not based on the current path and set file paths accordingly\nif os.getcwd() == '/kaggle/working':\n\n    print('Running on Kaggle')\n\n    INPUT_DATA_FILE='/kaggle/input/equity-post-HCT-survival-predictions/test.csv'\n    PREDICTION_OUTPUT_FILE='submission.csv'\n\n    # Models and other assets for each pipeline step\n    dataset_path='/kaggle/input/hsct-survival-two-stage-ensemble-model-assets'\n    DATA_CLEANING_ASSETS=f'{dataset_path}/01-data_cleaning.pkl'\n    DATA_ENCODING_ASSETS=f'{dataset_path}/02-data_encoding.pkl'\n    SURVIVAL_MODEL_ASSETS=f'{dataset_path}/03-survival.pkl'\n    KLD_MODEL_ASSETS=f'{dataset_path}/04-kullback-leibler_divergence.pkl'\n    CLASSIFIER_MODEL_FILE=f'{dataset_path}/05-EFS_classifier.pkl'\n    REGRESSOR_ASSETS_FILE=f'{dataset_path}/06-regressor.pkl'\n\nelse:\n    INPUT_DATA_FILE='../../pipeline_assets/data/train.csv'\n    PREDICTION_OUTPUT_FILE='../predictions/two_stage_ensemble_submission.csv'\n\n    # Models and other assets for each pipeline step\n    models='../../pipeline_assets/models'\n    DATA_CLEANING_ASSETS=f'{models}/01-data_cleaning.pkl'\n    DATA_ENCODING_ASSETS=f'{models}/02-data_encoding.pkl'\n    SURVIVAL_MODEL_ASSETS=f'{models}/03-survival.pkl'\n    KLD_MODEL_ASSETS=f'{models}/04-kullback-leibler_divergence.pkl'\n    CLASSIFIER_MODEL_FILE=f'{models}/05-EFS_classifier.pkl'\n    REGRESSOR_ASSETS_FILE=f'{models}/06-regressor.pkl'\n\n\n#######################################################\n# FUNCTIONS ###########################################\n#######################################################\n\ndef clean_data(\n        data_df:pd.DataFrame,\n        data_cleaning_assets:str\n) -> pd.DataFrame:\n\n    '''Main function to do feature cleaning.'''\n\n    with open(data_cleaning_assets, 'rb') as input_file:\n        assets=pickle.load(input_file)\n\n    for feature, translation_dict in assets['feature_levels'].items():\n        data_df[feature]=data_df[feature].replace(translation_dict)\n\n    data_df.replace(assets['nan_placeholders'], inplace=True)\n\n    return data_df\n\n\ndef encode_data(\n        data_df:pd.DataFrame,\n        data_cleaning_assets:str,\n        data_encoding_assets:str,\n) -> pd.DataFrame:\n\n    '''Main function to do data encoding.'''\n\n\n    #######################################################\n    # ASSET LOADING #######################################\n    #######################################################\n\n    # Load feature type definitions\n    with open(data_cleaning_assets, 'rb') as input_file:\n        assets=pickle.load(input_file)\n\n    feature_types_dict=assets['feature_types']\n\n    # Load encoder and transformer models\n    with open(data_encoding_assets, 'rb') as input_file:\n        assets=pickle.load(input_file)\n\n    target_encoder=assets['target_encoder']\n    knn_imputer=assets['knn_imputer']\n\n\n    #######################################################\n    # FEATURE ENCODING ####################################\n    #######################################################\n\n    # Get categorical features\n    categorical_df=data_df[feature_types_dict['Nominal'] + feature_types_dict['Ordinal']]\n\n    # Encode the nominal & ordinal features\n    encoded_categorical_features=target_encoder.transform(categorical_df)\n\n    # Rebuild the dataframe\n    encoded_categorical_features_df=pd.DataFrame(\n        encoded_categorical_features,\n        columns=feature_types_dict['Nominal'] + feature_types_dict['Ordinal']\n    )\n\n    #######################################################\n    # DATA CLEANING #######################################\n    #######################################################\n\n    # Clean NANs in the interval features\n    imputed_interval_df=impute_numerical_features(\n        df=data_df,\n        features=feature_types_dict['Interval'],\n        knn_imputer=knn_imputer\n    )\n\n    # Join the data back together\n    data_df=pd.concat([encoded_categorical_features_df, imputed_interval_df], axis=1)\n\n    return data_df\n\n\ndef engineer_features(\n        data_df:pd.DataFrame,\n        survival_model_assets:str,\n        kld_model_assets:str,\n        classifier_model_file:str\n) -> pd.DataFrame:\n\n    '''Main function to run feature engineering operations.'''\n\n    #######################################################\n    # ASSET LOADING #######################################\n    #######################################################\n\n    with open(survival_model_assets, 'rb') as input_file:\n        assets=pickle.load(input_file)\n\n    coxph_features=assets['coxph_features']\n    waft_features=assets['weibullaft_features']\n    coxph_model=assets['coxph_model']\n    waft_model=assets['weibullaft_model']\n\n    with open(kld_model_assets, 'rb') as input_file:\n        kld_models=pickle.load(input_file)\n\n    with open(classifier_model_file, 'rb') as input_file:\n        classifier_model=pickle.load(input_file)\n\n    #######################################################\n    # FEATURE ENGINEERING #################################\n    #######################################################\n\n    data_df=cox_ph(\n        data_df=data_df,\n        coxph_features=coxph_features,\n        coxph_model=coxph_model\n    )\n\n    print(f' CoxPH features added, nan count: {data_df.isnull().sum().sum()}')\n\n    data_df=weibull_aft(\n        data_df=data_df,\n        waft_features=waft_features,\n        waft_model=waft_model\n    )\n\n    print(f' Weibul AFT features added, nan count: {data_df.isnull().sum().sum()}')\n\n    data_df=kullback_leibler_score(\n        data_df=data_df,\n        kld_models=kld_models\n    )\n\n    print(f' Kullback-Leibler divergence scores added, nan count: {data_df.isnull().sum().sum()}')\n\n    data_df=learned_efs(\n        data_df=data_df,\n        classifier_model=classifier_model\n    )\n\n    print(f' Learned EFS probability added, nan count: {data_df.isnull().sum().sum()}')\n    print()\n\n    return data_df\n\n\n#######################################################\n# PREDICTION ##########################################\n#######################################################\n\ndef predict(data_df:pd.DataFrame, assets_file:str) -> list:\n    '''Main inference function.'''\n\n    # Load model\n    with open(assets_file, 'rb') as input_file:\n        assets=pickle.load(input_file)\n\n    # Unpack the assets\n    scaler=assets['scaler']\n    model=assets['model']\n\n    # Scale the data\n    data_df=scaler.transform(data_df)\n\n    # Make predictions\n    predictions=model.predict(data_df)\n\n    return predictions\n\n\n#######################################################\n# OTHER HELPER FUNCTIONS ##############################\n#######################################################\n\ndef impute_numerical_features(\n        df:pd.DataFrame,\n        features:list,\n        knn_imputer:Callable\n) -> pd.DataFrame:\n\n    '''Takes a set of numerical features, fills NAN with KNN imputation, returns clean features\n    as Pandas dataframe.'''\n\n    # Select all of the numeric columns for input into imputation\n    numerical_df=df.select_dtypes(include='number').copy()\n\n    # Impute missing values\n    imputed_data=knn_imputer.transform(numerical_df)\n\n    # Re-build dataframe\n    imputed_df=pd.DataFrame(\n        imputed_data,\n        columns=numerical_df.columns\n    )\n\n    # Select only the target features\n    imputed_df=imputed_df[features].copy()\n\n    # Fix the index\n    imputed_df.set_index(df.index, inplace=True)\n\n    # Set the types\n    imputed_df=imputed_df.astype('float64').copy()\n\n    return imputed_df\n\n\ndef cox_ph(\n        data_df:pd.DataFrame,\n        coxph_features:list,\n        coxph_model:Callable\n) -> pd.DataFrame:\n\n    '''Adds Cox PH features.'''\n\n    survival_functions=coxph_model.predict_survival_function(data_df[coxph_features])\n    partial_hazards=coxph_model.predict_partial_hazard(data_df[coxph_features])\n    data_df['coxph_survival']=survival_functions.iloc[-1]\n    data_df['coxph_partial_hazard']=partial_hazards\n\n    return data_df\n\n\ndef weibull_aft(\n        data_df:pd.DataFrame,\n        waft_features:list,\n        waft_model:Callable\n) -> pd.DataFrame:\n\n    '''Adds Weibull AFT features.'''\n\n    survival_functions=waft_model.predict_survival_function(data_df[waft_features])\n    expectations=waft_model.predict_expectation(data_df[waft_features])\n    data_df['weibullaft_survival']=survival_functions.iloc[-1]\n    data_df['weibullaft_expectation']=expectations\n\n    return data_df\n\n\ndef kullback_leibler_score(\n        data_df:pd.DataFrame,\n        kld_models:dict\n) -> pd.DataFrame:\n\n    '''Adds Kullback-Leibler divergence scores for Cox PH and Weibull AFT features'''\n\n    for feature, kernel_density_estimate in kld_models.items():\n\n        data=np.array(data_df[feature])\n        workers=mp.cpu_count() - 1\n\n        if workers >= data.shape[0]:\n            kld_score=kernel_density_estimate(data)\n\n        else:\n            with mp.Pool(workers) as p:\n                kld_score=np.concatenate(p.map(kernel_density_estimate, np.array_split(data, workers)))\n\n        data_df[f'{feature}_kld']=kld_score\n\n    return data_df\n\n\ndef learned_efs(\n        data_df:pd.DataFrame,\n        classifier_model:Callable\n) -> pd.DataFrame:\n\n    '''Adds learned EFS probability feature.'''\n\n    data_df['learned_efs']=classifier_model.predict_proba(data_df)[:,1]\n\n    return data_df\n\n#######################################################\n# MAIN INFERENCE PIPELINE #############################\n#######################################################\n\nif __name__ == '__main__':\n\n    print()\n    print('########################################################')\n    print('# HSCT Survival inference pipeline #####################')\n    print('########################################################')\n    print()\n\n    # Read the data\n    data_df=pd.read_csv(INPUT_DATA_FILE)\n\n    # Save the id column for later\n    ids=data_df['ID']\n\n    # Drop unnecessary columns, if present\n    data_df.drop(['efs', 'efs_time', 'ID'], axis=1, inplace=True, errors='ignore')\n\n    print(f'Loaded data from: {INPUT_DATA_FILE}')\n    print(f'Data shape: {data_df.shape}')\n    print(f'nan count: {data_df.isnull().sum().sum()}')\n    print()\n\n    # Translate feature level values and replace string NAN\n    # placeholders with actual np.nan.\n    print('Starting data cleaning.')\n    start_time=time.time()\n\n    data_df=clean_data(\n        data_df=data_df,\n        data_cleaning_assets=DATA_CLEANING_ASSETS\n    )\n\n    dt=time.time()-start_time\n    print(f'Data cleaning complete, run time: {dt:.0f} seconds')\n    print(f'nan count: {data_df.isnull().sum().sum()}')\n    print()\n\n    # Encodes features with SciKit-learn continuous target encoder, then\n    # applies a power transform with standardization using the Yeo-Johnson\n    # method. Missing values filled with KNN imputation.\n    print('Starting feature encoding.')\n    start_time=time.time()\n\n    data_df=encode_data(\n        data_df=data_df,\n        data_cleaning_assets=DATA_CLEANING_ASSETS,\n        data_encoding_assets=DATA_ENCODING_ASSETS\n    )\n\n    dt=time.time()-start_time\n    print(f'Feature encoding complete, run time: {dt:.0f} seconds')\n    print(f'nan count: {data_df.isnull().sum().sum()}')\n    print()\n\n    # Does feature engineering. Adds survival model features and their\n    # corresponding Kullback-Leibler divergence scores. Also adds\n    # learned EFS probability\n    print(f'Starting feature engineering.')\n    start_time=time.time()\n\n    data_df=engineer_features(\n        data_df=data_df,\n        survival_model_assets=SURVIVAL_MODEL_ASSETS,\n        kld_model_assets=KLD_MODEL_ASSETS,\n        classifier_model_file=CLASSIFIER_MODEL_FILE\n    )\n\n    dt=time.time()-start_time\n    print(f'Feature engineering complete, run time: {dt:.0f} seconds')\n    print(f'nan count: {data_df.isnull().sum().sum()}')\n    print()\n    print('Inference dataset:')\n    print(data_df.head().transpose())\n    print()\n\n    # Does the actual inference run\n    print(f'Starting inference.')\n    start_time=time.time()\n\n    predictions=predict(\n        data_df=data_df,\n        assets_file=REGRESSOR_ASSETS_FILE\n    )\n\n    predictions_df=pd.DataFrame.from_dict({'ID': ids, 'prediction': predictions})\n\n    dt=time.time()-start_time\n    print(f'Inference complete, run time: {dt:.0f} seconds')\n    print()\n    print('Predictions:')\n    print(predictions_df.head(20))\n\n    predictions_df.to_csv(PREDICTION_OUTPUT_FILE, index=False)","metadata":{"_uuid":"bbb29811-6aa0-4f6f-a715-6a7c4ab7eeac","_cell_guid":"794bca27-d054-4e0b-87be-60d3d087859d","trusted":true,"collapsed":false,"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-03-05T21:37:29.181372Z","iopub.execute_input":"2025-03-05T21:37:29.181814Z","iopub.status.idle":"2025-03-05T21:37:29.934776Z","shell.execute_reply.started":"2025-03-05T21:37:29.181766Z","shell.execute_reply":"2025-03-05T21:37:29.933782Z"}},"outputs":[],"execution_count":null}]}