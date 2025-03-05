'''Main runner function for HSCT survival project submodules.'''

import argparse
from training_pipeline import training
from inference_pipeline import inference
from estimator_testing import estimator_test
from estimator_testing import classifiers
from estimator_testing import regressors


if __name__=='__main__':

    parser=argparse.ArgumentParser()
    parser.add_argument(
        '--task',
        choices=['run_training', 'run_inference', 'test_classifiers', 'test_regressors'],
        help='task to run'
    )

    args=parser.parse_args()

    if args.task is None:
        print('Specify a task with the --task flag.')

    if args.task == 'run_training':
        training.run()

    if args.task == 'run_inference':
        inference.run()

    if args.task == 'test_classifiers':
        estimator_test.run(
            input_file=('./data/processed/01.2-continuous_target_encoded_data_df.pkl'),
            output_file='./data/results/data/sklearn_classifier_test.pkl',
            models=classifiers.models,
            hyperparameters=classifiers.hyperparameters,
            task='classification'
        )

    if args.task == 'test_regressors':
        estimator_test.run(
            input_file='./data/processed/01.2-continuous_target_encoded_data_df.pkl',
            output_file='./data/results/data/sklearn_regressor_test.pkl',
            models=regressors.models,
            hyperparameters=regressors.hyperparameters,
            task='regression'
        )
