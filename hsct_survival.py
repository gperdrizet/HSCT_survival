'''Main runner function for HSCT survival project submodules.'''

import classifier_testing.classifier_test as test

if __name__=='__main__':

    test.run(
        './data/processed/02.1-no-multicollinearity_one_hot_ordinal_nan_imputed_data_df.pkl',
        './data/results/data/sklearn_classifier_test.pkl'
    )
