'''Main runner function for HCST survival project submodules.'''

import classifier_testing.classifier_test as test
import configuration as config

if __name__=='__main__':

    datasets=config.DATASETS

    test.run()
