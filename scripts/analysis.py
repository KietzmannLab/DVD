import os
import numpy as np
import pandas as pd

from neuroai.bias_test import evaluate_shape_bias

EVALUATE_BIAS = [False, True][1]
PLOT_SHAPE_BIAS_ACROSS_EPOCH = [False, True][1]
EVALUATE_ACC_ROBUSTNESS = [False, True][0]
PLOT_ACC_ROBUSTNESS = [False, True][0]

PREPROCESSING_INFANTS2ADULT_EEG = [False, True][0]
ANALYSIS_INFANTS2ADULT_EEG_ANN_FUSION = [False, True][0]
PREPROCESSING_INFANTS2ADULT_fMRI = [False, True][0]
ANALYSIS_INFANTS2ADULT_fMRI_ANN_FUSION = [False, True][0]


