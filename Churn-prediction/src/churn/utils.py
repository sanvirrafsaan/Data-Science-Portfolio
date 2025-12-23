from functools import cached_property
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import statsmodels.api as sm
from scipy.stats import (
    chi2_contingency,
    kstest,
    levene,
    mannwhitneyu,
    norm,
    shapiro,
    ttest_ind,
    zscore,
)
from scipy.stats.contingency import association

