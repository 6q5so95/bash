import lightgbm as lgb
import numpy as np
import pandas as pd
import random

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.utils import check_random_state
#from sklearn.datasets import load_boston
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error
from gplearn.genetic import SymbolicTransformer, SymbolicRegressor
from gplearn.functions import make_function
from gplearn.fitness import make_fitness
from statistics import mean
from math import sqrt


# basic library
from glob import glob
from pathlib import Path
import os
import sys
import toml

import pandas as pd
import numpy as np

from typing import Union, List, Optional

from matplotlib_venn import venn2
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from contextlib import contextmanager
from time import time

import dtale
from dataprep.datasets import get_dataset_names
from dataprep.eda import create_report

import lightgbm as lgb
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GroupKFold

from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from pprint import pprint

import warnings
warnings.filterwarnings('ignore')
