import pandas as pd
from pycaret import classification

data_classification = pd.read_csv('./db_GIST.csv')
classification_setup = classificartion.setup(data= data_classification, target='strok')
tune_catboost = classification.tune_model('catboost')
