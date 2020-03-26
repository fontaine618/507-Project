import data
import models
import pandas as pd

train, test, features = data.load_train_test_and_feature_list()

len(train["movie_id"].unique())
len(test["movie_id"].unique())
