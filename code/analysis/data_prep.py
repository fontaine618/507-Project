import data
import models
import pandas as pd

train, test, features = data.load_train_test_and_feature_list()

genres = [col for col in train.columns if col.startswith("genre")]
tags = [col for col in train.columns if col.startswith("tag")]

print(set(train["movie_id"].unique()) == set(test["movie_id"].unique()))
print(set(train["user_id"].unique()) == set(test["user_id"].unique()))

