import data
import models

train, test, features = data.load_train_test_and_feature_list()


# Train knn
for type in ["Classifier", "Regressor"]:
	for k in [3, 5, 7, 10, 15, 20, 50]:
		knn = models.KNN(n_neighbors=k, type=type)
		knn.add_train_data(train, "rating", features)
		knn.train()
		knn.cv()
		knn.test(test)
		knn.log()


for prop in [0, 0.001, 0.01, 0.1, 1.0, 10., 100.]:
	knn = models.KNN(n_neighbors=5, type="Regressor", tag_prop=prop)
	knn.add_train_data(train, "rating", features)
	knn.train()
	knn.cv()
	knn.test(test)
	knn.log()

for key, val in knn.metrics.items():
	print(key, val)


import pandas as pd
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 1000)

metrics = pd.read_table("models/log/knn.tsv", index_col="description")
metrics[[id for id in metrics.columns if id.startswith("cv_")]]