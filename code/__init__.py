import data
import models
from multiprocessing import Pool
import itertools

train, test, features = data.load_train_test_and_feature_list()



# Train NN

nn = models.NN(type="Classifier")
nn.add_train_data(train, "rating", features)
nn.train()
nn.cv()
nn.test(test)
nn.log()

# cross validation
types = ["Classifier", "Regressor"]
ks = [3, 5, 10, 20, 50, 100]
user_props = [0.0, 1.0, 10., 100.]
tag_props = [0.0, 0.001, 0.01, 0.1, 1.0]
setups = list(itertools.product(types, ks, user_props, tag_props))


def do_cv(type, k, user_prop, tag_prop):
	knn = models.KNN(n_neighbors=k, type=type, user_prop=user_prop, tag_prop=tag_prop)
	knn.add_train_data(train, "rating", features)
	knn.train()
	knn.cv()
	knn.test(test)
	knn.log()

# using multiprocessing
with Pool(6) as pool:
	cv_metrics = pool.starmap(do_cv, setups)


# # Train knn
# for type in ["Classifier", "Regressor"]:
# 	for k in [3, 5, 7, 10, 15, 20, 50]:
# 		knn = models.KNN(n_neighbors=k, type=type)
# 		knn.add_train_data(train, "rating", features)
# 		knn.train()
# 		knn.cv()
# 		knn.test(test)
# 		knn.log()
#
#
# for prop in [0, 0.001, 0.01, 0.1, 1.0, 10., 100.]:
# 	knn = models.KNN(n_neighbors=5, type="Regressor", tag_prop=prop)
# 	knn.add_train_data(train, "rating", features)
# 	knn.train()
# 	knn.cv()
# 	knn.test(test)
# 	knn.log()
#
# for key, val in knn.metrics.items():
# 	print(key, val)
#
#
import pandas as pd
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 1000)

metrics = pd.read_table("models/log/nn.tsv", index_col="description")
print(metrics[[id for id in metrics.columns if id.startswith("cv_")]])



#Train svd
# for embed_dim in [10, 20, 50]:
# 	for iter_nums in [20, 30, 50]:
# 		svd = models.SVD(embed_dim=embed_dim, iter_nums=iter_nums)
# 		svd.add_train_data(train, "rating", features)
# 		svd.train()
# 		svd.test(test)
# 		svd.log()
