
import data
import models
from multiprocessing import Pool
import itertools
import pandas as pd

#train, test, features = data.load_train_test_and_feature_list()

# Train NN
# for type in ["Regressor"]:
# 	for layers in [
# 		[(100, "relu"), (100, "relu")],
# 	]:
# 		nn = models.NNEmbed(type=type, layers=layers)
# 		nn.add_train_data(train, "rating", features)
# 		nn.train()
# 		nn.cv()
# 		nn.test(test)
# 		nn.log()

# nn = models.NN(type="Regressor", layers=[(10000, "relu"), (1000, "relu")])
# nn.add_train_data(train, "rating", features)
# nn.train()
#
# self = nn
# data_loader = self.train_loader
#
# from scipy import stats
# stats.kendalltau(ys, preds)
# from sklearn.metrics import confusion_matrix
# confusion_matrix(ys, preds_class)
# from sksurv.metrics import concordance_index_censored
# concordance_index_censored(True, ys, preds)
# import matplotlib.pyplot as plt
# plt.hist(preds)
# plt.show()
# plt.hist(preds - ys)
# plt.show()




# # cross validation
# types = ["Classifier", "Regressor"]
# ks = [3, 5, 10, 20, 50, 100]
# user_props = [1000.]
# tag_props = [0.0, 0.001, 0.01, 0.1, 1.0]
# setups = list(itertools.product(types, ks, user_props, tag_props))
#
#
# def do_cv(type, k, user_prop, tag_prop):
# 	knn = models.KNN(n_neighbors=k, type=type, user_prop=user_prop, tag_prop=tag_prop)
# 	knn.add_train_data(train, "rating", features)
# 	knn.train()
# 	knn.cv()
# 	knn.test(test)
# 	knn.log()
#
# # using multiprocessing
# with Pool(6) as pool:
# 	cv_metrics = pool.starmap(do_cv, setups)


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


# import pandas as pd
# pd.set_option('display.max_rows', 100)
# pd.set_option('display.max_columns', 20)
# pd.set_option('display.width', 1000)
#
# metrics = pd.read_table("models/log/knn.tsv", index_col="description")
# print(metrics[[id for id in metrics.columns if id.startswith("cv_")]])
#
#
# metrics["cv_accuracy"].sort_values()


#Train svd
# pd.options.mode.chained_assignment = None
# train, test = data.load_train_test_ratings()
# for embed_dim in [4, 5, 8]:
# 	for iter_nums in [120]:
# 		svd = models.SVD(embed_dim=embed_dim, iter_nums=iter_nums)
# 		svd.add_train_data(train, "rating", ["user_id", "movie_id"])
# 		svd.train()
# 		svd.cv()
# 		svd.test(test)
# 		svd.log()

##play with RBM
pd.options.mode.chained_assignment = None
train, test = data.load_train_test_ratings()

for num_hidden_nodes in [15, 20, 30]:
    for Gibbs_iters in [1, 5]:
        for max_iters in [1, 5, 10, 20]:
            for learning_rate in [0.1, 1]:
                #train, test = data.load_train_test_ratings()
                rbm = models.RBM(num_hidden_nodes=num_hidden_nodes, max_iters=max_iters, learning_rate=learning_rate, Gibbs_iters=Gibbs_iters)
                rbm.add_train_data(train, "rating", ["user_id", "movie_id"])
                rbm.train()
                rbm.cv()
                rbm.test(test)
                rbm.log()




