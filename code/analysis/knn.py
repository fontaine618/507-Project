import data
import models
import itertools
from multiprocessing import Pool

train, test, features = data.load_train_test_and_feature_list()

# cross validation
types = ["Regressor"]
ks = [50, 75, 100]
user_props = [500., 200.]
tag_props = [0.2, 0.5]
setups = list(itertools.product(types, ks, user_props, tag_props))


def do_cv(type, k, user_prop, tag_prop):
	knn = models.KNN(n_neighbors=k, type=type, user_prop=user_prop, tag_prop=tag_prop)
	knn.add_train_data(train, "rating", features)
	knn.train()
	knn.cv()
	knn.test(test)
	knn.log()


# using multiprocessing
with Pool(3) as pool:
	cv_metrics = pool.starmap(do_cv, setups)
