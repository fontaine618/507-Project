import data
import models

train, test, features = data.load_train_test_and_feature_list()


# Train knn
for type in ["Classifier", "Regressor"]:
	for k in [3, 5, 7, 10, 15, 20, 50]:
		knn = models.KNN(n_neighbors=k, type=type)
		knn.add_train_data(train, "rating", features)
		knn.train()
		knn.test(test)
		knn.log()