import data
import models

train, test, features = data.load_train_test_and_feature_list()


# Train Kknn.NN
knn = models.KNN(n_neighbors=5)
knn.model
knn.add_train_data(train, "rating", features)
knn.train()
knn.test(test)
knn.log(path="models/log/knn.tsv", description="5NN standardized")