import data
import models

train, test, features = data.load_train_test_and_feature_list()

# Train NN
for type in ["Regressor"]:
	for layers in [
		[(100, "relu"), (100, "relu")],
	]:
		nn = models.NNEmbed(type=type, layers=layers)
		nn.add_train_data(train, "rating", features)
		nn.train()
		nn.cv()
		nn.test(test)
		nn.log()
