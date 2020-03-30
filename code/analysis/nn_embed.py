import data
import models
import pandas as pd

train, test, features = data.load_train_test_and_feature_list()

# Train NN
for type in ["Regressor", "RegressorRelu6", "RegressorSigmoid", "Ordinal", "Classifier"]:
    for layers in [
        [(256, "relu"), (128, "relu"), (64, "relu")],
        [(256, "relu"), (64, "relu"), (64, "relu"), (64, "relu")],
    ]:
        for n_embed_users in [64, 128]:
            for n_embed_movies in [64, 128]:
                nn = models.NNEmbed(
                    type=type,
                    layers=layers,
                    n_embed_users=n_embed_users,
                    n_embed_movies=n_embed_movies,
                    use_features=False
                )
                nn.add_train_data(train, "rating", features)
                nn.train()
                nn.cv()
                nn.test(test)
                nn.log()


pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 1000)

metrics = pd.read_table("models/log/nn.tsv", index_col="description")
print(metrics[[id for id in metrics.columns if id.startswith("cv_")]])

metrics["cv_mse"].sort_values()
