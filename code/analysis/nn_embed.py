import data
import models
import pandas as pd

train, test, features = data.load_train_test_and_feature_list()

# Train NN
for type in ["Regressor", "RegressorRelu6", "RegressorSigmoid", "Ordinal", "Classifier"]:
    for layers in [
        [(512, "relu"), (64, "relu")],
        [(1024, "relu"), (64, "relu")],
        [(1024, "relu"), (128, "relu")],
        [(1024, "relu"), (256, "relu"), (64, "relu")],
    ]:
        for n_embed_users in [32, 64]:
            for n_embed_movies in [64, 128]:
                nn = models.NNEmbed(
                    type=type,
                    layers=layers,
                    n_embed_users=n_embed_users,
                    n_embed_movies=n_embed_movies,
                    use_features=True
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

metrics["test_mse"].sort_values()


import matplotlib.pyplot as plt
import numpy as np
xs = np.linspace(start=-4, stop=10, num=1000)

fig = plt.figure(figsize=(6, 3))
ax = fig.gca()
ax.grid(linestyle=":")
ax.plot(xs, xs, label="Identity")
ax.plot(xs, np.minimum(np.maximum(xs, 0), 6), label="ReLU6")
ax.plot(xs, 10. / (1. + np.exp(-(xs - 3.))) - 2., label="Sigmoid")
ax.legend(title="Function")
ax.hlines([1, 5], xmin=-2, xmax=8, linestyle="--")
ax.set_xlabel("x")
ax.set_ylabel("f(x)")
fig.tight_layout()
fig.savefig("../tex/Report/fig/functions_nn.pdf")