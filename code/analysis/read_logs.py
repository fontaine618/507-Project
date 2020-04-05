import pandas as pd
import re
import ast

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 1000)



def read_logs(metrics: pd.DataFrame):
    # get columns
    columns = set(metrics.columns)
    descriptions = []
    find_key = re.compile(r"[\s$][a-zA-Z_]+:")
    for desc in metrics.index:
        string = " " + desc[(desc.find("(") + 1):-1]
        string = string.replace("cuda:0", "'cuda0'")
        string = string.replace("cpu", "'cpu'")
        for type in ["RegressorRelu6", "RegressorSigmoid", "Regressor", "Ordinal", "Classifier"]:
            if type in string:
                string = string.replace(type, "'" + type + "'")
                break
        keys = find_key.findall(string)
        keys = [key.strip(" ").strip(":") for key in keys]
        for key in keys:
            string = string.replace(key, "'" + key + "'")
        string = "{" + string + "}"
        descriptions.append(ast.literal_eval(string))
    for desc in descriptions:
        columns.update(desc.keys())
    df = pd.DataFrame(columns=columns)
    for m, d in enumerate(descriptions):
        df = df.append(metrics.iloc[m].append(pd.Series(d)), ignore_index=True)
    return df

metrics = pd.read_table("models/log/svd.tsv", index_col="description")

# NN
metrics = pd.read_table("models/log/nn.tsv", index_col="description")
df = read_logs(metrics)
df["layers"] = [str(l) for l in df["layers"]]
df_features = df.set_index(["use_features", "type", "n_embed_users", "n_embed_movies", "layers"]).drop(columns=[
    "num_epochs", "seed", "device", "lr", "n_movies", "n_users", "batch_size", "time"
])

metrics = pd.read_table("models/log/nn_nofeatures.tsv", index_col="description")
df = read_logs(metrics)
df["layers"] = [str(l) for l in df["layers"]]
df_nofeatures = df.set_index(["use_features", "type", "n_embed_users", "n_embed_movies", "layers"]).drop(columns=[
    "num_epochs", "seed", "device", "lr", "n_movies", "n_users", "batch_size", "time"
])

df = df_nofeatures
bests = df[(df["cv_mse"] < 1.08) | (df["cv_accuracy"] > 0.36)]
bests.index = bests.index.droplevel()
table = bests[["cv_mse", "cv_accuracy", "test_mse", "test_accuracy"]]
table.sort_values(by="cv_mse", inplace=True)
table = table.applymap("{0:.4f}".format)
table.to_latex(
    buf="../tex/Report/table/nn_nofeatures.tex",
    caption="sdfsdfs",
    label="tab:results.nn"
)



# KNN
metrics = pd.read_table("models/log/knn.tsv", index_col="description")
df = read_logs(metrics)
df_knn = df.set_index(["type", "n_neighbors", "user_prop", "tag_prop"]).drop(columns=[
    "time"
])
df_knn.sort_index(level=[0, 1, 2, 3], inplace=True)
bests = df_knn[(df_knn["cv_mse"] < 1.02) | (df_knn["cv_accuracy"] > 0.378)]
table = bests[["cv_mse", "cv_accuracy", "test_mse", "test_accuracy"]]
table.sort_values(by="cv_mse", inplace=True)
table = table.applymap("{0:.4f}".format)
table.to_latex(
    buf="../tex/Report/table/knn.tex",
    caption="sdfsdfs",
    label="tab:results.knn"
)