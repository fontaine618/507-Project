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

df = pd.concat([df_features, df_nofeatures])

df["cv_mse"].sort_values(ascending=True)
df["cv_accuracy"].sort_values(ascending=False)