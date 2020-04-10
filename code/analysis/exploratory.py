import data
import models
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pygam import LinearGAM, s, l, f, te
from sklearn.preprocessing import LabelEncoder

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 1000)

pd.options.mode.chained_assignment = None
train, test = data.load_train_test_ratings()
whole = pd.concat([train, test], axis=0)

svd = models.SVD(embed_dim=4, iter_nums=100)
svd.add_train_data(whole, "rating", ["user_id", "movie_id"])
svd.train()

pred_df = svd.pred_df.clip(1, 5)
pred_df = (pred_df.transpose() - pred_df.mean(axis=1)).transpose()
pred_df["user_id"] = pred_df.index
pred = pd.melt(pred_df, id_vars="user_id", value_name="pred_rating")

movies = data.load_movies()
movies = movies[[g for g in movies.columns if g.startswith("genre_")]]
movies.columns = [g[6:] for g in movies.columns]
movies["movie_id"] = movies.index
movies = pd.melt(movies, id_vars="movie_id", value_name="genre01", var_name="genre")
movies = movies[movies["genre01"] == 1]
movies.set_index("movie_id", inplace=True)
movies.drop(columns="genre01", inplace=True)

users = data.load_users()
users.drop(columns="zip_code", inplace=True)
users["age_binned"] = pd.cut(users["age"], bins=[5, 20, 35, 50, 75])
users["age_binned"].value_counts()

ratings = pred.merge(movies, how="left", on="movie_id", right_index=True)
ratings = ratings.merge(users, how="left", on="user_id", right_index=True)
ratings.dropna(0, inplace=True)

avg_rating = ratings.groupby(["genre", "gender", "age_binned"]).agg({"pred_rating": "mean"})
out = avg_rating.pivot_table(
    values="pred_rating",
    columns=["age_binned", "gender"],
    index=["genre"]
).round(2)

encoder = LabelEncoder()
X = np.concatenate([
    ratings["age"].to_numpy().reshape((-1, 1)),
    encoder.fit_transform(ratings["genre"]).reshape((-1, 1)),
    # encoder.fit_transform(ratings["gender"]).reshape((-1, 1))
], axis=1).astype(int)
y = ratings["pred_rating"].to_numpy()


encoder = LabelEncoder()
plt.style.use("seaborn")


fig, axs = plt.subplots(3, 3, constrained_layout=True, sharex=True, sharey=False)
for genre, ax in zip(ratings["genre"].unique(), axs.flat):
    df = ratings[ratings["genre"] == genre]
    X = np.concatenate([
        df["age"].to_numpy().reshape((-1, 1)),
        encoder.fit_transform(df["gender"]).reshape((-1, 1))
    ], axis=1).astype(int)
    y = df["pred_rating"].to_numpy()
    gam = LinearGAM(
        te(0, 1, n_splines=5, dtype=["numerical", "categorical"]),
        fit_intercept=False
    )
    gam.fit(X, y)
    term = gam.terms[0]



    XXF = np.concatenate(
        [np.arange(7, 74).reshape((-1, 1)), np.ones((67, 1))],
        axis=1
    ).astype(int)
    pdep, confi = gam.partial_dependence(term=0, X=XXF, width=0.95)
    ax.plot(XXF[:, 0], pdep, c="r", label="Female")
    ax.plot(XXF[:, 0], confi, c='r', ls='--', alpha=0.2)

    XXM = np.concatenate(
        [np.arange(7, 74).reshape((-1, 1)), np.zeros((67, 1))],
        axis=1
    ).astype(int)
    pdep, confi = gam.partial_dependence(term=0, X=XXM, width=0.95)
    ax.plot(XXM[:, 0], pdep, c="b", label="Male")
    ax.plot(XXM[:, 0], confi, c='b', ls='--', alpha=0.2)

    ax.hlines(0, xmin=7, xmax=73)

    ax.set_title(genre)

plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1,
            borderaxespad=0, frameon=False, title="Gender")
plt.show()