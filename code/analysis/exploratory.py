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
], axis=1)
y = ratings["pred_rating"].to_numpy()

gam = LinearGAM(
    te(0, 1, n_splines=5, dtype=["numerical", "categorical"])
)
gam.fit(X, y)
gam.summary()

for i, term in enumerate(gam.terms):
    if term.isintercept:
        continue

    XX = gam.generate_X_grid(term=i)
    pdep, confi = gam.partial_dependence(term=i, X=XX, width=0.95)

    plt.figure()
    plt.plot(XX[:, term.feature], pdep)
    plt.plot(XX[:, term.feature], confi, c='r', ls='--')
    plt.title(repr(term))
    plt.show()