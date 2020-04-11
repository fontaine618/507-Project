import data
import models
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering, KMeans
from scipy import stats

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 1000)

plt.style.use("seaborn")

pd.options.mode.chained_assignment = None
train, test = data.load_train_test_ratings()
whole = pd.concat([train, test], axis=0)

svd = models.SVD(embed_dim=4, iter_nums=100)
svd.add_train_data(whole, "rating", ["user_id", "movie_id"])
svd.train()

axes = pd.plotting.scatter_matrix(pd.DataFrame(svd.U), alpha=0.2, figsize=(6, 6))
plt.suptitle("User clusters",
             fontsize=14, x=0.025, y=0.99,
             horizontalalignment='left', verticalalignment='top')
plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.savefig("../tex/Report/fig/users.pdf")
plt.show()

axes = pd.plotting.scatter_matrix(pd.DataFrame(svd.Vt.T), alpha=0.2, figsize=(6, 6))
plt.suptitle("Movie clusters",
             fontsize=14, x=0.025, y=0.99,
             horizontalalignment='left', verticalalignment='top')
plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.savefig("../tex/Report/fig/movies.pdf")
plt.show()


# user clusters
score = []
for k in range(2, 20):
    cluster_users = KMeans(k)
    cluster_users.fit(svd.U)
    score.append(cluster_users.inertia_)

plt.plot(range(2, 20), score)
plt.show()


n_user_clusters = 3
cluster_users = KMeans(n_user_clusters)
pred_users = cluster_users.fit_predict(svd.U)
pred_users = pd.DataFrame({"cluster": pred_users}, index=svd.pred_df.index)
axes = pd.plotting.scatter_matrix(
    pd.DataFrame(svd.U),
    alpha=1.0,
    c=pred_users["cluster"],
    figsize=(6, 6)
)
plt.suptitle("User embedding and clusters",
             fontsize=14, x=0.025, y=0.99,
             horizontalalignment='left', verticalalignment='top')
plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.savefig("../tex/Report/fig/user_clusters.pdf")
plt.show()




# movie clusters
score = []
for k in range(2, 20):
    cluster_movies = KMeans(k)
    cluster_movies.fit(svd.Vt.T)
    score.append(cluster_movies.inertia_)

plt.plot(range(2, 20), score)
plt.show()

n_movie_clusters = 4
cluster_movies = KMeans(n_movie_clusters)
pred_movies = cluster_movies.fit_predict(svd.Vt.T)
pred_movies = pd.DataFrame({"cluster": pred_movies}, index=svd.pred_df.columns)
axes = pd.plotting.scatter_matrix(
    pd.DataFrame(svd.Vt.T),
    alpha=1.0,
    c=pred_movies["cluster"],
    figsize=(6, 6)
)
plt.suptitle("Movie embedding and clusters",
             fontsize=14, x=0.025, y=0.99,
             horizontalalignment='left', verticalalignment='top')
plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.savefig("../tex/Report/fig/movie_clusters.pdf")
plt.show()


# cluster ratings compbinations
clusters_link = cluster_users.cluster_centers_.dot(np.diag(svd.sigma)).dot(cluster_movies.cluster_centers_.T)

pred_df = svd.pred_df.clip(1, 5)
pred_df["user_id"] = pred_df.index
pred = pd.melt(pred_df, id_vars="user_id", value_name="pred_rating")
pred = pred.merge(pred_users, how="left", on="user_id")
pred = pred.merge(pred_movies, how="left", on="movie_id")
clusters_link = pred.groupby(["cluster_x", "cluster_y"]).agg(
    {"pred_rating": "mean"}
).reset_index().pivot(
    index="cluster_x", columns="cluster_y", values="pred_rating"
)


plt.figure(figsize=(5, 3))
plt.imshow(clusters_link)
plt.yticks(ticks=range(n_user_clusters), labels=range(1, n_user_clusters + 1))
plt.ylabel("User cluster")
plt.xticks(ticks=range(n_movie_clusters), labels=range(1, n_movie_clusters + 1))
plt.xlabel("Movie cluster")
plt.grid(False)
plt.colorbar()
plt.title("Ratings for user-movie cluster combinations")
plt.tight_layout()
plt.savefig("../tex/Report/fig/user_movie_clusters.pdf")
plt.show()



movies = data.load_movies()
movies = movies[[g for g in movies.columns if g.startswith("genre_")]]
movies.columns = [g[6:] for g in movies.columns]
movies["movie_id"] = movies.index
movies = pd.melt(movies, id_vars="movie_id", value_name="genre01", var_name="genre")
movies = movies[movies["genre01"] == 1]
movies.set_index("movie_id", inplace=True)
movies.drop(columns="genre01", inplace=True)

df = movies.merge(pred_movies, how="left", on="movie_id", right_index=True).dropna(axis=0)
props = pd.crosstab(df["cluster"], df["genre"], normalize="columns").round(2).drop(columns="Unknown")*100
props[""] = 0
props["all"] = (df["cluster"].value_counts() / len(df)).round(2)*100


plt.figure(figsize=(8, 2.8))
plt.imshow(props)
plt.xticks(ticks=range(len(props.columns)), labels=props.columns, rotation="vertical")
plt.yticks(ticks=range(n_movie_clusters), labels=range(1, n_movie_clusters + 1))
plt.grid(False)
plt.colorbar()
plt.title("Cluster frequency per movie genre")
plt.tight_layout()
plt.savefig("../tex/Report/fig/movie_clusters_genre.pdf")
plt.show()




users = data.load_users()
users.drop(columns="zip_code", inplace=True)
users["age_binned"] = pd.cut(users["age"], bins=[5, 20, 35, 50, 75])
users["age_binned"].value_counts()

x = np.arange(7, 73, 0.1)
fig, axs = plt.subplots(2, 1, sharex=True)
for i, gender in enumerate(["M", "F"]):
    ax = axs[i]
    for k in range(cluster_users.n_clusters):
        dat = users["age"][(users["gender"] == gender) & (pred_users["cluster"] == k)]
        density = stats.kde.gaussian_kde(dat)
        ax.plot(x, density(x), label="Cluster {}".format(k+1))
    ax.legend(ncol=1, title="Male" if gender == "M" else "Female")
ax.set_xlabel("Age")
fig.suptitle("Cluster composition by gender and age",
             fontsize=14, x=0.025, y=0.99,
             horizontalalignment='left', verticalalignment='top')
fig.tight_layout()
fig.subplots_adjust(top=0.92)
plt.savefig("../tex/Report/fig/user_clusters_gender_age.pdf")
plt.show()

df = users.merge(pred_users, how="left", on="user_id", right_index=True).dropna(axis=0)

props = pd.crosstab(df["cluster"], df["gender"], normalize="columns").round(2)*100


props = pd.crosstab(df["cluster"], df["occupation"], normalize="columns").round(2)*100
props[""] = 0
props["all"] = (df["cluster"].value_counts() / len(df)).round(2)*100

plt.figure(figsize=(10, 2.5))
plt.imshow(props)
plt.xticks(ticks=range(len(props.columns)), labels=props.columns, rotation="vertical")
plt.yticks(ticks=range(n_user_clusters), labels=range(1, n_user_clusters + 1))
plt.grid(False)
plt.colorbar()
plt.title("Cluster frequency per user occupation")
plt.tight_layout()
plt.savefig("../tex/Report/fig/user_clusters_occupation.pdf")
plt.show()