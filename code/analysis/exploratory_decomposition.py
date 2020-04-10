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

axes = pd.plotting.scatter_matrix(pd.DataFrame(svd.U), alpha=0.2)
plt.tight_layout()
plt.show()

axes = pd.plotting.scatter_matrix(pd.DataFrame(svd.Vt.T), alpha=0.2)
plt.tight_layout()
plt.show()



score = []
for k in range(2, 20):
    cluster_users = KMeans(k)
    cluster_users.fit(svd.U)
    score.append(cluster_users.inertia_)

plt.plot(range(2, 20), score)
plt.show()
# clusters


cluster_users = KMeans(3)
pred_users = cluster_users.fit_predict(svd.U)
axes = pd.plotting.scatter_matrix(
    pd.DataFrame(svd.U),
    alpha=1.0,
    c=pred_users
)
plt.tight_layout()
plt.show()





score = []
for k in range(2, 20):
    cluster_movies = KMeans(k)
    cluster_movies.fit(svd.Vt.T)
    score.append(cluster_movies.inertia_)

plt.plot(range(2, 20), score)
plt.show()
# clusters


cluster_movies = KMeans(3)
pred_movies = cluster_movies.fit_predict(svd.Vt.T)
pred_movies = pd.DataFrame({"cluster": pred_movies}, index=svd.pred_df.columns)
axes = pd.plotting.scatter_matrix(
    pd.DataFrame(svd.Vt.T),
    alpha=1.0,
    c=pred_movies["cluster"]
)
plt.tight_layout()
plt.show()




clusters_link = cluster_users.cluster_centers_.dot(np.diag(svd.sigma)).dot(cluster_movies.cluster_centers_.T)
plt.imshow(clusters_link)
plt.yticks(ticks=range(3), labels=range(1, 4))
plt.ylabel("User cluster")
plt.xticks(ticks=range(3), labels=range(1, 4))
plt.xlabel("Movie cluster")
plt.grid(False)
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


props = pd.crosstab(df["genre"], df["cluster"], normalize="index").round(2).drop(index="Unknown")*100

plt.imshow(props)
plt.yticks(ticks=range(0, 18), labels=props.index)
plt.xticks(ticks=range(3), labels=range(1, 4))
plt.grid(False)
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
        dat = users["age"][(users["gender"] == gender) & (pred_users == k)]
        density = stats.kde.gaussian_kde(dat)
        ax.plot(x, density(x),
                 color=["red", "green", "blue"][k],
                 label="Cluster {} ({})".format(k+1,gender)
                 )
    ax.legend(ncol=1)
plt.tight_layout()
plt.show()