import pandas as pd
import numpy as np


def load_ratings():
	return pd.read_csv(
		"data/ml-100k/u.data",
		sep="\t",
		header=None,
		names=[
			"user_id", "movie_id", "rating", "timestamp"
		]
	)


def load_movies():
	return pd.read_csv(
		"data/ml-100k/u.item",
		sep="|",
		encoding="utf-8",
		header=None,
		names=[
			"movie_id", "movie_title", "release_date", "video_release_date",
			"IMDb URL", "genre_unknown", "genre_Action", "genre_Adventure", "genre_Animation",
			"genre_Children's", "genre_Comedy", "genre_Crime", "genre_Documentary", "genre_Drama", "genre_Fantasy",
			"genre_Film-Noir", "genre_Horror", "genre_Musical", "genre_Mystery", "genre_Romance", "genre_Sci-Fi",
			"genre_Thriller", "genre_War", "genre_Western"
		],
		index_col="movie_id"
	).drop(columns="video_release_date")


def load_users():
	return pd.read_csv(
		"data/ml-100k/u.user",
		sep="|",
		header=None,
		names=[
			"user_id", "age", "gender", "occupation", "zip_code"
		],
		index_col="user_id"
	)


def load_users_wide():
	users = load_users()
	occupations_one_hot = pd.get_dummies(users["occupation"], prefix="occupation")
	gender_one_hot = pd.get_dummies(users["gender"], prefix="gender")
	users = users.drop(columns=["occupation", "gender", "zip_code"]).join(
		occupations_one_hot
	).join(
		gender_one_hot
	)
	return users


def load_tag_relevance_wide():
	df = pd.read_csv(
		"data/tag-genome/tag_relevance.dat",
		sep="\t",
		header=None,
		names=[
			"movies_genome_id", "tag_id", "tag_relevance"
		]
	).pivot(
		index="movies_genome_id",
		columns="tag_id",
		values="tag_relevance"
	)
	df.columns = ["tag_"+tag for tag in load_tag_name()["tag"]]
	return df


def load_tag_name():
	return pd.read_csv(
		"data/tag-genome/tags.dat",
		sep="\t",
		header=None,
		names=[
			"tag_id", "tag", "tag_popularity"
		],
		index_col="tag_id"
	)


def load_tag_relevance_long():
	return pd.read_csv(
		"data/tag-genome/tag_relevance.dat",
		sep="\t",
		header=None,
		names=[
			"movie_id", "tag_id", "tag_relevance"
		],
		index_col=["movie_id", "tag_id"]
	)


def load_tags():
	return pd.read_csv(
		"data/tag-genome/tags.dat",
		sep="\t",
		header=None,
		names=[
			"tag_id", "tag", "tag_popularity"
		],
		index_col="tag_id"
	)


def load_movies_genome():
	return pd.read_csv(
		"data/tag-genome/movies-renamed.dat",
		sep="\t",
		header=None,
		names=[
			"movie_id", "movie_title", "movie_popularity"
		],
		index_col="movie_id"
	)


def find_index_in_movies_genome(title: str, movies_genome: pd.DataFrame):
	if title in movies_genome["movie_title"].values:
		return (movies_genome["movie_title"] == title).idxmax()
	return None


def movie_to_movies_genome(movies: pd.DataFrame, movies_genome: pd.DataFrame):
	mtmg_dict = {
		i: find_index_in_movies_genome(title, movies_genome)
		for i, title in movies["movie_title"].items()
	}
	return pd.DataFrame(mtmg_dict.values(), index=mtmg_dict.keys(), dtype=int, columns=["movies_genome_id"])


def load_movies_with_genome_id(drop=True):
	movies = load_movies()
	movies_genome = load_movies_genome()
	mtmg_dict = movie_to_movies_genome(movies, movies_genome)
	movies = movies.join(mtmg_dict)
	if drop:
		movies = movies[movies["movies_genome_id"] > 0]
	return movies


def load_movies_with_tags():
	movies = load_movies_with_genome_id()
	movies["movie_id"] = movies.index
	tags = load_tag_relevance_wide()
	movies = pd.merge(movies, tags, on='movies_genome_id', how='left')
	movies.index = movies["movie_id"]
	return movies.drop(columns=["movies_genome_id", "movie_id"])


def load_ratings_with_tags_movies_and_user_info():
	"""Returns the dataset with complete entries.

	Notes
	-----
	Columns are as follows:
	- Index is a dummy index.
	- user_id
	- movie_id is the id from the 100K dataset, not from genome dataset (see movie_to_movies_genome for the mapping)
	- rating 1-5
	- timestamp is the time of the rating
	- Genres (18) named genre_<genre>
	- Tags (1127) named tag_<tag>
	- age
	- occupation in one-hot encoding (20) named occupation_<occupation>
	- gender F/M in one-hot encoding (2) named genre_<genre>
	"""
	movies = load_movies_with_tags()
	users = load_users_wide()
	ratings = load_ratings()
	ratings_with_movies = pd.merge(ratings, movies, how="left", on="movie_id").drop(columns=[
		"movie_title", "release_date", "IMDb URL",
	])
	ratings_with_movies_and_users = pd.merge(ratings_with_movies, users, how="left", on="user_id")
	ratings_with_movies_and_users.dropna(axis='index', inplace=True)
	ratings_with_movies_and_users.index.name = "rating_id"
	return ratings_with_movies_and_users


def load_train_test_and_feature_list(test_pct=0.25, n_folds=5, seed=1):
	ratings = load_ratings_with_tags_movies_and_user_info()
	np.random.seed(seed)
	n = len(ratings)
	n_test = int(n * test_pct)
	test_id = np.random.choice(ratings.index, n_test, replace=False)
	ratings["test"] = [i in test_id for i in ratings.index]
	ratings["fold_id"] = np.random.randint(1, n_folds + 1, n)
	ratings = ratings.groupby("test")
	train = ratings.get_group(0)
	test = ratings.get_group(1)
	features = train.columns.tolist()
	for colname in ["user_id", "movie_id", "rating", "timestamp", "test", "fold_id"]:
		features.remove(colname)
	return train, test, features
