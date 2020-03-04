import pandas as pd


def load_ratings():
	return pd.read_csv(
		"data/ml-100k/u.data",
		sep="\t",
		header=None,
		names=[
			"user_id", "movie_id", "rating", "timestamp"
		],
		index_col="user_id"
	)


def load_movies():
	return pd.read_csv(
		"data/ml-100k/u.item",
		sep="|",
		encoding="utf-8",
		header=None,
		names=[
			"movie_id", "movie_title", "release_date", "video_release_date",
			"IMDb URL", "unknown", "Action", "Adventure", "Animation",
			"Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
			"Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
			"Thriller", "War", "Western"
		],
		index_col="movie_id"
	)


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


def load_tag_relevance_wide():
	return pd.read_csv(
		"data/tag-genome/tag_relevance.dat",
		sep="\t",
		header=None,
		names=[
			"movie_id", "tag_id", "tag_relevance"
		]
	).pivot(
		index="movie_id",
		columns="tag_id",
		values="tag_relevance"
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
		names = [
			"movie_id", "movie_title", "movie_popularity"
		],
		index_col="movie_id"
	)
