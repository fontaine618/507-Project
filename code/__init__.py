import pandas as pd
import numpy as np
from data import *




movies = load_movies()
movies_genome = load_movies_genome()





def find_index_in_movies_genome(title: str):
	if title in movies_genome["movie_title"].values:
		return (movies_genome["movie_title"] == title).idxmax()
	return None


movie_to_movies_genome = {
	i: find_index_in_movies_genome(title)
	for i, title in movies["movie_title"].items()
}


missing = {
	id: movies["movie_title"][id]
	for id, id_genome in movie_to_movies_genome.items()
	if id_genome is None
}

for id, title in missing.items():
	print(id, "------", title)

print(np.array([1 if id_genome is None else 0 for id_genome in movie_to_movies_genome.values()]).sum())




