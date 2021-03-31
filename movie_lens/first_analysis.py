# Analysis of the movielens dataset from Kaggle
# https://www.kaggle.com/grouplens/movielens-20m-dataset
import pandas as pd
from timeit import default_timer
import seaborn as sns
from movie_lens import reader_and_cleaner as rc
from movie_lens import analyser as an
import matplotlib.pyplot as plt

start = default_timer()
pd.options.display.max_columns = 40

genome_scores = rc.read_and_clean_genome_scores()
genome_tags = rc.read_and_clean_genome_tags()
link = rc.read_and_clean_link()
movie = rc.read_and_clean_movie()
ratings = rc.read_and_clean_ratings()

an.analyse_genome_scores(genome_scores)
an.analyse_genomeTags(genome_tags)
an.analyse_genomeTags(link)
an.analyse_movie(movie)
an.analyse_ratings(ratings)

# Explore the data with some basic plots
# number of movies and ratings per year:
# year muss nach timestamp gecastet werden
movies_per_year = movie.groupby(by="year").count()
ratings_per_year = ratings.groupby(by="timestamp").count()

fig, ax1 = plt.subplots(figsize=(10, 5))

col = movies_per_year.columns[0]
g = sns.lineplot(x=movies_per_year.index, y=movies_per_year.loc[:, col])
g.set(xlabel='Year', ylabel='Number of Movies released', title="Number of Movies per Year")

col2 = ratings_per_year.columns[1]
ax2 = ax1.twinx()
g2 = sns.lineplot(x=ratings_per_year.index, y=ratings_per_year.loc[:, col2], ax=ax2, color="green")
g2.set(xlabel='Year', ylabel='Number of Ratings')

end = default_timer()
print(end - start)
plt.show()
