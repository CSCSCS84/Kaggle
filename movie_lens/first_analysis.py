# Analysis of the movielens dataset from Kaggle
# https://www.kaggle.com/grouplens/movielens-20m-dataset
import pandas as pd
from timeit import default_timer
import seaborn as sns

import movie_lens.analyser
from movie_lens import reader_and_cleaner as rc
from movie_lens import analyser as an
import matplotlib.pyplot as plt
import sys

start = default_timer()
pd.options.display.max_columns = 40
#nrows = 10000
nrows=sys.maxsize
genome_scores = rc.read_and_clean_genome_scores(nrows)
genome_tags = rc.read_and_clean_genome_tags(nrows)
link = rc.read_and_clean_link(nrows)
movies = rc.read_and_clean_movie(nrows)
ratings = rc.read_and_clean_ratings(nrows)


def analyse_datasets():
    an.analyse_genome_scores(genome_scores)
    an.analyse_genomeTags(genome_tags)
    an.analyse_genomeTags(link)
    an.analyse_movie(movies)
    an.analyse_ratings(ratings)


# Explore the data with some basic plots
# number of movies and ratings per year:
def number_of_movies_and_ratings():
    movies_per_year = movies.groupby(by="year").count()
    ratings_per_year = ratings.groupby(by="timestamp").count()
    fig, ax1 = plt.subplots(figsize=(10, 5))
    col = movies_per_year.columns[0]
    g = sns.lineplot(x=movies_per_year.index, y=movies_per_year.loc[:, col])
    g.set(xlabel='Year', ylabel='Number of Movies released', title="Number of Movies per Year")
    col2 = ratings_per_year.columns[1]
    ax2 = ax1.twinx()
    g2 = sns.lineplot(x=ratings_per_year.index, y=ratings_per_year.loc[:, col2], ax=ax2, color="green")
    g2.set(xlabel='Year', ylabel='Number of Ratings')


# Plot Cumulative number of movies, in total and per genre.
def plot_cumsum_of_movies_and_genres(movies):
    plt.style.use('seaborn')



    genres_cumsum = movie_lens.analyser.group_by_year_and_genre(movies, genres)
    genres_cumsum.plot.area(stacked=True, figsize=(10, 5))
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Cumulative number of movies-genre', fontsize=12)
    plt.title('Total movies-genre', fontsize=17)
    plt.legend(loc="upper left", ncol=2)

    movies_cumsum = movie_lens.analyser.cumsum_of_movies(movies)
    plt.plot(movies_cumsum.loc[:, "cumsum of movies"], marker='o', markerfacecolor='black', markersize=5,
             label="CumSum of Movies")
    plt.show()


# Plot movies per genre tag
def plot_movies_per_genre(movies):
    movies_per_genre = movies.iloc[:, 3:].sum()
    sns.barplot(x=movies_per_genre.index, y=movies_per_genre.values)
    plt.gcf().subplots_adjust(bottom=0.25)
    plt.xticks(rotation='vertical')
    plt.xlabel('Genre', fontsize=12)
    plt.ylabel('Number of movies tagged', fontsize=12)
    plt.title('Movies per genre tag', fontsize=17)
    plt.show()


analyse_datasets()

genres = rc.get_all_genres(movies)
movies = rc.extract_genres(movies)

plot_cumsum_of_movies_and_genres(movies)
plot_movies_per_genre(movies)
