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
nrows = 10000
# nrows = sys.maxsize
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


# analyse_datasets()

# number_of_movies_and_ratings()
# plt.show()

genres = rc.get_all_genres(movies)
movies = rc.extract_genres(movies)


# plot_cumsum_of_movies_and_genres(movies)
# plot_movies_per_genre(movies)

def analyse_ratings(movies_with_ratings):
    global ax1
    fig, ax1 = plt.subplots(figsize=(10, 5))
    sns.displot(ratings, x="rating", binwidth=0.5, ax=ax1)

    genres_ax = ax1.twinx()
    for genre in genres:
        ratings_for_genre = movies_with_ratings.loc[movies_with_ratings.loc[:, genre], :]
        sns.kdeplot(data=ratings_for_genre, x="rating", ax=genres_ax, label=genre).set_title("Ratings per genre")
    plt.xlabel('Ratings', fontsize=12)
    genres_ax.legend(loc="upper left", ncol=2)
    plt.show()


# analyse ratings
print(movies.info())
print(ratings.info())
movies_with_ratings = pd.merge(movies, ratings, left_index=True, right_index=True, how="inner")
mapper_columns = {"movieId_x": "movieId"}
movies_with_ratings.rename(columns=mapper_columns, inplace=True)
print("movies with ratings")
print(movies_with_ratings.info())


# some basic statistic per genre
def calc_rating_statistics():
    ratings_statistic = pd.DataFrame(columns=["average_rating", "std", "num_of_ratings", "ratings_mean"])
    for genre in genres:
        genre_ratings = movies_with_ratings.loc[movies_with_ratings.loc[:, genre], :]

        ratings_statistic.loc[genre, "average_rating"] = genre_ratings.rating.mean()
        ratings_statistic.loc[genre, "std"] = genre_ratings.rating.std()
        ratings_statistic.loc[genre, "num_of_ratings"] = genre_ratings.shape[0]
    ratings_statistic["ratings_mean"] = ratings_statistic.average_rating.mean()
    print(ratings_statistic)
    return ratings_statistic


# plot basic rating statistics
def basic_statistic_for_ratings():
    global ax1

    ratings_statistic = calc_rating_statistics()

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ratings_statistic.loc[:, ['average_rating', 'std']].plot(kind='bar', color=['b', 'r'], grid=False, ax=ax1)
    plt.title("Movie rating descriptive stats")
    plt.xlabel('Genre', fontsize=12)
    ratings_statistic.loc[:, ['ratings_mean']].plot(kind='line', style="--", color="black", grid=False, ax=ax1)
    ax1.legend(loc="center", ncol=1)
    plt.xticks(rotation='vertical')

    plt.gcf().subplots_adjust(bottom=0.30)
    ax2 = ax1.twinx()
    ratings_statistic.loc[:, ['num_of_ratings']].plot(kind='line', color="black", grid=False, ax=ax2)
    ax2.legend(loc="center right", ncol=1)

    f, ax1 = plt.subplots(figsize=(18, 12))
    ratings_statistic.loc[:, ["num_of_ratings"]].plot.pie(y="num_of_ratings", ax=ax1, title="Number of Ratings")
    plt.ylabel('', fontsize=12)
    ax1.legend(loc=(-0.7, 0), labels=ratings_statistic.index)

    plt.show()


# basic_statistic_for_ratings()


ratings_mean = ratings.groupby(["movieId"]).mean()
ratings_per_movie = pd.merge(movies, ratings_mean, right_on="movieId", left_on="movieId")


# ratings per movie average
def ratings_per_movie_average():
    plt.plot(ratings_per_movie.year, ratings_per_movie.rating, "g.", markersize=4)
    plt.show()


# ratings_per_movie_average()

# average rating for each year and genrex
def average_rating_for_each_year_and_genre():
    count = 1
    plt.subplots(figsize=(20, 10))
    for genre in genres:
        ratings_in_genre = ratings_per_movie.loc[ratings_per_movie.loc[:, genre], :]
        sns.lineplot(x=ratings_in_genre.year, y=ratings_in_genre.rating, ci=None, label=genre)
        name_plot()
        if count % 5 == 0:
            sns.lineplot(x=ratings_per_movie.year, y=ratings_per_movie.rating, ci=None, color="black", lw=4)

            plt.show()
            plt.subplots(figsize=(20, 10))
        count = count + 1
    name_plot()
    plt.show()


def name_plot():
    plt.xlabel('Year')
    plt.ylabel('Average Rating')
    plt.title('Average rating per year per genre')
    plt.legend(loc=(0, 0), ncol=2)


average_rating_for_each_year_and_genre()
