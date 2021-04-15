import pandas as pd
import re


def read_and_clean_genome_scores(nrwosValue):
    genome_scores = pd.read_csv("data//genome_scores.csv", nrows=nrwosValue)
    genome_scores.dropna()
    return genome_scores


def read_and_clean_genome_tags(nrwosValue):
    genome_tags = pd.read_csv("data//genome_tags.csv", nrows=nrwosValue)
    genome_tags.dropna()
    rename_dict = {genome_tags.columns.values[0]: "KindOfMovieID", genome_tags.columns.values[1]: "Kind Of Movie"}
    genome_tags.rename(mapper=rename_dict, axis=1, inplace=True)
    return genome_tags


def read_and_clean_link(nrwosValue):
    link = pd.read_csv("data//link.csv", nrows=nrwosValue)
    link.dropna()
    return link


def read_and_clean_movie(nrwosValue):
    movies = pd.read_csv("data//movie.csv", nrows=nrwosValue)
    movies.dropna()

    movies.loc[:, "year"] = movies.loc[:, "title"].apply(extract_year_and_name_of_title)
    movies.loc[:, "year"] = pd.to_datetime(movies.loc[:, "year"], format='%Y')
    return movies


def read_and_clean_ratings(nrwosValue):
    ratings = pd.read_csv("data//rating.csv", nrows=nrwosValue)
    ratings.dropna()
    ratings.loc[:, "timestamp"] = pd.to_datetime(ratings.loc[:, "timestamp"])
    ratings.loc[:, "timestamp"] = ratings.loc[:, "timestamp"].dt.year
    ratings.loc[:, "timestamp"] = pd.to_datetime(ratings.loc[:, "timestamp"], format='%Y')
    return ratings


def extract_year_and_name_of_title(title):
    year = title.split("(")[-1].strip()
    year = re.sub("[^0-9]", "", year)

    try:
        int(year)
    except ValueError:
        return
    if int(year) > 2022 or int(year) < 1880:
        print("Not a valid year: {} {}".format(title, year))
        return
    return year

def extract_genres(movies):
    dummies_for_genre = movies.genres.str.get_dummies().astype(bool)
    movies_with_genres = movies.join(dummies_for_genre)
    movies_with_genres.drop("genres", inplace=True, axis=1)
    return movies_with_genres



def get_all_genres(movies):
    genres_list = movies.genres.str.split('|').to_list()
    genres_set = set()
    [genres_set.update(e) for e in genres_list]
    return genres_set

