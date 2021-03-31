import pandas as pd
import re


def read_and_clean_genome_scores():
    genome_scores = pd.read_csv("data//genome_scores.csv")
    genome_scores.dropna()
    return genome_scores


def read_and_clean_genome_tags():
    genome_tags = pd.read_csv("data//genome_tags.csv")
    genome_tags.dropna()
    rename_dict = {genome_tags.columns.values[0]: "KindOfMovieID", genome_tags.columns.values[1]: "Kind Of Movie"}
    genome_tags.rename(mapper=rename_dict, axis=1, inplace=True)
    return genome_tags


def read_and_clean_link():
    link = pd.read_csv("data//link.csv")
    link.dropna()
    return link


def read_and_clean_movie():
    movie = pd.read_csv("data//movie.csv")
    movie.dropna()

    movie.loc[:, "year"] = movie.loc[:, "title"].apply(extract_year_and_name_of_title)
    movie.loc[:, "year"] = pd.to_datetime(movie.loc[:, "year"], format='%Y')
    return movie


def read_and_clean_ratings():
    ratings = pd.read_csv("data//rating.csv")
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
