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
    movie = pd.read_csv("data//movie.csv", nrows=nrwosValue)
    movie.dropna()

    movie.loc[:, "year"] = movie.loc[:, "title"].apply(extract_year_and_name_of_title)
    movie.loc[:, "year"] = pd.to_datetime(movie.loc[:, "year"], format='%Y')
    return movie


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


def extract_genre_from_row(genres_from_movie, genres):
    genres_from_row = genres_from_movie.split("|")
    print(genres_from_row)
    genres.update(genres_from_row)


def get_all_genres(movies):
    genres_list = movies.genres.str.split('|').to_list()
    s = set()
    [s.update(e) for e in genres_list]
    return s


def extract_genres(movies):
    dummies = movies.genres.str.get_dummies().astype(bool)

    # print(dummies)
    # print(type(dummies))
    movies = movies.join(dummies)
    # print(movies.head(10))
    movies.drop("genres", inplace=True, axis=1)
    # print(movies)
    return movies
    # movies.loc[:,"genres"].apply(extract_genre_from_row,genres=genres_set)

    # return s


def group_by_year_and_genre(movies, genres):
    df = pd.DataFrame(index=movies["year"].unique())
    df.sort_index(inplace=True)
    print(df.info())
    # grouped_movies=movies.groupyby(["year"])
    print(movies.info())
    for genre in genres:
        df[genre]=movies.loc[:, ["year", genre]].groupby(["year"]).sum()
        df.loc[:,genre]=df.loc[:,genre].cumsum()
        # print(movies[genre])
    return df
    #print(df.head())

def cumsum_of_movies(movies):
    cumsum_of_movies=pd.DataFrame(index=movies["year"].unique())
    cumsum_of_movies.sort_index(inplace=True)
    #cumsum_of_movies["movies_per_year"]=
    #print("movies per year")
    #print(movies.groupby("year").count())
    cumsum_of_movies["cumsum of movies"]=movies.groupby("year").count().loc[:,"movieId"]
    cumsum_of_movies.loc[:, "cumsum of movies"] = cumsum_of_movies.loc[:, "cumsum of movies"].cumsum()
    return cumsum_of_movies