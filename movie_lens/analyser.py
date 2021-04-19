import pandas as pd


def analyse_genome_scores(genome_scores):
    print(genome_scores.info(memory_usage="deep"))
    print(genome_scores.head(20))
    duplicated = genome_scores.duplicated()
    print(duplicated.sum())
    print(genome_scores.isna().sum())
    print(genome_scores.describe())
    tag_id = genome_scores.loc[:, "tagId"]
    print(tag_id.is_unique)


def analyse_genomeTags(genome_tags):
    print(genome_tags.info())
    print(genome_tags.describe())
    print(genome_tags.columns.values)
    print(genome_tags.columns)
    print(genome_tags.isna().sum())
    print(genome_tags.head(10))


def analyse_link(link):
    print(link.info(memory_usage="deep"))
    print(link.head())
    print(link.isna().sum())


def analyse_movie(movies):
    print(movies.info(memory_usage="deep"))
    print(movies.head(10))
    print(movies.isna().sum())
    print(movies.head())


def analyse_ratings(ratings):
    print(ratings.head())
    print(ratings.info(memory_usage="deep"))
    print(ratings.isna().sum())
    print(ratings.head(10))
    print(ratings.rating.value_counts())


def group_by_year_and_genre(movies, genres):
    year_and_genre = pd.DataFrame(index=movies["year"].unique())
    year_and_genre.sort_index(inplace=True)
    for genre in genres:
        year_and_genre[genre] = movies.loc[:, ["year", genre]].groupby(["year"]).sum()
        year_and_genre.loc[:, genre] = year_and_genre.loc[:, genre].cumsum()
    return year_and_genre


def cumsum_of_movies(movies):
    cumsum_of_movies = pd.DataFrame(index=movies["year"].unique())
    cumsum_of_movies.sort_index(inplace=True)
    cumsum_of_movies["cumsum of movies"] = movies.groupby("year").count().loc[:, "movieId"]
    cumsum_of_movies.loc[:, "cumsum of movies"] = cumsum_of_movies.loc[:, "cumsum of movies"].cumsum()
    return cumsum_of_movies






