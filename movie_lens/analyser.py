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


def analyse_movie(movie):
    print(movie.info(memory_usage="deep"))
    print(movie.head(10))
    print(movie.isna().sum())
    print(movie.head())


def analyse_ratings(ratings):
    print(ratings.head())
    print(ratings.info(memory_usage="deep"))
    print(ratings.isna().sum())
    print(ratings.head(10))
