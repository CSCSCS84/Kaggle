# Analysis of the movielens dataset from Kaggle
# https://www.kaggle.com/grouplens/movielens-20m-dataset
import pandas as pd
from timeit import default_timer
import seaborn as sns
from movie_lens import reader_and_cleaner as rc
from movie_lens import analyser as an
import matplotlib.pyplot as plt
import sys

start = default_timer()
pd.options.display.max_columns = 40
#nrows = 100000
nrows=sys.maxsize
genome_scores = rc.read_and_clean_genome_scores(nrows)
genome_tags = rc.read_and_clean_genome_tags(nrows)
link = rc.read_and_clean_link(nrows)
movie = rc.read_and_clean_movie(nrows)
ratings = rc.read_and_clean_ratings(nrows)


def analyse_datasets():
    an.analyse_genome_scores(genome_scores)
    an.analyse_genomeTags(genome_tags)
    an.analyse_genomeTags(link)
    an.analyse_movie(movie)
    an.analyse_ratings(ratings)


analyse_datasets()
genres = rc.get_all_genres(movie)
cumsum_of_movies=rc.cumsum_of_movies(movie)
movie = rc.extract_genres(movie)
genres_cumsum = rc.group_by_year_and_genre(movie, genres)

print(cumsum_of_movies.tail(10))


def number_of_movies_and_ratings():
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


# Explore the data with some basic plots
# number of movies and ratings per year:
# year muss nach timestamp gecastet werden

# number_of_movies_and_ratings()

# cumsum of genre per year
# rc.group_by_year_and_genre(movie,genres)

end = default_timer()
print(end - start)
plt.style.use('seaborn')
print(cumsum_of_movies.info())
genres_cumsum.plot.area(stacked=True,figsize=(10,5))
print(cumsum_of_movies.head(10))
plt.plot(cumsum_of_movies.loc[:,"cumsum of movies"],marker='o', markerfacecolor='black',markersize=5,label="CumSum of Movies")


plt.xlabel('Year', fontsize=12)
plt.ylabel('Cumulative number of movies-genre', fontsize=12)
plt.title('Total movies-genre',fontsize=17)
plt.legend(loc="upper left", ncol=2)

#movies per genre
print(movie.info())
movies_per_genre=movie.iloc[:,3:].sum()
print("movies per genre")
print(type(movies_per_genre))
print(movies_per_genre)
print(movies_per_genre.describe())
#print(movies_per_genre)
plt.show()
sns.barplot(x=movies_per_genre.index,y=movies_per_genre.values)
#make room for the labels
plt.gcf().subplots_adjust(bottom=0.25)
plt.xticks(rotation='vertical')
plt.xlabel('Genre', fontsize=12)
plt.ylabel('Number of movies tagged', fontsize=12)
plt.title('Movies per genre tag',fontsize=17)
plt.show()
