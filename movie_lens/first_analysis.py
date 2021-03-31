#Analysis of the movielens dataset from Kaggle
# https://www.kaggle.com/grouplens/movielens-20m-dataset
import pandas as pd
import numpy as np
import re
from timeit import default_timer
import seaborn as sns
from movie_lens import reader_and_cleaner as rc
from movie_lens import analyser as an

import matplotlib.pyplot as plt
start = default_timer()
#analyse the datasets


pd.options.display.max_columns=40







genome_scores=rc.read_genome_scores()
genome_tags=rc.read_genome_tags()

an.analyse_genome_scores(genome_scores)
an.analyse_genomeTags(genome_tags)

#----------------read  link data-----------------------------#
link=pd.read_csv("data//link.csv")
print(link.info(memory_usage="deep"))
print(link.head())
print(link.isna().sum())
link.dropna()

#---------------read movie data------------------------------#
def extract_year_and_name_of_title(title):
    year=title.split("(")[-1].strip()

    year=re.sub("[^0-9]", "", year)

    try:

        int(year)
    except ValueError:

        #print("Not a int: {} {}".format(title,year))
        return
    if int(year)>2022 or int(year)<1880:
        print("Not a valid year: {} {}".format(title, year))
        return
    return year
   # print("{} {}".format(year[0:-1],title))

movie=pd.read_csv("data//movie.csv")
print(movie.info(memory_usage="deep"))
print(movie.head(10))
print(movie.isna().sum())
movie.loc[:,"year"]=movie.loc[:,"title"].apply(extract_year_and_name_of_title)
movie.loc[:,"year"] = pd.to_datetime(movie.loc[:,"year"], format = '%Y')

print(movie.head())
link.dropna()

#------read ratings------------------#
print("--------------ratings---------------")
ratings=pd.read_csv("data//rating.csv")
print(ratings.head())
print(ratings.info(memory_usage="deep"))

print(ratings.isna().sum())
#ratings.loc[:,"timestamp"] = pd.to_datetime(ratings.loc[:,"timestamp"], format = '%Y')
#ratings.timestamp = pd.to_datetime(ratings.timestamp)
#ratings.timestamp = ratings.timestamp.dt.year
ratings.loc[:,"timestamp"] = pd.to_datetime(ratings.loc[:,"timestamp"])
ratings.loc[:,"timestamp"] = ratings.loc[:,"timestamp"].dt.year
ratings.loc[:,"timestamp"] = pd.to_datetime(ratings.loc[:,"timestamp"], format = '%Y')
print(ratings.head(10))
end=default_timer()
print(end-start)

#was tun:

#Explore the data with some basic plots
#number of movies and ratings per year:
#year muss nach timestamp gecastet werden

print(movie.isna().sum())
movie.groupby(by="year")
sum_of_year=movie.groupby(by="year").count()
ratings_of_year=ratings.groupby(by="timestamp").count()
print(sum_of_year)
col=sum_of_year.columns[0]
col2=ratings_of_year.columns[1]
print(sum_of_year.info())
fig, ax1 = plt.subplots(figsize=(10,5))
g=sns.lineplot(x=sum_of_year.index, y=sum_of_year.loc[:,col])

ax2= ax1.twinx()
g2=sns.lineplot(x=ratings_of_year.index, y=ratings_of_year.loc[:,col2],ax=ax2,color="green")
print(type(g))
g.set(xlabel='Year', ylabel='Number of Movies released',title="Number of Movies per Year")

g2.set(xlabel='Year', ylabel='Number of Movies released',title="Number of Movies per Year")
print(ratings_of_year)
plt.show()
print(type(ratings_of_year.index))
print(type(sum_of_year.index))