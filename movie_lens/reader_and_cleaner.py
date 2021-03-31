import pandas as pd
import numpy as np
import re


def read_genome_scores():
    genome_scores=pd.read_csv("data//genome_scores.csv", nrows=100)
    return genome_scores

def read_genome_tags():
    genome_tags = pd.read_csv("data//genome_tags.csv")
    rename_dict = {genome_tags.columns.values[0]: "KindOfMovieID", genome_tags.columns.values[1]: "Kind Of Movie"}

    genome_tags.rename(mapper=rename_dict, axis=1, inplace=True)
    return genome_tags