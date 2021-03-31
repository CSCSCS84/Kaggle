import pandas as pd
import numpy as np
import re

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