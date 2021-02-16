#first seaborn scripts
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import covid19_cleaning as cc



covid_data = cc.read_data(nrwos=1000000)
covid_data = cc.clean_data(covid_data)

sns.boxplot(x="age_group", y="death_yn", data=covid_data, palette='rainbow')
sns.countplot(covid_data['age_group'], hue='death_yn', data=covid_data)

plt.show()
