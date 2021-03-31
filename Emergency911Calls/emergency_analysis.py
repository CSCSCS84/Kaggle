# Milestone project of Udemy Data Science cource, not cleaned or structured
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/911.csv")
df["Reason"] = df["title"].apply(lambda x: x.split(":")[0])


def print_some_basic_informations():
    print(df.info())
    print(df.head(1))
    print("top 5 zipcodes")
    print(df["zip"].value_counts(ascending=False).head())
    print("unique titles")
    print(len(df["title"].unique()))
    print(df.columns)
    print(df.head())
    print("Most common reasons for call")
    print(df["Reason"].value_counts(ascending=False).head())


def countplot_reason():
    sns.countplot(df["Reason"])
    plt.show()


def cleaning_timestamp():
    df["timeStamp"] = pd.to_datetime(df["timeStamp"])
    print(df["timeStamp"])
    print(df.dtypes)
    df["Hour"] = df["timeStamp"].apply(lambda x: x.hour)
    df["Month"] = df["timeStamp"].apply(lambda x: x.month)
    df["WeekDay"] = df["timeStamp"].apply(lambda x: x.dayofweek)
    dmap = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
    df["WeekDay"] = df["WeekDay"].map(dmap)


print_some_basic_informations()
# countplot_reason()
cleaning_timestamp()
print(df.head())


# sns.countplot(x="WeekDay", data=df, hue="Reason")


def countplot_month():
    sns.countplot(x="Month", data=df, hue="Reason")
    plt.legend(bbox_to_anchor=(1.02, 1.0), loc=2, borderaxespad=0.)


# countplot_month()

# some month are missing

month_group = df.groupby("Month").count()
print(month_group.head())
print(month_group.index)
# month_group["Hour"].plot()

# month_group.index=month_group.reset_index()
# print(month_group.index)

# sns.lmplot(x="Month", y="Hour", data=month_group.reset_index())
# print(month_group.index)

df["Date"] = df["timeStamp"].apply(lambda x: x.date())
date_group = df.groupby("Date").count()
# date_group["zip"].plot(lw=1)
# plt.show()


date_ems = df[df["Reason"] == "EMS"]
date_ems_group = date_ems.groupby(["Date", "Reason"]).count()
# date_ems_group["zip"].plot()
# plt.show()
m# print(date_ems_group.head())

date_hour_group = df.groupby(["WeekDay", "Hour"]).count()["Reason"].unstack(level=-1)
# sns.heatmap(date_hour_group)
# sns.clustermap(date_hour_group)
# plt.figure(figsize=(12,6))
# plt.show()
print(date_hour_group.head())

# heatmap the monts
date_month_group = df.groupby(["WeekDay", "Month"]).count()["Reason"].unstack(level=-1)
sns.heatmap(date_month_group)
sns.clustermap(date_month_group)
plt.figure(figsize=(12, 6))
plt.show()
