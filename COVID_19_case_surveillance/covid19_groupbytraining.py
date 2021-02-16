import pandas as pd


pd.set_option('display.max_columns', 11)
df=pd.read_csv("data//COVID-19_Case_Surveillance_Public_Use_Data.csv",nrows=1000000)



def death_per_age_group():

    sex_group = df.groupby("sex")
    for name, group in sex_group:
        print("sex: {}".format(name))
        deaths_in_group = group[group["death_yn"] == "Yes"]
        number_of_deaths = len(deaths_in_group.index)
        print("number of deaths: {}".format(number_of_deaths))
        print("number in group: {}".format(len(group)))
        print("death ratio {}".format(number_of_deaths / len(group)))
        print()

def icu_per_sex_group():

    sex_group = df.groupby("sex")
    for name, group in sex_group:
        print("sex: {}".format(name))
        deaths_in_group = group[group["icu_yn"] == "Yes"]
        number_of_deaths = len(deaths_in_group.index)
        print("number of icu: {}".format(number_of_deaths))
        print("number in group: {}".format(len(group)))
        print("icu ratio {}".format(number_of_deaths / len(group)))
        print()


def death_per_ethnic():
    ethnic_group = df.groupby("Race and ethnicity (combined)")
    for name, group in ethnic_group:
        print("ethnic: {}".format(name))
        deaths_in_group = group[group["death_yn"] == "Yes"]
        number_of_deaths = len(deaths_in_group.index)
        print("number of deaths: {}".format(number_of_deaths))
        print("number in group: {}".format(len(group)))
        print("death ratio {}".format(number_of_deaths / len(group)))
        print()

def case_per_day():
    date_group = df.groupby("cdc_report_dt")
    for name, group in date_group:
        print("{} : {} ".format(name,len(group)))

def death_per_ethnic():
    ethnic_group = df.groupby("Race and ethnicity (combined)")
    for name, group in ethnic_group:
        print("ethnic: {}".format(name))
        deaths_in_group = group[group["death_yn"] == "Yes"]
        number_of_deaths = len(deaths_in_group.index)
        print("number of deaths: {}".format(number_of_deaths))
        print("number in group: {}".format(len(group)))
        print("death ratio {}".format(number_of_deaths / len(group)))
        print()

def death_per_icu():
    ethnic_group = df.groupby("icu_yn")
    for name, group in ethnic_group:
        print("icu: {}".format(name))
        deaths_in_group = group[group["death_yn"] == "Yes"]
        number_of_deaths = len(deaths_in_group.index)
        print("number of deaths: {}".format(number_of_deaths))
        print("number in group: {}".format(len(group)))
        print("death ratio {}".format(number_of_deaths / len(group)))
        print()

def death_per_ethnie_and_age():
    ethnic_group = df.groupby(["age_group","Race and ethnicity (combined)"])
    for name, group in ethnic_group:

        deaths_in_group = group[group["death_yn"] == "Yes"]
        number_of_deaths = len(deaths_in_group.index)
        print("{} : {}".format(name,number_of_deaths/len(group)))

def icu_per_ethnie_and_age():
    ethnic_group = df.groupby(["age_group","Race and ethnicity (combined)"])
    for name, group in ethnic_group:

        deaths_in_group = group[group["icu_yn"] == "Yes"]
        number_of_deaths = len(deaths_in_group.index)
        print("{} : {}".format(name,number_of_deaths/len(group)))



print("Overview of Deaths in sex each group:")
print()
death_per_age_group()

print("Overview of Icu in each sex group:")
print()
icu_per_sex_group()

print("Overview of Deaths in each ethnic group:")
print()
death_per_ethnic()

print("Overview of cases per day")
print()
case_per_day()
print()

print("Overview of death per icu")
print()
death_per_icu()


print("Overview death per age group and ethnic")
death_per_ethnie_and_age()

icu_per_ethnie_and_age()


