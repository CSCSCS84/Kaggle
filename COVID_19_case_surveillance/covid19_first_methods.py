#some first scripts
import pandas as pd


pd.set_option('display.max_columns', 11)
df = pd.read_csv("data//COVID-19_Case_Surveillance_Public_Use_Data.csv",nrows=1000)

print("information of dataframe")
print(df.info())
print()
#print(df.isnull())

print("all columns")
print(df.columns)

print()
print("values of columns")
for col in df.columns:
    print()
    print(df[col].value_counts())
    df[col].isnull().sum()

print("")
print("check data for null values")
#check data for isnull
for col in df.columns:
    print(col)
    print(df[col].isnull().sum())

#groupby
print("group by age_group")
age_groups = df.groupby("age_group")

print("Infected person per age_group")
for name, group in age_groups:
    print("{}: {} ".format(name,len(group)))

print()
print("group by Race and ethnicity")
race_group = df.groupby("Race and ethnicity (combined)")

print("Infected person per Race and ethnicity")
for name, group in race_group:
    print("{}: {} ".format(name,len(group)))


print()
print("group by cdc_report_dt")
cdc_report_dt_group = df.groupby("cdc_report_dt")

print("Infected person per cdc_report_dt")
for name, group in cdc_report_dt_group:
    print("{}: {} ".format(name,len(group)))


print()
print("group by SEX and death")
sex_group=df.groupby(["sex"]).agg("count")
sex_and_death_group=df.groupby(["sex","death_yn"]).agg("sum")
print("Infected person per sex")

for group in sex_group:
    print(" {}:{} ".format(2,len(group)))



sex_group=df["sex"].value_counts()
print(sex_group)