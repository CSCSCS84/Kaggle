# some python pandas training
import pandas as pd


def analyse_data(data):
    print(covid_data.head())
    print(covid_data.loc[200])
    print()
    print(covid_data.iloc[201])
    # see the datatypes
    print(covid_data.dtypes)
    # check for is null values
    print(covid_data.isnull().sum())
    for col in covid_data.columns:
        print(covid_data[col].unique())


def clean_data(data):
    drop_unused_columns(data)
    change_datatypes(data)
    data = replace_values(data)
    make_death_col_categorial(data)
    return data


def make_death_col_categorial(data):
    data['death_yn'] = data['death_yn'].astype('category')
    data["death_yn"] = data["death_yn"].cat.codes
    print()


def replace_values(data):
    data = data.replace("Missing", "Unknown")
    data["death_yn"] = data["death_yn"].replace("Missing", "No")
    data["death_yn"] = data["death_yn"].replace("Unknown", "No")
    return data


def change_datatypes(data):
    data["cdc_report_dt"] = pd.to_datetime(covid_data["cdc_report_dt"])
    data["pos_spec_dt"] = pd.to_datetime(covid_data["pos_spec_dt"])


def drop_unused_columns(data):
    to_drop = ["onset_dt"]
    data.drop(to_drop, inplace=True, axis=1)


def read_data(nrwos):
    pd.set_option('display.max_columns', 11)
    data = pd.read_csv("data//COVID-19_Case_Surveillance_Public_Use_Data.csv", nrows=nrwos)
    return data


covid_data = read_data(nrwos=10000)
clean_data(covid_data)
print(covid_data.head())
