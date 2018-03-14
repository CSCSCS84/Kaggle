import numpy
import pandas
import matplotlib.pyplot as plt
import seaborn as sns


def plotNumerical(feature):
    sns.set_style("whitegrid")
    sns.distplot(train_data[feature].dropna(), hist=False, color='blue');

    plt.ylabel('Density', fontsize=14);
    sns.distplot(train_data.loc[train_data.Survived == 0, feature].dropna(), color='red',
                 hist_kws=dict(edgecolor=None, linewidth=0), hist=False)


    sns.distplot(test_data[feature].dropna(), hist=False, color='black');
    plt.ylabel('Density', fontsize=14);
    sns.distplot(test_data.loc[test_data.Survived == 0, feature].dropna(), color='purple',
                 hist_kws=dict(edgecolor=None, linewidth=0), hist=False)
    plt.show();

train_data = pandas.read_csv('Input/train.csv',index_col='PassengerId')
print(train_data)
train_data['target_name'] = train_data['Survived'].map({0: 'Not Survived', 1: 'Survived'})

test_data=pandas.read_csv('Input/test.csv',index_col='PassengerId')
survived=calcResult();
print(survived)
test_data=test_data.merge(survived, left_index=True, sort=False,right_on="PassengerId")

#print(test_data)

ordinal_features = ['Pclass', 'SibSp', 'Parch']
nominal_features = ['Sex', 'Embarked']
survived_feature=['Survived']
numerical_feature=['Age','Fare']

correlation=train_data[ordinal_features + nominal_features+survived_feature+numerical_feature].corr().round(2);
print(correlation)
correlation=test_data[ordinal_features + nominal_features+survived_feature+numerical_feature].corr().round(2);
print(correlation)

#plot Age and Fare Comparison
#plotNumerical('Age')
#plotNumerical('Fare')
