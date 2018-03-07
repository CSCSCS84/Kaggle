import numpy
import pandas
import matplotlib.pyplot as plt
import seaborn as sns

#


### Seaborn style
sns.set_style("whitegrid")

train_data = pandas.read_csv('train.csv',index_col='PassengerId')
train_data['target_name'] = train_data['Survived'].map({0: 'Not Survived', 1: 'Survived'})
train_data['Sex'] = train_data['Sex'].map({'male': 1, 'female': 0})
train_data['Embarked'] = train_data['Embarked'].map({'S': 0, 'C': 1,'Q':2})

sns.countplot(train_data.target_name);
plt.xlabel('Survived?');
plt.ylabel('Number of occurrences');
#plt.show()
plt.clf()


# Categorical Features
ordinal_features = ['Pclass', 'SibSp', 'Parch']
nominal_features = ['Sex', 'Embarked']
survived_feature=['Survived']
numerical_feature=['Age','Fare']

correlation=train_data[ordinal_features + nominal_features+survived_feature+numerical_feature].corr().round(2);
print(correlation)

#print(train_data['Age']);
#sns.countplot(train_data['Age']);
#sns.distplot(train_data['Age'].dropna());
#plt.ylabel('Density', fontsize=14);
#plt.show()


sns.distplot(train_data['Pclass'].dropna());
plt.ylabel('Density', fontsize=14);
sns.distplot(train_data.loc[train_data.Survived == 0, 'Pclass'].dropna(),color='red',hist_kws=dict(edgecolor=None,linewidth=0))
plt.show();

sns.distplot(train_data['Age'].dropna());
plt.ylabel('Density', fontsize=14);
sns.distplot(train_data.loc[train_data.Survived==0, 'Age'].dropna(),color='red',hist_kws=dict(edgecolor=None,linewidth=0))
plt.show();

sns.distplot(train_data['Sex'].dropna());
plt.ylabel('Density', fontsize=14);
sns.distplot(train_data.loc[train_data.Survived==0, 'Sex'].dropna(),color='red',hist_kws=dict(edgecolor=None,linewidth=0))
plt.show();

sns.distplot(train_data['Embarked'].dropna());
plt.ylabel('Density', fontsize=14);
sns.distplot(train_data.loc[train_data.Survived==0, 'Embarked'].dropna(),color='red',hist_kws=dict(edgecolor=None,linewidth=0))
plt.show();

sns.distplot(train_data['Fare'].dropna(),hist=False);
plt.ylabel('Fare', fontsize=14);
sns.distplot(train_data.loc[train_data.Survived==0, 'Fare'].dropna(),color='red',hist_kws=dict(edgecolor=None,linewidth=0),hist=False)
plt.show();