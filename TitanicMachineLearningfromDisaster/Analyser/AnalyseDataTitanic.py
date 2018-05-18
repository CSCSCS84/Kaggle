import seaborn as sns
import matplotlib.pyplot as plt
from TitanicMachineLearningfromDisaster.DataWorker import FilenameBuilder
from TitanicMachineLearningfromDisaster.TitanicInstance import TitanicInstanceCreator
import numpy

def generalAnalysis(dataset):
    print(dataset.isnull().sum())
    print(dataset.info())
    print(dataset.describe())


def printCorrelation(dataset):
    ordinal_features = ['Pclass', 'SibSp', 'Parch']
    nominal_features = ['Sex', 'Embarked']
    survived_feature = ['Survived']
    numerical_feature = ['Age', 'Fare']
    correlation = dataset[ordinal_features + nominal_features + survived_feature + numerical_feature].corr().round(
        2);
    print(correlation)


def plotSurvivalProbability(dataset, features):
    for f in features:
        plot = sns.factorplot(x=f, y='Survived', data=dataset, kind='bar')
        plot = plot.set_ylabels("survival probability")


def plotSurvivalProbabilityHue(dataset, features, hue):
    for f in features:
        plot = sns.factorplot(x=f, y='Survived', hue=hue, data=dataset, kind='bar')
        plot = plot.set_ylabels("survival probability")
        plt.show()


def plotSkewness(dataset, features):
    for f in features:
        plot = sns.distplot(dataset[f], color="m", label="Skewness : %.2f" % (dataset[f].skew()))
        plot = plot.legend(loc="best")
        plt.show()


def analyseAge(dataset, features):
    for f in features:
        plot = sns.factorplot(x=f, y='Age', data=dataset, kind='box')
        plot = plot.set_ylabels("Age")
        plt.show()


def survivalProbabilityGroupBySex(dataset):
    print(dataset[['Sex', 'Survived']].groupby('Sex').mean());


def plotSibSpFare(dataset):
    plot = sns.factorplot(x='MediumFamily', y='Fare', data=dataset, kind='bar')
    plot = plot.set_ylabels("Fare")
    plt.show()


def PclassAnalyse(dataset):
    plot = sns.factorplot(x='Fare', y='Survived', data=dataset, kind='bar')
    plot = plot.set_ylabels("survival probability")
    plt.show()


def groupAnalyse(dataset):
    datasetMen = dataset[(dataset['Sex'] == 0) & (dataset['Pclass_2'] == 1)]
    # print(datasetYoungMenSmall)
    # plot = sns.countplot(x='Age',data=datasetYoungMen,hue='Survived')
    # plt.show()
    # print(datasetMen.shape[0])

    # datasetMenGroup1=datasetMen[(datasetMen['GroupSize_4.0']==1)]
    # print(datasetMenGroup1)
    print(datasetMen)
    datasetMen = datasetMen.sort_values(by='Age')
    plot = sns.countplot(data=datasetMen, x='Age', hue='Survived')
    # datasetMen['Age'].value_counts().plot(kind="bar")
    plot.set(xticks=datasetMen.Age[1::5])
    plt.show()
    # plot = sns.countplot(x='GroupSurvivedRatio2', data=datasetMenGroup1, hue='Survived')
    # plt.show()
    # plot = sns.factorplot(x='Age', y='Survived', data=datasetMenGroup1, kind='bar')
    # plot = plot.set_ylabels("survival probability")
    # plt.show()

def pclassSexAnalyse(dataset):
    datasetMen1 = dataset[(dataset['Sex'] == 0) & (dataset['Pclass_1'] == 1)]
    datasetMen2 = dataset[(dataset['Sex'] == 0) & (dataset['Pclass_2'] == 1)]
    datasetMen3 = dataset[(dataset['Sex'] == 0) & (dataset['Pclass_3'] == 1)]

    men1=numpy.zeros(dataset.shape[0])

    for index, row in dataset.iterrows():
        if (row['Sex'] == 0) & (row['Pclass_1'] == 1):
            men1[index]=1

    dataset['MenPc1']=men1
    #print(datasetMen1)
    plot = sns.countplot(data=datasetMen1,  y='Survived')
    plt.show()
    plot = sns.countplot(data=datasetMen2, y='Survived')
    plt.show()
    plot = sns.countplot(data=datasetMen3, y='Survived')
    plt.show()
    datasetWomen1 = dataset[(dataset['Sex'] == 1) & (dataset['Pclass_1'] == 1)]
    datasetWomen2 = dataset[(dataset['Sex'] == 1) & (dataset['Pclass_2'] == 1)]
    datasetWomen3 = dataset[(dataset['Sex'] == 1) & (dataset['Pclass_3'] == 1)]
    # print(datasetMen1)
    plot = sns.countplot(data=datasetWomen1, y='Survived')
    plt.show()
    plot = sns.countplot(data=datasetWomen2, y='Survived')
    plt.show()
    plot = sns.countplot(data=datasetWomen3, y='Survived')
    plt.show()

def fareAnalysis(dataset):
    fares = dataset['Fare']
    bound = 100

    stepSize = 5
    surProb = []
    for i in range(0, bound,stepSize):
        print(i)
        passengers = dataset[(dataset['Fare'] < i + 1*stepSize) & (dataset['Fare'] > i)]
        survived = len(passengers[passengers['Survived'] == 1])
        dead = len(passengers[passengers['Survived'] == 0])
        if (dead + survived) > 0:
            surProb.append(survived / (dead + survived))
        else:
            surProb.append(-1)
    #print(surProb)

    dataset['Cheap'] = fares.map(lambda s: 1 if s <= 10 else 0)
    dataset['Medium'] = fares.map(lambda s: 1 if 10 < s <= 35 else 0)
    dataset['High'] = fares.map(lambda s: 1 if 35 < s <= 100 else 0)
    dataset['VeryHigh'] = fares.map(lambda s: 1 if s >= 100 else 0)
    #datasetCheap=dataset[dataset['Cheap']==1]
    #plot = sns.factorplot(x='Fare', y='Survived', data=datasetCheap, kind='bar')
    #plt.show()
    plot = sns.countplot(data=dataset, x='Cheap', hue='Survived')
    plt.show()
    plot = sns.countplot(data=dataset, x='Medium', hue='Survived')
    plt.show()

    plot = sns.countplot(data=dataset, x='High', hue='Survived')
    plt.show()
    plot = sns.countplot(data=dataset, x='VeryHigh', hue='Survived')
    plt.show()

def sexPcAnalysis(dataset):
    datasetMen1 = dataset[(dataset['Womenc3'] == 1)]
    #datasetMen1Young=datasetMen1[datasetMen1['Age']>10]
    #plot = sns.countplot(data=datasetMen1Young, y='Survived')

    fares = datasetMen1['Age']
    bound = 100

    stepSize = 5
    surProb = []
    for i in range(0, bound, stepSize):
        #print(i)
        passengers = datasetMen1[(datasetMen1['Age'] < i + 1 * stepSize) & (datasetMen1['Age'] > i)]
        survived = len(passengers[passengers['Survived'] == 1])
        dead = len(passengers[passengers['Survived'] == 0])
        if (dead + survived) > 0:
            surProb.append(survived / (dead + survived))
        else:
            surProb.append(-1)
    print(surProb)
    #plot = sns.factorplot(x='Age', y='Survived', data=datasetMen1, kind='bar')
    g = sns.FacetGrid(datasetMen1, col='Survived')
    g = g.map(sns.distplot, "Age")
    #plt.show()

def pc3TitleAnalysis(dataset):
    #datasetMen3 = dataset[(dataset['Sex'] == 0)]
    #datasetMen3=datasetMen3[datasetMen3['MenPc3']==1]
    plot = sns.countplot(data=dataset,x='GroupSurvival', hue='Survived')
    plt.show()


def startAnalysis(train, features):
    # plotSibSpFare(train)
    # groupAnalyse(train)
    #fareAnalysis(train)
    pc3TitleAnalysis(train)
    sexPcAnalysis(train)
    pclassSexAnalyse(train)
    generalAnalysis(train)
    printCorrelation(train)
    survivalProbabilityGroupBySex(train)
    analyseAge(train, features=['Sex', 'SibSp', 'Parch', 'Pclass'])
    plotSurvivalProbability(train, features)
    plotSurvivalProbabilityHue(train, features, hue='Sex')
    plotSkewness(train, ['Fare'])
    analyseAge(train, features=['Sex', 'SibSp', 'Parch', 'Pclass'])


fileNameExtension = 'ABCFGHIJKPQQRTUVY2'
fileNameExtensionTest = 'ABCFGHIJKPQQRTUVY'
featureNumber = 8
titanic = TitanicInstanceCreator.createInstance(fileNameExtension, fileNameExtensionTest, featureNumber)
#
# plotSkewness(train, ['Fare'])

# trainPclass=train[train['Pclass']==3]
# testPclass=test[test['Pclass']==3]

startAnalysis(titanic.train, titanic.features)
# PclassAnalyse(trainPclass)

#correlation = titanic.train[titanic.features].corr().round(
#        2)
#print(correlation)
#ax = sns.heatmap(correlation,annot=True)

#plt.show()