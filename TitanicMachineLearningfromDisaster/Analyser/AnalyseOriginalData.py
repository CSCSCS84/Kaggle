import seaborn as sns
import matplotlib.pyplot as plt
from TitanicMachineLearningfromDisaster.DataWorker import FilenameBuilder
from TitanicMachineLearningfromDisaster.TitanicInstance import TitanicInstanceCreator
import numpy

def countNanValues(data):

    men=data[data['Sex']=='male']

    menPc1=men[men['Pclass']==1]
    print(menPc1['Age'].isnull().sum())
    menPc2 = men[men['Pclass'] == 2]
    print(menPc2['Age'].isnull().sum())
    menPc3 = men[men['Pclass'] == 3]
    print(menPc3['Age'].isnull().sum())

    menPc1AgeNotKnown=menPc1[menPc1['Age'].isnull()]

    #plot = sns.countplot(data=menPc1AgeNotKnown, y='Survived')

    menPc1AgeKnown=menPc1[menPc1['Age']>=0]
    #plot = sns.countplot(data=menPc1AgeNotKnown, y='Survived')
    plot = sns.countplot(data=menPc1AgeKnown, y='Survived')
    plt.show()

    menPc3AgeNotKnown = menPc3[menPc3['Age'].isnull()]

    plot = sns.countplot(data=menPc3AgeNotKnown, y='Survived')
    menPc3AgeKnown = menPc3[menPc3['Age'] >= 0]
    plot = sns.countplot(data=menPc3AgeNotKnown, y='Survived')
    # plot = sns.countplot(data=menPc1AgeKnown, y='Survived')
    plt.show()

def analyseData(data):
    men = data[data['Sex'] == 'male']

    menPc1 = men[men['Pclass'] == 1]
    print(menPc1['Age'].isnull().sum())
    menPc2 = men[men['Pclass'] == 2]
    print(menPc2['Age'].isnull().sum())
    menPc3 = men[men['Pclass'] == 3]
    print(menPc3['Age'].isnull().sum())

    menPc3Single=menPc3[menPc3['SibSp']==0]
    plot = sns.countplot(x='Age',hue='Survived', data=menPc1)

    plt.show()


titanic = TitanicInstanceCreator.createOriginalInstance()
#countNanValues(titanic.train)
analyseData(titanic.train)