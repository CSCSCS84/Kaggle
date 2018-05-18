import pandas
from TitanicMachineLearningfromDisaster.TitanicInstance import TitanicInstanceCreator
import numpy
from TitanicMachineLearningfromDisaster.DataWorker import FilenameBuilder
import seaborn as sns
import matplotlib.pyplot as plt


def analyse(features):
    wrongPrediction = pandas.read_csv("WrongPrediction.csv", index_col='PassengerId')
    train = pandas.read_csv("PreparedTrain.csv", index_col='PassengerId')
    wrongDecribe = wrongPrediction.describe()
    trainDescribe = train.describe()

    for f in features:
        print(wrongDecribe[f])
        print(trainDescribe[f])
        print("")

def getInformationForPassengers(passengers, data):
    incorPas=pandas.DataFrame(columns=data.columns)
    for index, row in passengers.iterrows():
        pId=row['PassengerId']
        incorPas=incorPas.append(data.ix[pId])
    return incorPas

def analyseIncorrect(data):
    men = data[data['Sex'] == 'male']

    menPc1 = men[men['Pclass'] == 1]
    print(menPc1.shape[0])
    menPc2 = men[men['Pclass'] == 2]
    print(menPc2.shape[0])
    menPc3 = men[men['Pclass'] == 3]
    print(menPc3.shape[0])
    #plot = sns.countplot(x='AgeMissing',data=menPc3)

    plt.show()

fileNameExtensionTrain = 'ABCFGHIJKPQQRTUVY2'
fileNameExtensionTest = 'ABCFGHIJKPQQRTUVY'
featureNumber=0
#titanic = TitanicInstanceCreator.createInstance(fileNameExtensionTrain,fileNameExtensionTest,featureNumber)
titanic=TitanicInstanceCreator.createOriginalInstance()
fileIncorrect=FilenameBuilder.getIncorrectPredictionFilename(fileNameExtensionTrain,5)

passengers01 = pandas.read_csv(fileIncorrect[0])
incorPas=getInformationForPassengers(passengers01,titanic.train)
analyseIncorrect(incorPas)

passengers10 = pandas.read_csv(fileIncorrect[1])
incorPas=getInformationForPassengers(passengers10,titanic.train)
#print(incorPas)
analyseIncorrect(incorPas)