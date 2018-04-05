import numpy
import pandas
from TitanicMachineLearningfromDisaster import LogisticRegressionCS
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix


def regression(classifier, train, test, y):
    classifier.fit(train, y)
    ytest = classifier.predict(test)
    return ytest;

def saveResultsToFile(ytest,result):
    result['Survived'] = ytest;
    result.to_csv('Calcedresults.csv', header='PassengerId\tSurvived', sep=',')

def printScore(classifier,train,y):
    score = classifier.score(train, y)
    print('Accuracy of logistic regression classifier: {:.2f}'.format(score))

def printConfusionMatrix(classifier,train,y):
    ypredTrain = classifier.predict(train)
    confusionMatrix = confusion_matrix(y, ypredTrain)
    print('Confusion Matrix')
    print(confusionMatrix)

def wrongPrediction(train, y, ytrainPrediction):
    dataWrongPrediction = pandas.DataFrame()
    i = 0;
    for index, row in train.iterrows():

        if ytrainPrediction[i] != y.values[i]:
            dataWrongPrediction = dataWrongPrediction.append(row)
        i += 1;
    return dataWrongPrediction;

def startModule():
    train = pandas.read_csv("Data/Input/PreparedData/PreparedTrain.csv", index_col='PassengerId')
    test = pandas.read_csv("Data/Input/PreparedData/PreparedTest.csv", index_col='PassengerId')


    features = ['Age', 'Fare', 'Sex', 'Single', 'SmallFamily', 'MediumFamily', 'LargeFamily', 'EM_C', 'EM_Q', 'EM_S',
                'Title_0', 'Title_1', 'Title_2', 'Title_3', 'Pc_1', 'Pc_2', 'Pc_3']
    y = pandas.DataFrame(train['Survived'])
    train = train[features]
    test = test[features]

    classifierCS = LogisticRegressionCS.LogisticRegressionCS(max_iter=1000, tolerance=0.00001);
    result = pandas.DataFrame(index=test.index)
    ytestCS = regression(classifierCS, train, test, y)

    saveResultsToFile(ytestCS,result)
    printScore(classifierCS, train, y)
    printConfusionMatrix(classifierCS, train, y)

    classifier = LogisticRegression(random_state=0, max_iter=10, fit_intercept=False, tol=0.00001)
    ytest = regression(classifier, train, test, y)
    ytest = classifier.predict(train)
    printScore(classifier, train, y)
    printConfusionMatrix(classifier, train, y)

    ytrainPrediction = classifierCS.predict(train)


    dataWrongPrediction = wrongPrediction(train, y, ytrainPrediction)

startModule()