from statistics import mean
import numpy
import pandas
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from TitanicMachineLearningfromDisaster.ClassifierTuning import ClassifierFactory
from TitanicMachineLearningfromDisaster import ClassifierTuning
from TitanicMachineLearningfromDisaster import CrossValidation
from TitanicMachineLearningfromDisaster import MultipleClassifier


def start(train,test, y, fileNameExtension,featureNumber):

    firstLevelPrediction(train, test, y,fileNameExtension,featureNumber)
    ##print(result)
    #secondLevePrediction(train, test,y)

def firstLevelPrediction(train, test, y, fileNameExtension,featureNumber):
    # y = pandas.DataFrame(train['Survived'])
    trainData = train
    testData = test
    mc = MultipleClassifier.MultipleClassifier(getClassifiers(fileNameExtension,featureNumber))
    classifiers = [mc]

    fitClassifiers(classifiers, trainData, y)
    predict(classifiers, trainData)
    mc.yPredictions.to_csv('Data/Output/FirstLevelResultsTrain%s.csv' % (fileNameExtension), header='PassengerId\tSurvived',
                  sep=',')

    result=predict(classifiers, testData)
    mc.yPredictions.to_csv('Data/Output/FirstLevelResultsTest%s.csv' % (fileNameExtension),
                           header='PassengerId\tSurvived',
                           sep=',')

    #return
    result.to_csv('Data/Output/PredictedResults%s.csv' % (fileNameExtension), header='PassengerId\tSurvived', sep=',')

def secondLevePrediction(train, test, y):
    trainData = pandas.read_csv("Data/Output/FirstLevelResultsTrain%s.csv" % (fileNameExtension),
                        index_col='PassengerId')

    testData = pandas.read_csv("Data/Output/FirstLevelResultsTest%s.csv" % (fileNameExtension),
                        index_col='PassengerId')
    mc = MultipleClassifier.MultipleClassifier([getLogisticRegressionTuned()])
    classifiers = [mc]

    fitClassifiers(classifiers, trainData, y)
    scores = crossValidate(classifiers, trainData, y, 2, 10)



    yPredict = predict(classifiers, testData)
    result = pandas.DataFrame(index=testData.index)
    result['Survived']=yPredict

    result.to_csv('Data/Output/PredictedResults%s.csv' % (fileNameExtension), header='PassengerId\tSurvived', sep=',')




def secondLevelClassifier():
    return SVC(C=3, cache_size=200, class_weight=None, coef0=0.0,
               decision_function_shape='ovr', degree=2, gamma=0.1, kernel='rbf',
               max_iter=-1, probability=True, random_state=None, shrinking=True,
               tol=0.001, verbose=False)


def fitClassifiers(classifiers, trainData, y):
    for c in classifiers:
        c.fit(trainData, y.values.ravel())

def crossValidate(classifiers, trainData, y, numOfValidations,k):
    scores = []
    for c in classifiers:
        meanSum=0
        for i in range(0, numOfValidations):
            score = CrossValidation.validate(trainData, k, c, y)
            meanSum+= mean(score)
            print(score)
        average=numpy.round(meanSum/numOfValidations,3)
        scores.append(average)

    return scores

def predict(classifiers, testData):
    result = pandas.DataFrame(index=testData.index)
    for c in classifiers:
        yAll = c.predict(testData)
        result[c.__class__.__name__] = yAll
    return result


def getClassifiers(fileNameExtension, featureNumber):
    classifiers = []
    classifiers.append(ClassifierFactory.constructClassifier("SVC",fileNameExtension,featureNumber,"0.8309"))
    #classifiers.append(getExtraTreeTuned())
    #classifiers.append(getGradientBoostingTuned())
    #classifiers.append(getMLPTuned())
    #classifiers.append(getLinearDiscriminantTuned())
    #classifiers.append(getLogisticRegressionTuned())
    #classifiers.append(getRandomForestTuned())
    #classifiers.append(getAdaboostTuned())
    #classifiers.append(getDecisionTreeTuned())

    #classifiers.append(getKNeighborsTuned())
    return classifiers

def votingClassifier(train, y_train,test):
    votingC = VotingClassifier(estimators=[('rfc', getGradientBoostingTuned()), ('extc', getAdaboostTuned()),
                                           ('svc', getLinearDiscriminantTuned()), ('adac', getLogisticRegressionTuned()), ('gbc', getKNeighborsTuned())], voting='soft',
                               n_jobs=4)

    votingC = votingC.fit(train, y_train)
    test_Survived = pandas.Series(votingC.predict(test), name="Survived")
    yPredicted=votingC.predict(test)
    result=pandas.DataFrame(index=test.index)
    result['Survived']=yPredicted
    #print(test_Survived)
    #results = pandas.concat([IDtest, test_Survived], axis=1)

    #results.to_csv("ensemble_python_voting.csv", index=False)#
    result.to_csv('Data/Output/PredictedResultsVotingClassifier.csv' , header='PassengerId\tSurvived', sep=',')

fileNameExtension = 'ABCEFGHI12'
fileNameExtensionTest = 'ABCEFGHI1'
train = pandas.read_csv("Data/Input/PreparedData/ABCEFGHI12/PreparedTrain_%s.csv" % (fileNameExtension),
                        index_col='PassengerId')
test = pandas.read_csv("Data/Input/PreparedData/ABCEFGHI12/PreparedTest_%s.csv" % (fileNameExtensionTest),
                       index_col='PassengerId')

features = ['Age',  'Fare','Parch', 'Sex', 'SibSp', 'Single',
            'SmallFamily', 'MediumFamily', 'LargeFamily', 'Embarked_C',
            'Embarked_Q', 'Embarked_S', 'Title_0', 'Title_1', 'Title_2', 'Title_3',
            'Pclass_1', 'Pclass_2', 'Pclass_3', 'Cabin_A', 'Cabin_B', 'Cabin_C',
            'Cabin_D', 'Cabin_E', 'Cabin_F', 'Cabin_G', 'Cabin_T', 'Cabin_X',  'Ticket_A', 'Ticket_A4', 'Ticket_A5', 'Ticket_AQ3', 'Ticket_AQ4',
                'Ticket_AS', 'Ticket_C', 'Ticket_CA', 'Ticket_CASOTON', 'Ticket_FC',
                'Ticket_FCC', 'Ticket_Fa', 'Ticket_LINE', 'Ticket_LP', 'Ticket_PC',
                'Ticket_PP', 'Ticket_PPP', 'Ticket_SC', 'Ticket_SCA3', 'Ticket_SCA4',
                'Ticket_SCAH', 'Ticket_SCOW', 'Ticket_SCPARIS', 'Ticket_SCParis',
                'Ticket_SOC', 'Ticket_SOP', 'Ticket_SOPP', 'Ticket_SOTONO2',
                'Ticket_SOTONOQ', 'Ticket_SP', 'Ticket_STONO', 'Ticket_STONO2',
                'Ticket_STONOQ', 'Ticket_SWPP', 'Ticket_WC', 'Ticket_WEP', 'Ticket_X']

y=train['Survived']
trainData = train[features]
testData = test[features]
start(trainData,testData,y, fileNameExtension, featureNumber=1)
#votingClassifier(trainData,y,testData)

