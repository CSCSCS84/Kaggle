import numpy
import pandas
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from TitanicMachineLearningfromDisaster import MultipleClassifier
from TitanicMachineLearningfromDisaster import CrossValidation

def start():
    train = pandas.read_csv("PreparedTrain.csv", index_col='PassengerId')
    test = pandas.read_csv("PreparedTest.csv", index_col='PassengerId')

    # ytest = regression(classifier, train, test, y)
    features = ['Age', 'Fare', 'Sex', 'Single', 'SmallFamily', 'MediumFamily', 'LargeFamily', 'EM_C', 'EM_Q', 'EM_S',
                'Title_0', 'Title_1', 'Title_2', 'Title_3', 'Pc_1', 'Pc_2', 'Pc_3']

    y = pandas.DataFrame(train['Survived'])
    trainData = train[features]
    testData = test[features]

    classifiers = []
    classifierLogReg = LogisticRegression(random_state=0, max_iter=1000, fit_intercept=False, tol=0.00001)
    random_state = 2
    classifierForest = RandomForestClassifier(random_state=random_state)
    classifierAdaBoost = AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state),
                                            random_state=random_state, learning_rate=0.1)
    classifierForestDecisionTree=DecisionTreeClassifier(random_state=random_state)
    classifierKneigbhors = KNeighborsClassifier()
    classifiers.append(classifierLogReg)
    classifiers.append(classifierForest)
    classifiers.append(classifierAdaBoost)
    classifiers.append(classifierForestDecisionTree)
    classifiers.append(classifierKneigbhors)
    # validate(train[features], 10, classifierCS, y)
    #CrossValidation.validate(trainData, 10, classifierAdaBoost, y)
    MultipleClassifier.fit(classifiers,trainData,y)
    yAll = MultipleClassifier.predict(classifiers,testData)
    result = pandas.DataFrame(index=test.index)


    result['Survived'] = yAll;
    result.to_csv('Calcedresults.csv', header='PassengerId\tSurvived', sep=',')



start()