import numpy
import pandas


def printMeansOfPredictions(correctPrediction, incorrectPrediction, features):
    print('Mean values of the features: correct vs. incorrect')
    for f in features:
        print('%s: %.2f vs. %.2f' % (f, correctPrediction[f].mean(), incorrectPrediction[f].mean()))


def getPassengers(passengerIds, testdata):
    incorrectPrediction = pandas.DataFrame()
    correctPrediction = pandas.DataFrame()

    for index, row in testdata.iterrows():
        passengerId = row['PassengerId']
        if passengerId in passengerIds:
            incorrectPrediction = incorrectPrediction.append(row)
        else:
            correctPrediction = correctPrediction.append(row)

    return [correctPrediction, incorrectPrediction]


def checkPredition(realResults, predictedResults):
    incorrectPredictionMen = []
    correctPredictionMen = []
    incorrectPredictionWomen = []
    correctPredictionWomen = []
    for index, row in realResults.iterrows():

        status = row['Survived']
        passenger = predictedResults.ix[index]
        predictedStatus = passenger['Survived']

        if status == predictedStatus:

            if row['Sex']==0:
                correctPredictionMen.append(row['PassengerId'])
            else:
                correctPredictionWomen.append(row['PassengerId'])
        else:
            if row['Sex'] == 0:
                incorrectPredictionMen.append(row['PassengerId'])
            else:
                incorrectPredictionWomen.append(row['PassengerId'])


    return [correctPredictionMen, incorrectPredictionMen,correctPredictionWomen,incorrectPredictionWomen]


def compare(realResults, predictedResults, testdataNumerical):
    predictions = checkPredition(realResults, predictedResults)
    print("Correct Prediction Men: %.f" % (len(predictions[0])))
    print("Incorrect Prediction Men: %.f" % (len(predictions[1])))
    print("Score Men %.4f" % (len(predictions[0])/(len(predictions[0])+len(predictions[1]))))

    print("Correct Prediction Women: %.f" % (len(predictions[2])))
    print("Incorrect Prediction Women: %.f" % (len(predictions[3])))
    print("Score %.4f" % (len(predictions[2]) / (len(predictions[2]) + len(predictions[3]))))

    print("Correct Prediction: %.f" % (len(predictions[0])+len(predictions[2])))
    print("Incorrect Prediction: %.f" % (len(predictions[1])+len(predictions[3])))

    correct=len(predictions[0])+len(predictions[2])
    incorrect=len(predictions[0])+len(predictions[1])+len(predictions[2])+len(predictions[3])
    print("Score %.4f" % (correct/incorrect))

    passengers = getPassengers(predictions[1], testdataNumerical)

    features = testdataNumerical.columns.values
    features = features[features != 'PassengerId']
    features = features[features != 'Name']
    features = features[features != 'Cabin']
    #printMeansOfPredictions(passengers[0], passengers[1], features)





realResults = pandas.read_csv('../Data/Output/Realresults.csv', delimiter=',')
predictedResults = pandas.read_csv('../Data/Output/NaiveBayesClassifier.csv', delimiter=',')
testdataNumerical = pandas.read_csv('../Data/Input/PreparedData/ABCEFGHIJK12/PreparedTest_ABCEFGHIJK1.csv',
                                    delimiter=',')
incorrect=compare(realResults, predictedResults, testdataNumerical)
print(incorrect)