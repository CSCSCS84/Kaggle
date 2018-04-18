import numpy
import pandas

def printMeansOfPredictions(correctPrediction, incorrectPrediction, features):
    print('Mean values of the features: correct vs. incorrect')
    for f in features:
        print('%s: %.2f vs. %.2f' % (f,correctPrediction[f].mean(),incorrectPrediction[f].mean()))


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
    incorrectPrediction = []
    correctPrediction = []
    for index, row in realResults.iterrows():
        status = row['Survived']
        passenger = predictedResults.ix[index]
        predictedStatus = passenger['Survived']

        if status == predictedStatus:
            correctPrediction.append(row['PassengerId'])
        else:
            incorrectPrediction.append(row['PassengerId'])

    return [correctPrediction, incorrectPrediction]

def compare(realResults, predictedResults, testdataNumerical):

    predictions = checkPredition(realResults, predictedResults)
    print("Correct Prediction: %.f" % (len(predictions[0])))
    print("Incorrect Prediction: %.f" % (len(predictions[1])))

    passengers = getPassengers(predictions[1], testdataNumerical)

    features = testdataNumerical.columns.values
    features = features[features != 'PassengerId']
    features = features[features != 'Name']
    features = features[features != 'Cabin']
    printMeansOfPredictions(passengers[0], passengers[1], features)

realResults = pandas.read_csv('Data/Output/Realresults.csv', delimiter='\t')
predictedResults = pandas.read_csv('Data/Output/PredictedResultsABCEFGHI12.csv', delimiter=',')
testdataNumerical = pandas.read_csv('Data/Input/testNumerical.csv', delimiter=',')
compare(realResults, predictedResults, testdataNumerical)