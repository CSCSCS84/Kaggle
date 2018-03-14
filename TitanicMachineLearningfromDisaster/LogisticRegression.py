import numpy
import pandas

def sigmoid(x):
    return 1.0/(1+numpy.exp(-x))

def calcResult(probability,test_data,predictFeature):
    test_data[predictFeature] = numpy.where(probability[predictFeature] < 0.5, 0, 1)
    return test_data;

def calcThetaLogisticRegression(train_data,features):
    alpha = 0.0001;
    numOfSteps=50000;
    difBorder=0.0001

    numberOfRemarks=len(features);
    m=numberOfRemarks;
    theta=pandas.DataFrame(numpy.ones((m, 1)),index=features)


    y=pandas.DataFrame(train_data['Survived']);

    XX=train_data[features]
    XXtranpose=XX.transpose();
    thetaBef=theta;
    dif=1.0;
    for i in range(1,numOfSteps):
        if dif > difBorder:
            left = XXtranpose.multiply(alpha/m);
            sigMat = XX.dot(thetaBef);
            right = sigmoid(sigMat).values-(y);
            theta = thetaBef.values-(left.dot(right));
            diff=(thetaBef.values-theta.values)
            dif=numpy.absolute(diff).sum()

            thetaBef=theta;
        else:
            break;
    print(dif)
    return theta;

#use theta to prognose
def calcPredictionDataframe(theta,features,testdata):
    XX = testdata[features]
    prob=sigmoid(XX.dot(theta))
    return prob;

def logRegression(trainDataSet,testdata,features):
    theta = calcThetaLogisticRegression(trainDataSet, features);
    print(theta)
    probability = calcPredictionDataframe(theta, features, testdata)
    # print(testdata)
    result = calcResult(probability, testdata, 'Survived')
    return result;
