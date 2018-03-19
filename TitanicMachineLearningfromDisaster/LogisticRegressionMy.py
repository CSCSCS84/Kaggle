import numpy
import pandas



class LogisticRegressionCS:

    def __init__(self, max_iter=None):
        if max_iter is None:
            self.max_iter = 100;
        else:
            self.max_iter = max_iter


    def sigmoid(self,x):
        return 1.0/(1+numpy.exp(-x))

    def calcResult(self,probability,test_data,predictFeature):
        test_data[predictFeature] = numpy.where(probability[predictFeature] < 0.5, 0, 1)
        return test_data;

    def calcThetaLogisticRegression(self,train_data,features):
        alpha = 0.001;
        numOfSteps=100;
        tol = 0.00001

        numberOfRemarks=len(features);
        m=numberOfRemarks;
        theta=pandas.DataFrame(numpy.ones((m, 1)),index=features)


        y=pandas.DataFrame(train_data['Survived']);

        XX=train_data[features]
        XXtranpose=XX.transpose();
        thetaBef=theta;
        cost=1.0;
        for i in range(1,self.max_iter ):
            if cost > tol:
                left = XXtranpose.multiply(alpha/m);
                sigMat = XX.dot(thetaBef);
                right = self.sigmoid(sigMat).values-(y);
                theta = thetaBef.values-(left.dot(right));
                diff=(thetaBef.values-theta.values)


                thetaBef=theta;
                cost=self.calcError(features,y,theta,XX)
                #print(cost)
            else:
                break;

        print("Theta from my logistic Regression")
        print(theta)
        print(cost)
        return theta;


    def calcError(self,features,y,theta,X):
        cost=0;
        tx=self.sigmoid(X.dot(theta))
        for i in range(1,len(features)):

            cost=cost+y.values[i]*numpy.log(tx.values[i]);+(1-y.values[i])*numpy.log((1-tx.values[i]))
        cost=cost*(-1/len(y))
        return cost;

    #use theta to prognose
    def calcPredictionDataframe(self,theta,features,testdata):
        XX = testdata[features]
        prob = self.sigmoid(XX.dot(theta))
        return prob;

    def logRegression(self,trainDataSet,testdata,features):
        theta = self.calcThetaLogisticRegression(trainDataSet, features);
        probability = self.calcPredictionDataframe(theta, features, testdata)
        result = self.calcResult(probability, testdata, 'Survived')
        return result;
