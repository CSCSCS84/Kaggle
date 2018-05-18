from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from TitanicMachineLearningfromDisaster.TitanicInstance import TitanicInstanceCreator
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB


from TitanicMachineLearningfromDisaster.ClassifierTuning import ClassifierTuner
from TitanicMachineLearningfromDisaster.TitanicInstance import TitanicInstanceCreator

fileNameExtension = 'ABCFGHIJKPRTVYQQU'
fileNameExtensionTest = 'ABCFGHIJKPRTVYQQU'
featureNumber = 9

titanic = TitanicInstanceCreator.createInstance(fileNameExtension, fileNameExtensionTest, featureNumber)
searchGrid = ClassifierTuner.getBernoulliNBGrid()

d=DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=None,
            max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, presort=False, random_state=None,
            splitter='best')
d = BernoulliNB()

classifier = BernoulliNB()

tunedClassifier = ClassifierTuner.tuneClassifier(titanic, classifier, searchGrid)
print(tunedClassifier)

ClassifierTuner.saveTunedClassifier(tunedClassifier, fileNameExtension, featureNumber)