

class TunedClassifier():

    def __init__(self, classifier=None, kfoldScore=None):
        if classifier is None:
            self.classifier = None
        else:
            self.classifier = classifier

        if kfoldScore is None:
            self.kfoldScore = 0.0
        else:
            self.kfoldScore = kfoldScore

    def __str__(self):
        return "kfoldScore: %.4f \n %s" % (self.kfoldScore, str(self.classifier))