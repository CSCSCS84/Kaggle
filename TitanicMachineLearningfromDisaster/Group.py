class Group:
    def __init__(self, ticketnumber=None, fare=None, numOfDead=None, numOfSurvived=None):


        if ticketnumber is None:
            self.ticketnumber = 0
        else:
            self.ticketnumber = int(ticketnumber)
        if fare is None:
            self.fare = 0
        else:
            self.fare = fare

        if numOfDead is None:
            self.numOfDead = 0
        else:
            self.numOfDead = numOfDead
        if numOfSurvived is None:
            self.numOfSurvived = 0
        else:
            self.numOfSurvived = numOfSurvived

    def __hash__(self):
        return hash((self.fare))

    def __eq__(self, obj):
        #if (( (self.ticketnumber !=-1) and (abs(self.ticketnumber - obj.ticketnumber)) <= 1) and (self.fare == obj.fare)):
        #     return True
        if   ( (self.ticketnumber !=-1)  and (self.ticketnumber==obj.ticketnumber) ):
            return True
        else:
            return False

    def __str__(self):
        return "%s %s %s %s" % (self.ticketnumber, self.fare, self.numOfDead, self.numOfSurvived)
