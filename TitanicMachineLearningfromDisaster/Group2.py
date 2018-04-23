class Group:
    def __init__(self, ticketnumber=None, fare=None):


        if ticketnumber is None:
            self.ticketnumberAvg = 0
        else:

            self.ticketnumberAvg = int(ticketnumber)
        if fare is None:
            self.fareAvg = 0
        else:
            self.fareAvg = fare
        passengers=[]


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
