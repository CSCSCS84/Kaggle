import numpy
import pandas


class MyFirstClass:
    a=1.5;
    def printa(self):
        print(self.a)


    def __init__(self, a=None):
        if a is None:
            self.a=1.5
        else:
            self.a=a


cl=MyFirstClass()
cl.printa();
