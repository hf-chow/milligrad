from Scaler import Scaler
import random

class Node:

    """
    Need to figure out where backprop comes in and need to
    test if the grad are callable

    We might not need a backprop method. We can just call 
    backprop on the training loss

    The way we call and init node is super ugly. Need to 
    clean that up
    """

    def __init__(self, numIn):
        self.weights = [Scaler(random.uniform(-1, 1)) for i in range(numIn)]
        self.bias = Scaler(random.uniform(-1, 1))

    def forward(self, inputs):
        output = Scaler(0)
        zipped = zip(self.weights, inputs)
        for weight, _input in zipped:
            output += (weight*_input + self.bias).relu()
        return output 

    #def backprop(self):
    #    pass
