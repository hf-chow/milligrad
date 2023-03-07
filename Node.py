from Scaler import Scaler
import random

class Node:

    def __init__(self, numIn):
        self.weights = [Scaler(random.uniform(-1, 1)) for i in range(numIn)]
        self.bias = Scaler(random.uniform(-1, 1))
        self.relu = Scaler

    def forward(self, inputs):
        output = Scaler(0)
        zipped = zip(self.weights, inputs)
        for weight, _input in zipped:
            output += (weight*_input + self.bias).relu()
        return output 

    def backprop(self):
        pass


#inputs = [Scaler(3), Scaler(2), Scaler(-1)]
#node = Node(len(inputs))

#print(node.weights, node.bias)
#print(node.forward(inputs))
