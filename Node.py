from Scaler import Scaler
import random

class Node:

    def __init__(self, numIn):
        self.weights = [Scaler(random.uniform(-1, 1)) for i in range(numIn)]
        self.biases = Scaler(random.uniform(-1, 1))

    def foward(self):
        pass

    def backprop(self):
        pass

