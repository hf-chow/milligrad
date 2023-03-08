from Scaler import Scaler
from Node import Node

class Layer:

    def __init__(self, numIn, numOut):
        self.neurons = [Node(numIn) for i in range(numOut)]


L = Layer(5,3)
print(L.neurons)

for i in L.neurons:
    print(i.weights, i.bias)
