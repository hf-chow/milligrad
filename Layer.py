from Scaler import Scaler
from Node import Node

class Layer:

    def __init__(self, numIn, numOut):
        self.neurons = [Node(numIn) for i in range(numOut)]

    def forward(self, inputs):
        output = Scaler(0)
        for n in self.neurons:
            output += n.forward(inputs)
        return output


L = Layer(5,3)
print(L.neurons)

inputs = [Scaler(i) for i in range(5)]

for i in L.neurons:
    print(i.weights, i.bias)
    print(L.forward(inputs))
