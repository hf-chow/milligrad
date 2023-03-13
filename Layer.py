from Scaler import Scaler
from Node import Node

class Layer:

    def __init__(self, in_num, out_num):
        self.neurons = [Node(in_num) for i in range(out_num)]

    def forward(self, inputs):
        scaler_inputs = []
        for i in inputs:
            if isinstance(i, Scaler):
                scaler_inputs.append(i) 
            else:
                scaler_inputs.append(Scaler(i))
        outputs = [] 
        for n in self.neurons:
            outputs.append(n.forward(scaler_inputs))
        return outputs

