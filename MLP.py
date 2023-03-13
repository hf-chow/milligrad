from Scaler import Scaler
from Node import Node
from Layer import Layer

from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

"""
TO DO:
    -Convert the script to a formalized structure
    -Solve the bug where calling the val of Layer output will cause error
    -Test backprop
    -Implement loss function
"""

#X, y = make_blobs(centers=2, random_state=64)

class MLP:
    
    def __init__(self, in_num, layers_dims):
        """
        What we want:
            Use two number to construct the MLP, no. of inputs, 
            and a list of layer sizes. The input 
        """
        self.first_layer = Layer(in_num, layers_dims[0])
        self.layers = [self.first_layer]+[Layer(layers_dims[i-1], layers_dims[i]) for i in range(len(layers_dims))]

model = MLP(2, [16, 16, 1])

print(model.layers)
        # 3 layers: 2 -> 16 -> 16 -> 1
        # [ [2, 16],
        #   [16, 16],
        #   [16, 1] ]
# H(P,Q) = -sum x in X P(x) * log(Q(x))
