from Scaler import Scaler
from Node import Node
from Layer import Layer

"""
TO DO:
    -Convert the script to a formalized structure
    -Solve the bug where calling the val of Layer output will cause error
    -Test backprop
    -Implement loss function
"""

inputs = range(5)

layer_1 = Layer(5,3)
layer_2 = Layer(3,2)
layer_3 = Layer(2,1)

layer_1_out = layer_1.forward(inputs)
layer_2_out = layer_2.forward(layer_1_out)
layer_3_out = layer_3.forward(layer_2_out)

print(layer_3_out)
