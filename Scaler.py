import numpy as np
from math import e

class Scaler:
    """
    TO DO
    - Implement topological sort for backprop()
    - Add support for sigmoid, tanh as activation function

    """   
    def __init__(self, val, _history=()):
        self.val = float(val)
        self._prev = set(_history)
        self.grad = 0.0
        self._backprop = lambda: None

    def __repr__(self):
        return f"Scaler(val={self.val})"

    def __add__(self, other):
        """
        According to chain rule,
        dL/da = dL/dx * dx/da       where x = a+b;  L=loss, a is local, b is bias
        dL/da is known, as it is the result from a prev step of backprop
        
        dL/da = dL/dx * 1
        dL/dx = dL/da

        i.e. local grad is the same as the prev step grad 
        """
        output = Scaler(self.val + other.val, (self, other))
        
        def _backprop():
            self.grad = output.grad    #Moving the next step's grad to the prev step
            other.grad = output.grad

        output._backprop = _backprop    #Recurvsively calling _backprop

        return output

    def __mul__(self, other):
        """
        According to chain rule,
        dL/da = dL/dx * dx/da       where x = a*b; L=loss, a is local, b is weight
        dL/da is known, as it is the result from a prev step of backprop

        dL/da = dL/dx * (b(da/da) +a*(db/da))
        dL/da = dL/dx * b 
        """
        output = Scaler(self.val * other.val, (self, other))

        def _backprop():
            self.grad = output.grad*other.val
            other.grad = output.grad*self.val

        output._backprop = _backprop

        return output 

    def __truediv__(self, other):
        return Scaler(self.val / other.val, (self, other))

    def __sub__(self, other):
        return Scaler(self.val - other.val, (self, other))

    def __pow__(self, other):
        return Scaler(self.val ** other.val, (self, other))

    def relu(self):
        output = Scaler(max(0, self.val), (self, ))
        
        def _backprop():
            if output.val > 0:
                self.grad = 1 * output.grad
            else:
                self.grad = 0 * output.grad

        output._backprop = _backprop
 
        return output

    def sigmoid(self):
        return Scaler(e**(self.val)/(1+e**(self.val))).val

    def backprop(self):
        self._backprop()
        for prev in self._prev:
            print("reach", prev)
            prev._backprop()
            prev.backprop()


e = Scaler(-1) 
f = Scaler(2)
d = Scaler(4)
b = Scaler(5)

c = e*f
a = c+d
C = a*b
L = C.relu()

print(L._prev)
L.grad = 1.0
L.backprop()
print(f.grad)

