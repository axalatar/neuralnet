import numpy as np

class CostFunction:
    def __init__(self, name):
        self.name = name

    def get_name(self):
        return self.name
    
    def get_value(self, output, expected):
        raise Exception("Abstract class CostFunction should not be instantiated")
    
    def get_derivative(self, output, expected):
        raise Exception("Abstract class CostFunction should not be instantiated")
    


class CrossEntropyLoss(CostFunction):
    def __init__(self):
        super().__init__("Crossentropy")

    def get_value(self, output, expected):
        return -np.sum(expected * np.log(output + 1e-9))
    
    def get_derivative(self, output, expected):
        return output - expected