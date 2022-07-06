import torch
import torch.nn as nn
import torch.nn.parallel
import numpy as np
from torch.nn.parameter import Parameter

class rational(nn.Module):
    def __init__(self):
        super(rational, self).__init__()
        print('rational!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        self.a0 = Parameter(torch.FloatTensor([0.0218]))
        self.a1 = Parameter(torch.FloatTensor([0.5000]))
        self.a2 = Parameter(torch.FloatTensor([1.5957]))
        self.a3 = Parameter(torch.FloatTensor([1.1915]))
        self.b0 = Parameter(torch.FloatTensor([1.0000]))
        self.b1 = Parameter(torch.FloatTensor([0.0000]))
        self.b2 = Parameter(torch.FloatTensor([2.3830]))
    def forward(self, x):
        y = (self.a3*x**3+self.a2*x**2+self.a1*x+self.a0)/(self.b2*x**2+self.b1*x+self.b0)
        return y

class mlp(nn.Module):
    def __init__(self, n_hidden, input_size=4, output_size=4):
        super(mlp, self).__init__()
        layers = [nn.Linear(input_size, n_hidden),
                  rational(),
                  nn.Linear(n_hidden, output_size)]
        self.nn = nn.Sequential(*layers)

    def forward(self, p):

        y = self.nn(p)
        return y

class NeurVec(nn.Module):
    def __init__(self):
        super(NeurVec, self).__init__()

        self.error = mlp(n_hidden=1024, input_size=4, output_size=4)


if __name__ == '__main__':
    a = mlp(40, 40)
    b = torch.randn(10, 40)
    print(a(b).size())