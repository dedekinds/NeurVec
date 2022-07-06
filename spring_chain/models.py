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
        # print(input_size, n_hidden)
        layers = [nn.Linear(input_size, n_hidden),
                  rational(),
                  nn.Linear(n_hidden, output_size)]

        self.nn = nn.Sequential(*layers)



    def forward(self, p):
        y = self.nn(p)
        return y


class SprChainHamilt(nn.Module):
    def __init__(self):
        super(SprChainHamilt, self).__init__()

        chain_param = np.load('../data/spring_chain/chain_params.npz')
        k = chain_param['k_true']
        m = chain_param['m_true']

        self.m = torch.from_numpy(m).cuda()
        self.k = torch.from_numpy(k).cuda()

        self.null = Parameter(torch.Tensor(10 + 1, 10 + 1))
        self.error = mlp(n_hidden=1024, input_size=40, output_size=40)

    def forward(self, p_vec, q_vec):
        p, q = p_vec, q_vec
        kinetic = 0.5 * (p.pow(2) / self.m).sum(dim=1)
        q_diff = q[:, :-1] - q[:, 1:]
        potential = 0.5 * (self.k[1:-1] * q_diff.pow(2)).sum(dim=1) + 0.5 * (self.k[0] * q[:, 0].pow(2)) + 0.5 * (self.k[-1] * q[:, -1].pow(2))
        return kinetic + potential



if __name__ == '__main__':
    a = mlp(40, 40)
    b = torch.randn(10, 40)
    print(a(b).size())