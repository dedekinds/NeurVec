import numpy as np
import torch
from torch.autograd import grad
from tqdm import tqdm
import torch.nn as nn
import os
from torch.nn.parameter import Parameter
import torch.nn.parallel

os.environ['CUDA_VISIBLE_DEVICES'] = str(0)



import time
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

class NeurVec(nn.Module):
    def __init__(self):
        super(NeurVec, self).__init__()

        self.error = mlp(n_hidden=1024, input_size=4, output_size=4)



class f(nn.Module):
    def forward(self, u):
        p = u[:, :u.size(1)//2]
        q = u[:, u.size(1)//2:]
        return torch.cat((-(q[:,0:1] + 2*q[:,0:1]*q[:,1:2]), -(q[:,1:2] + q[:,0:1]*q[:,0:1] - q[:,1:2]*q[:,1:2]), p),1)

l0 = 10             #spring at rest
g = 9.81          #gravity
m = 1             #mass of particle
k = 40            #spring constant



def F(u):
    size = u.size(1)//4
    th, r, Vth, Vr = u[:, :size], u[:, size:size*2], u[:, size*2:size*3], u[:, size*3:size*4]
    Ath = (-g * torch.sin(th) - 2 * Vth * Vr) / r
    Ar = r * Vth ** 2 - k / m * (r - l0) + g * torch.cos(th)
    return torch.cat((Vth, Vr, Ath, Ar), 1)

def rk4(u_0, Func, T, dt, volatile=True):

    trajectories = torch.empty((T, u_0.shape[0], u_0.shape[1]), requires_grad=False).cuda()
    print('trajectories shape',trajectories.shape)
    u = u_0

    range_of_for_loop = range(T)

    for i in range_of_for_loop:
        # print('u shape',u.shape)

        if volatile:
            trajectories[i, :, :] = u.detach()
        else:
            trajectories[i, :, :] = u

        error = Func.error(u)

        k1 = F(u)
        k2 = F(u+k1*dt/2)
        k3 = F(u+k2*dt/2)
        k4 = F(u+k3*dt)

        u = u + (1/6*k1 + 1/3*k2 + 1/3*k3 + 1/6*k4)*dt + 1*error

    return trajectories


def numerically_integrate_rk4(u_0, model, T, dt, volatile, coarse=1):
    trajectory_simulated = rk4(u_0, model, (T-1)*coarse+1, dt/coarse, volatile=volatile)
    trajectory_simulated = trajectory_simulated[::coarse]
    return trajectory_simulated


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='PyTorch')
    parser.add_argument('--ckpt', default='test', type=str, help='checkpoint')
    parser.add_argument('--dt', type=float, help='dt')

    args = parser.parse_args()

    ckpt = args.ckpt
    model = NeurVec().cuda()
    model.load_state_dict(torch.load(ckpt+'/ckpt_500.pth.tar')['state_dict'])
    T = 500
    dt = args.dt
    volatile = False

    for index in range(2):
        print(index)

        test_data = np.load('../data/elastic_pend/testset/springball_'+str(index)+'_0.1_coarse1000.npy')
        trajectories = numerically_integrate_rk4(torch.from_numpy(test_data[0,:,:]).cuda(), model, T, dt, volatile)

        np.save('test_neurvec_'+str(index), trajectories.cpu().detach().numpy())
        del(trajectories)
        torch.cuda.empty_cache()
        time.sleep(5)