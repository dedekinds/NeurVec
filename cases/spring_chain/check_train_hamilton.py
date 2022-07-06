import numpy as np
import torch
from torch.autograd import grad
from tqdm import tqdm
import torch.nn as nn
import os
from torch.nn.parameter import Parameter

os.environ['CUDA_VISIBLE_DEVICES'] = str(0)

import torch
import torch.nn as nn
import torch.nn.parallel
import numpy as np
from torch.nn.parameter import Parameter
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


class SprChainHamilt(nn.Module):
    def __init__(self):
        super(SprChainHamilt, self).__init__()

        chain_param = np.load('../../data/spring_chain/chain_params.npz')
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


class f(nn.Module):
    def forward(self, u):
        p = u[:, :u.size(1)//2]
        q = u[:, u.size(1)//2:]
        
        chain_param = np.load('../../data/spring_chain/chain_params.npz')
        k = chain_param['k_true']
        m = chain_param['m_true']

        self.m = torch.from_numpy(m).cuda()
        self.k = torch.from_numpy(k).cuda()
        
        dq1 = self.k[0] * q[:, 0] + self.k[1] * (q[:, 0] -q[:, 1]) 
        dqm = self.k[-2] * (q[:,-1] - q[:,-2]) + self.k[-1] * q[:,-1] 
        result = -1 * dq1.unsqueeze(1)
        for i in range(1,len(q[0,:])-1):
            temp_result = self.k[i] * (q[:, i] - q[:, i-1]) + self.k[i+1] * (q[:,i] - q[:, i+1])    

            result = torch.cat( (result, -1 * temp_result.unsqueeze(1)    ),1)
        result = torch.cat( (result, -1 * dqm.unsqueeze(1), p/self.m),1)
        
        return result

def rk4(p_0, q_0, Func, T, dt, volatile=True, use_tqdm=False):

    trajectories = torch.empty((T, p_0.shape[0], 2 * p_0.shape[1]), requires_grad=False).cuda()
    p = p_0
    q = q_0
    u = torch.cat((p,q),1)

    range_of_for_loop = range(T)

    F = f()

    for i in range_of_for_loop:
        if volatile:
            trajectories[i, :, :] = u.detach()
        else:
            trajectories[i, :, :] = u
        error = Func.error(u)

        k1 = F(u)
        k2 = F(u+k1*dt/2)
        k3 = F(u+k2*dt/2)
        k4 = F(u+k3*dt)

        u = u + (1/6*k1 + 1/3*k2 + 1/3*k3 + 1/6*k4)*dt + error

    return trajectories



def euler(p_0, q_0, Func, T, dt, volatile=True, use_tqdm=False):

    trajectories = torch.empty((T, p_0.shape[0], 2 * p_0.shape[1]), requires_grad=False).cuda()

    p = p_0
    q = q_0
    u = torch.cat((p,q),1)
    range_of_for_loop = range(T)
    F = f()
    for i in range_of_for_loop:

        if volatile:
            trajectories[i, :, :] = u.detach()
        else:
            trajectories[i, :, :] = u
        error = Func.error(u)
        euler_term = F(u)



        u = u + euler_term*dt + error

    return trajectories

def numerically_integrate_euler(p_0, q_0, model, T, dt, volatile, coarse=1):
    trajectory_simulated = euler(p_0, q_0, model, (T-1)*coarse+1, dt/coarse, volatile=volatile)
    trajectory_simulated = trajectory_simulated[::coarse]
    return trajectory_simulated

def numerically_integrate_rk4(p_0, q_0, model, T, dt, volatile, coarse=1):
    trajectory_simulated = rk4(p_0, q_0, model, (T-1)*coarse+1, dt/coarse, volatile=volatile)
    trajectory_simulated = trajectory_simulated[::coarse]
    return trajectory_simulated



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch')
    parser.add_argument('--ckpt', default='test', type=str, help='checkpoint')
    parser.add_argument('--dt', type=float, help='dt')
    parser.add_argument('--solver', default='rk4', type=str, help='euler or rk4')
    args = parser.parse_args()

    ckpt = args.ckpt
    T = 100
    dt = args.dt
    volatile = False

    model = SprChainHamilt().cuda()
    model.load_state_dict(torch.load(ckpt+'/checkpoint.pth.tar')['state_dict'])

    for index in range(2):
        test_data = np.load('../../data/spring_chain/testset/index'+str(index)+'_150_dt_0.2_coarse2000.npy')  ##
        if args.solver == 'euler':
            trajectories = numerically_integrate_euler(torch.from_numpy(test_data[0,:,:20]).cuda(), torch.from_numpy(test_data[0,:,20:]).cuda(), model, T, dt, volatile)
        if args.solver == 'rk4':
            trajectories = numerically_integrate_rk4(torch.from_numpy(test_data[0,:,:20]).cuda(), torch.from_numpy(test_data[0,:,20:]).cuda(), model, T, dt, volatile)
        np.save(args.solver+'_test_neurvec_'+str(index), trajectories.cpu().detach().numpy())

        del(trajectories)
        torch.cuda.empty_cache()
        time.sleep(3)

