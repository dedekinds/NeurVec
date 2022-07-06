import torch
from torch.autograd import grad
from tqdm import tqdm
import torch.nn as nn
import numpy as np
def leapfrog(p_0, q_0, Func, T, dt, volatile=True, use_tqdm=False):

    trajectories = torch.empty((T, p_0.shape[0], 2 * p_0.shape[1]), requires_grad=False).cuda()

    p = p_0
    q = q_0
    p.requires_grad_()
    q.requires_grad_()

    if use_tqdm:
        range_of_for_loop = tqdm(range(T))
    else:
        range_of_for_loop = range(T)

    hamilt = Func(p, q)
    dpdt = -grad(hamilt.sum(), q, create_graph=not volatile)[0]

    for i in range_of_for_loop:
        p_half = p + dpdt * (dt / 2)

        if volatile:
            trajectories[i, :, :p_0.shape[1]] = p.detach()
            trajectories[i, :, p_0.shape[1]:] = q.detach()
        else:
            trajectories[i, :, :p_0.shape[1]] = p
            trajectories[i, :, p_0.shape[1]:] = q

        hamilt = Func(p_half, q)
        dqdt = grad(hamilt.sum(), p, create_graph=not volatile)[0]

        q_next = q + dqdt * dt

        hamilt = Func(p_half, q_next)
        dpdt = -grad(hamilt.sum(), q_next, create_graph=not volatile)[0]

        p_next = p_half + dpdt * (dt / 2)

        p = p_next
        q = q_next

    return trajectories




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



