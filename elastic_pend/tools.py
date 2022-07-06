import torch
from torch.autograd import grad
from tqdm import tqdm
import torch.nn as nn


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

    u = u_0

    range_of_for_loop = range(T)

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

        u = u + (1/6*k1 + 1/3*k2 + 1/3*k3 + 1/6*k4)*dt + 1*error

    return trajectories


def numerically_integrate_rk4(u_0, model, T, dt, volatile, coarse=1):
    trajectory_simulated = rk4(u_0, model, (T-1)*coarse+1, dt/coarse, volatile=volatile)
    trajectory_simulated = trajectory_simulated[::coarse]
    return trajectory_simulated