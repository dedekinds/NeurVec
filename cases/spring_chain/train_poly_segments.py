import argparse
import os
import time
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import numpy as np
from utils import Logger, mkdir_p, AverageMeter
from dataloader_segments import Dataset_training, Dataset_testing
from torch.utils import data
import math
from tools import numerically_integrate_rk4,numerically_integrate_euler
from models import SprChainHamilt

parser = argparse.ArgumentParser(description='PyTorch')
# Datasets
parser.add_argument('--train_dir', default='', type=str, help='train set dir')
parser.add_argument('--test_dir', default='', type=str, help='test set dir')
parser.add_argument('--ckpt', default='test', type=str, help='checkpoint')

parser.add_argument('--optim', default='sgd', type=str, help='optimizer')
parser.add_argument('--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--gpu_id', default=0, type=str, help='gpu id')

parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--batch_size', default=32, type=int, help='batch size')
parser.add_argument('--epoch', default=1000, type=int, help='epoch')

parser.add_argument('--train_coarse', default=20, type=int, help='train coarse')

parser.add_argument('--dt', default=0.1, type=float, help='dt')
parser.add_argument('--T_train', default=10, type=int, help='train time step')
parser.add_argument('--T_test', default=10, type=int, help='test time step')
parser.add_argument('--length', default=99, type=int, help='length')

parser.add_argument('--manualSeed', default=None, type=int,help='seed')

parser.add_argument('--model_name', type=str, help='')
parser.add_argument('--num_layer', default=1, type=int, help='number of hidden layers')
parser.add_argument('--solver', default='rk4', type=str, help='euler or rk4')
args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}
print(state)

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = 1
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
np.random.seed(args.manualSeed)

if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

def main():
    if not os.path.isdir(args.ckpt):
        mkdir_p(args.ckpt)

    with open(args.ckpt + "/config.txt", 'w+') as f:
        for (k, v) in args._get_kwargs():
            f.write(k + ' : ' + str(v) + '\n')

    # Model
    if args.model_name == 'orgNN':
        model = SprChainHamilt()
    else:
        model = None
    model = model.cuda()
    cudnn.benchmark = True

    print('Total params: %.6f' % (sum(p.numel() for p in model.parameters())))

    """
    Define Residual Methods and Optimizer
    """
    criterion = nn.MSELoss()
    if args.optim == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)

    title = ''
    logger = Logger(os.path.join(args.ckpt, 'log.txt'), title=title)
    logger.set_names(['Learning Rate', 'Train Mse', 'Test Mse'])
    # Train and test
    training_set = Dataset_training(data_dir=args.train_dir, train_T=args.T_train, seg_num=args.length)
    trainloader = data.DataLoader(training_set, shuffle=True, batch_size=args.batch_size, num_workers=args.workers)

    testing_set = Dataset_testing(data_dir=args.test_dir)
    testloader = data.DataLoader(testing_set, shuffle=False, batch_size=150, num_workers=args.workers)

    current_iters = 0
    iters_per_epoch = len(trainloader)
    for epoch in range(args.epoch):
        train_mse = AverageMeter()
        test_mse = AverageMeter()
        ## training
        for trajectories in trainloader:
            lr = cosine_lr(optimizer, args.lr, current_iters, iters_per_epoch * args.epoch)
            train_loss = train(trajectories, model, criterion, optimizer, epoch, current_iters, len(trainloader), lr)
            train_mse.update(train_loss, trajectories.size(0))
            current_iters += 1
        ## testing
        for trajectories in testloader:
            test_loss = test(trajectories, model, criterion, epoch)
            test_mse.update(test_loss, trajectories.size(0))
        logger.append([lr, train_mse.avg, test_mse.avg])

    # save model
        if epoch % 2 == 0:
            save_checkpoint({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
                            checkpoint=args.ckpt,filename='ckpt'+'_'+str(epoch)+'.pth.tar')

    save_checkpoint({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
                    checkpoint=args.ckpt)

    logger.close()

def train(trajectories, model, criterion, optimizer, epoch, current_iters, total_iters, lr):
    suffix = ''

    # switch to train mode
    model.train()
    start_time = time.time()

    # data
    trajectories = torch.FloatTensor(trajectories).cuda() # Bs*T*dim
    true_traj = trajectories.permute(1,0,2) # T*Bs*dim

    volatile = False

    coarse = args.train_coarse
    if args.solver == 'euler':
        predict_traj = numerically_integrate_euler(true_traj[0, :, :20], true_traj[0, :, 20:], model, args.T_train, args.dt, volatile, coarse=coarse) # T_train*Bs*dim
    if args.solver == 'rk4':
        predict_traj = numerically_integrate_rk4(true_traj[0, :, :20], true_traj[0, :, 20:], model, args.T_train, args.dt, volatile, coarse=coarse) # T_train*Bs*dim
    loss = criterion(predict_traj[1:], true_traj[1:]) #* 1000000
    # compute gradient and do optim step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # measure elapsed time
    batch_time = time.time() - start_time

    suffix += 'Training epoch: {epoch} iter: {iter:.0f} / {total_iters:.0f}|lr: {lr:.4f}| Batch: {bt:.2f}s | training MSE: {mse: .5f} |'.format(bt=batch_time,
            iter=current_iters, lr=lr, mse=loss.item(), epoch=epoch, total_iters=total_iters)
    print(suffix)

    return loss.item()

def test(trajectories, model, criterion, epoch):
    suffix = ''

    # switch to evaluation mode
    model.eval()
    start_time = time.time()

    # data
    trajectories = torch.FloatTensor(trajectories).cuda() # Bs*T*dim
    true_traj = trajectories.permute(1,0,2) # T*Bs*dim

    volatile = True
    coarse = 1

    if args.solver == 'euler':
        predict_traj = numerically_integrate_euler(true_traj[0, :, :20], true_traj[0, :, 20:], model, args.T_test, args.dt, volatile, coarse=coarse) # T_train*Bs*dim
    if args.solver == 'rk4':
        predict_traj = numerically_integrate_rk4(true_traj[0, :, :20], true_traj[0, :, 20:], model, args.T_test, args.dt, volatile, coarse=coarse) # T_train*Bs*dim
    # print(predict_traj.size(), true_traj.size())
    loss = criterion(predict_traj, true_traj[:args.T_test])

    # measure elapsed time
    batch_time = time.time() - start_time

    suffix += 'Testing epoch: {epoch} | Batch: {bt:.2f}s | training MSE: {mse: .5f} |'.format(bt=batch_time, mse=loss.item(), epoch=epoch)
    print(suffix)

    return loss.item()

def save_checkpoint(state, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)

def cosine_lr(opt, base_lr, e, epochs):
    lr = 0.5 * base_lr * (math.cos(math.pi * e / epochs) + 1)
    for param_group in opt.param_groups:
        param_group["lr"] = lr
    return lr

if __name__ == '__main__':
    main()

