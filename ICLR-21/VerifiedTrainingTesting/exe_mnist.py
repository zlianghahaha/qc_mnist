# %%

# import libraries

import torch
import numpy as np
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
from torchsummary import summary
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np
import math

import shutil
import os
import time
import sys
from pathlib import Path
import functools
from collections import Counter
import argparse
print = functools.partial(print, flush=True)

from lib_net import *
from lib_util import *


def parse_args():
    parser = argparse.ArgumentParser(description='QuantumFlow Classification Training')

    # ML related
    # parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-c','--interest_class',default="3, 6",help="investigate classes",)
    parser.add_argument('-r', '--run_num', default="0", help="investigate classes", )
    # QC related
    parser.add_argument('-nq', "--classic", help="classic computing test", action="store_true", )


    args = parser.parse_args()
    return args

args = parse_args()
interest_num = [int(x.strip()) for x in args.interest_class.split(",")]
classical_eval = args.classic

if classical_eval:
    name = "C-"
else:
    name = "Q-"
name+=args.interest_class+"_"+args.run_num

img_size = 28
# number of subprocesses to use for data loading
num_workers = 0
# how many samples per batch to load
batch_size = 16
inference_batch_size = 16
num_f1 = 16
num_f2 = len(interest_num)
init_lr = 0.01

save_to_file = False
if save_to_file:
    sys.stdout = open(save_path + "/log", 'w')
# save_path = "./model/" + os.path.basename(sys.argv[0]) + "_" + time.strftime("%Y_%m_%d-%H_%M_%S")
save_path = "./model/"+name
Path(save_path).mkdir(parents=True, exist_ok=True)

resume_path = ""
training = True
max_epoch = 10

criterion = nn.CrossEntropyLoss()
# criterion = nn.MSELoss()

milestones = [3, 5, 8]

print("=" * 100)
print("Training procedure for Quantum Computer:")
print("\tStart at:", time.strftime("%m/%d/%Y %H:%M:%S"))
print("\tProblems and issues, please contact Dr. Weiwen Jiang (wjiang2@nd.edu)")
print("\tEnjoy and Good Luck!")
print("=" * 100)
print()

# %%

# convert data to torch.FloatTensor
transform = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor()])
# transform = transforms.Compose([transforms.Resize((img_size,img_size)),transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
# choose the training and test datasets
train_data = datasets.MNIST(root='../../pytorch/data', train=True,
                            download=True, transform=transform)
test_data = datasets.MNIST(root='../../pytorch/data', train=False,
                           download=True, transform=transform)

train_data = select_num(train_data, interest_num)
test_data = select_num(test_data, interest_num)

# prepare data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                           num_workers=num_workers, shuffle=True, drop_last=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=inference_batch_size,
                                          num_workers=num_workers, shuffle=True, drop_last=True)


def save_checkpoint(state, is_best, save_path, filename):
    filename = os.path.join(save_path, filename)
    torch.save(state, filename)
    if is_best:
        bestname = os.path.join(save_path, 'model_best.tar')
        shutil.copyfile(filename, bestname)


# %%


def train(epoch):
    model.train()
    correct = 0
    epoch_loss = []
    for batch_idx, (data, target) in enumerate(train_loader):
        target, new_target = modify_target(target, interest_num)
        #
        # data = (data-data.min())/(data.max()-data.min())
        # data = (binarize(data-0.5)+1)/2
        #

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data, True)

        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        loss = criterion(output, target)
        epoch_loss.append(loss.item())
        loss.backward()

        optimizer.step()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {}/{} ({:.2f}%)'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss, correct, (batch_idx + 1) * len(data),
                       100. * float(correct) / float(((batch_idx + 1) * len(data)))))
    print("-" * 20, "training done, loss", "-" * 20)
    print("Training Set: Average loss: {}".format(round(sum(epoch_loss) / len(epoch_loss), 6)))


accur = []


def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        target, new_target = modify_target(target, interest_num)

        #
        # data = (data-data.min())/(data.max()-data.min())
        # data = (binarize(data-0.5)+1)/2

        data, target = data.to(device), target.to(device)

        # print("Debug")
        # output = model(data,2)
        #
        # sys.exit(0)
        # data, target = Variable(data, volatile=True), Variable(target)
        output = model(data, False)
        test_loss += criterion(output, target)  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    a = 100. * correct / len(test_loader.dataset)
    accur.append(a)
    test_loss /= len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * float(correct) / float(len(test_loader.dataset))))

    return float(correct) / len(test_loader.dataset)


# Training


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net(img_size, [num_f1, num_f2], True, [[1, -1, 1, -1], [-1, -1]],
            True, training, False, classical_eval, False).to("cpu")

# -nn "4, 2" -bin -qt -c $dataset -s 4 -l 0.1 -ql 0.0001 -e 5 -m "2, 4"
# def __init__(self,img_size,layers,with_norm,given_ang,train_ang,training,binary,classic,debug="False"):


print("=" * 10, "Model Info", "=" * 10)
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)

scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1)

#
#
# test()
#
#

# %%

if os.path.isfile(resume_path):
    print("=> loading checkpoint from '{}'<=".format(resume_path))
    checkpoint = torch.load(resume_path, map_location=device)
    epoch_init, acc = checkpoint["epoch"], checkpoint["acc"]
    model.load_state_dict(checkpoint["state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    scheduler.milestones = Counter(milestones)
    optimizer.load_state_dict(checkpoint["optimizer"])
else:
    epoch_init, acc = 0, 0

if training:
    for epoch in range(epoch_init, max_epoch + 1):
        print("=" * 20, epoch, "epoch", "=" * 20)
        print("Epoch Start at:", time.strftime("%m/%d/%Y %H:%M:%S"))

        print("-" * 20, "learning rates", "-" * 20)
        for param_group in optimizer.param_groups:
            print(param_group['lr'], end=",")
        print()

        print("-" * 20, "training", "-" * 20)
        print("Trainign Start at:", time.strftime("%m/%d/%Y %H:%M:%S"))
        train(epoch)
        print("Trainign End at:", time.strftime("%m/%d/%Y %H:%M:%S"))
        print("-" * 60)

        print()

        print("-" * 20, "testing", "-" * 20)
        print("Testing Start at:", time.strftime("%m/%d/%Y %H:%M:%S"))
        cur_acc = test()
        print("Testing End at:", time.strftime("%m/%d/%Y %H:%M:%S"))
        print("-" * 60)
        print()

        scheduler.step()

        is_best = False
        if cur_acc > acc:
            is_best = True
            acc = cur_acc

        print("Best accuracy: {}; Current accuracy {}. Checkpointing".format(acc, cur_acc))
        save_checkpoint({
            'epoch': epoch + 1,
            'acc': acc,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }, is_best, save_path, 'checkpoint_{}_{}.pth.tar'.format(epoch, round(cur_acc, 4)))
        print("Epoch End at:", time.strftime("%m/%d/%Y %H:%M:%S"))
        print("=" * 60)
        print()
else:
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1,
                                              num_workers=num_workers, shuffle=True, drop_last=True)
    test()

