# %%

# import libraries
import logging
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

from mnist import *

from collections import Counter

logging.basicConfig(stream=sys.stdout,
                    level=logging.WARNING,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
logger = logging.getLogger(__name__)

print = functools.partial(print, flush=True)

# For 4*4, 16->4->2: batch_size=32; init_lr=0.01; with_norm=True or False
# For 4*4, 16->4->1: batch_size=16; init_lr=0.1; with_norm=True, ang:20; or train

# interest_num = [0,1,2,3,4,5,6,7,8,9]
interest_num = [3, 6]
img_size = 4
# number of subprocesses to use for data loading
num_workers = 0
# how many samples per batch to load
batch_size = 32
inference_batch_size = 1
num_f1 = 4
# num_f2 = len(interest_num)
num_f2 = 1
init_lr = 0.1
init_qc_lr = 1
with_norm = False
save_chkp = False
# Given_ang to -1 to train the variable
given_ang = -1

save_to_file = False
if save_to_file:
    sys.stdout = open(save_path + "/log", 'w')

if save_chkp:
    save_path = "./model/" + os.path.basename(sys.argv[0]) + "_" + time.strftime("%Y_%m_%d-%H_%M_%S")
    Path(save_path).mkdir(parents=True, exist_ok=True)

resume_path = "./model/ipykernel_launcher.py_2020_04_22-15_15_31/model_best.tar"

# resume_path = ""
training = False
max_epoch = 10

criterion = nn.CrossEntropyLoss()
# criterion = nn.MSELoss()


print("=" * 100)
print("Training procedure for Quantum Computer:")
print("\tStart at:", time.strftime("%m/%d/%Y %H:%M:%S"))
print("\tProblems and issues, please contact Dr. Weiwen Jiang (wjiang2@nd.edu)")
print("\tEnjoy and Good Luck!")
print("=" * 100)
print()


# %%

def modify_target(target):
    for j in range(len(target)):
        for idx in range(len(interest_num)):
            if target[j] == interest_num[idx]:
                target[j] = idx
                break

    new_target = torch.zeros(target.shape[0], 2)

    for i in range(target.shape[0]):
        if target[i].item() == 0:
            new_target[i] = torch.tensor([1, 0]).clone()
        else:
            new_target[i] = torch.tensor([0, 1]).clone()

    return target, new_target


def select_num(dataset, interest_num):
    labels = dataset.targets  # get labels
    labels = labels.numpy()
    idx = {}
    for num in interest_num:
        idx[num] = np.where(labels == num)

    fin_idx = idx[interest_num[0]]
    for i in range(1, len(interest_num)):
        fin_idx = (np.concatenate((fin_idx[0], idx[interest_num[i]][0])),)

    fin_idx = fin_idx[0]

    dataset.targets = labels[fin_idx]
    dataset.data = dataset.data[fin_idx]

    # print(dataset.targets.shape)

    dataset.targets, _ = modify_target(dataset.targets)
    # print(dataset.targets.shape)

    return dataset


# convert data to torch.FloatTensor
transform = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor()])
# transform = transforms.Compose([transforms.Resize((img_size,img_size)),transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
# choose the training and test datasets
train_data = datasets.MNIST(root='data', train=True,
                            download=True, transform=transform)
test_data = datasets.MNIST(root='data', train=False,
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



from torch.nn.parameter import Parameter


class QC_Norm(nn.Module):
    def __init__(self, num_features, init_ang_inc=10, momentum=0.1):
        super(QC_Norm, self).__init__()

        self.x_running_rot = Parameter(torch.zeros(num_features), requires_grad=False)
        self.ang_inc = Parameter(torch.ones(1) * init_ang_inc)

        self.momentum = momentum

        self.printed = False
        self.x_mean_ancle = 0
        self.x_mean_rote = 0
        self.input = 0
        self.output = 0

    def forward(self, x, training=True):
        if not training:
            if not self.printed:
                print("self.ang_inc", self.ang_inc)
                self.printed = True

            x = x.transpose(0, 1)

            x_ancle = (x * 2 - 1).acos()
            x_final = x_ancle + self.x_running_rot.unsqueeze(-1)
            x_1 = (x_final.cos() + 1) / 2

            x_1 = x_1.transpose(0, 1)

        else:
            self.printed = False
            x = x.transpose(0, 1)
            x_sum = x.sum(-1).unsqueeze(-1).expand(x.shape)
            x_lack_sum = x_sum - x
            x_mean = x_lack_sum / x.shape[-1]

            x_mean_ancle = (x_mean * 2 - 1).acos()

            ang_inc = self.ang_inc.unsqueeze(-1).expand(x_mean_ancle.shape)
            # ang_inc = np.pi/2/(x.max(-1)[0].unsqueeze(-1).expand(x_mean_ancle.shape) -x.min(-1)[0].unsqueeze(-1).expand(x_mean_ancle.shape) )

            if given_ang != -1:
                x_mean_rote = (np.pi / 2 - x_mean_ancle) * given_ang
            else:
                x_mean_rote = (np.pi / 2 - x_mean_ancle) * ang_inc

            x_moving_rot = (x_mean_rote.sum(-1) / x.shape[-1])
            self.x_running_rot[:] = self.momentum * self.x_running_rot + \
                                    (1 - self.momentum) * x_moving_rot

            x_ancle = (x * 2 - 1).acos()
            x_final = x_ancle + x_mean_rote
            x_1 = (x_final.cos() + 1) / 2
            x_1 = x_1.transpose(0, 1)

        return x_1

    def reset_parameters(self):
        self.reset_running_stats()
        self.ang_inc.data.zeros_()


def print_degree(x, name="x"):
    print(name, x / np.pi * 180)


class QC_Norm_Real(nn.Module):
    def __init__(self, num_features, momentum=0.1):
        super(QC_Norm_Real, self).__init__()
        self.x_running_rot = Parameter(torch.zeros(num_features), requires_grad=False)
        self.momentum = momentum

        self.x_max = 0
        self.x_min = 0
        # print("Using Normal without real")

    def forward(self, x, training=True):
        if not training:
            x = x.transpose(0, 1)

            x_ancle = (x * 2 - 1).acos()
            # x_final = x_ancle+self.x_running_rot.unsqueeze(-1)
            x_final = ((x_ancle - self.x_min) / (self.x_max - self.x_min)) * np.pi

            x_1 = (x_final.cos() + 1) / 2
            x_1 = x_1.transpose(0, 1)

        else:

            x = x.transpose(0, 1)
            x_ancle = (x * 2 - 1).acos()
            x_rectify_ancle = (x_ancle.max(-1)[0] - x_ancle.min(-1)[0]).unsqueeze(-1).expand(x.shape)
            x_final = ((x_ancle - x_ancle.min(-1)[0].unsqueeze(-1)) / (x_rectify_ancle)) * np.pi

            x_moving_rot = x_final - x_ancle

            x_moving_rot_mean = x_moving_rot.sum(-1) / x.shape[-1]
            self.x_running_rot[:] = self.momentum * self.x_running_rot + \
                                    (1 - self.momentum) * x_moving_rot_mean

            self.x_max = self.momentum * x_ancle.max(-1)[0].unsqueeze(-1) + \
                         (1 - self.momentum) * self.x_max
            self.x_min = self.momentum * x_ancle.min(-1)[0].unsqueeze(-1) + \
                         (1 - self.momentum) * self.x_min

            x_1 = (x_final.cos() + 1) / 2
            x_1 = x_1.transpose(0, 1)

        return x_1


class QC_Norm_Real_Correction(nn.Module):
    def __init__(self, num_features, momentum=0.1):
        super(QC_Norm_Real_Correction, self).__init__()
        self.x_running_rot = Parameter(torch.zeros(num_features), requires_grad=False)
        self.momentum = momentum

    def forward(self, x, training=True):
        if not training:
            x = x.transpose(0, 1)

            x_ancle = (x * 2 - 1).acos()
            x_final = x_ancle + self.x_running_rot.unsqueeze(-1)
            x_1 = (x_final.cos() + 1) / 2
            x_1 = x_1.transpose(0, 1)

        else:

            x = x.transpose(0, 1)
            x_ancle = (x * 2 - 1).acos()
            x_moving_rot = -1 * (x_ancle.min(-1)[0])

            self.x_running_rot[:] = self.momentum * self.x_running_rot + \
                                    (1 - self.momentum) * x_moving_rot
            x_final = x_ancle + x_moving_rot.unsqueeze(-1)
            x_1 = (x_final.cos() + 1) / 2
            x_1 = x_1.transpose(0, 1)

        return x_1


class QC_Norm_Correction(nn.Module):
    def __init__(self, num_features, momentum=0.1):
        super(QC_Norm_Correction, self).__init__()
        self.x_running_rot = Parameter(torch.zeros(num_features), requires_grad=False)
        self.momentum = momentum

    def forward(self, x, training=True):
        if not training:
            x = x.transpose(0, 1)

            x_ancle = (x * 2 - 1).acos()
            x_final = x_ancle + self.x_running_rot.unsqueeze(-1)
            x_1 = (x_final.cos() + 1) / 2
            x_1 = x_1.transpose(0, 1)

        else:
            x = x.transpose(0, 1)
            x_sum = x.sum(-1).unsqueeze(-1).expand(x.shape)
            x_mean = x_sum / x.shape[-1]

            x_mean_ancle = (x_mean * 2 - 1).acos()
            x_mean_rote = (np.pi / 2 - x_mean_ancle)

            x_moving_rot = (x_mean_rote.sum(-1) / x.shape[-1])
            self.x_running_rot[:] = self.momentum * self.x_running_rot + \
                                    (1 - self.momentum) * x_moving_rot
            x_ancle = (x * 2 - 1).acos()
            x_final = x_ancle + x_mean_rote
            x_1 = (x_final.cos() + 1) / 2
            x_1 = x_1.transpose(0, 1)

        return x_1


## Define the NN architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = BinaryLinear(img_size * img_size, num_f1, bias=False)
        self.fc2 = BinaryLinear(num_f1, num_f2, bias=False)


        if with_norm:
            self.qc1 = QC_Norm(num_features=num_f1, init_ang_inc=10)
            self.qc2 = QC_Norm(num_features=num_f2, init_ang_inc=40)
            # self.qc3 = QC_Norm(num_features=num_f3)

            self.qc1a = QC_Norm_Correction(num_features=num_f1)
            self.qc2a = QC_Norm_Correction(num_features=num_f2)
            # self.qc3a = QC_Norm_Correction(num_features=num_f3)

    def forward(self, x, training=1):
        x = x.view(-1, img_size * img_size)

        if training == 1:
            if with_norm:
                x = self.qc1(self.qc1a(self.fc1(x)))
                x = self.qc2(self.qc2a(self.fc2(x)))
            else:
                x = self.fc1(x)
                x = self.fc2(x)
        elif training == 2:

            # x = binarize(x-0.0001)
            # x = (x+1)/2

            print("=" * 10, "layer 1", "=" * 10)
            print(x)
            torch.set_printoptions(profile="full")
            print(binarize(self.fc1.weight))
            torch.set_printoptions(profile="default")
            x = self.fc1(x)

            print("=" * 10, "layer 2", "=" * 10)
            print(x)
            torch.set_printoptions(profile="full")
            print(binarize(self.fc2.weight))
            torch.set_printoptions(profile="default")
            x = self.fc2(x)

            print("=" * 10, "results", "=" * 10)
            print(x)

        else:
            if with_norm:
                x = self.qc1(self.qc1a(self.fc1(x), training=False), training=False)
                x = self.qc2(self.qc2a(self.fc2(x), training=False), training=False)
            else:
                x = self.fc1(x)
                x = self.fc2(x)

        if num_f2 == 1:
            x = torch.cat((x, 1 - x), -1)

        return x


def train(epoch):
    model.train()
    correct = 0
    epoch_loss = []
    for batch_idx, (data, target) in enumerate(train_loader):
        target, new_target = modify_target(target)
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
        target, new_target = modify_target(target)

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

device = torch.device("cpu")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net().to(device)
print("=" * 10, "Model Info", "=" * 10)
print(model)

if with_norm and given_ang == -1:
    optimizer = torch.optim.Adam([
        {'params': model.fc1.parameters()},
        {'params': model.fc2.parameters()},
        # {'params': model.fc3.parameters()},
        {'params': model.qc1.parameters(), 'lr': init_qc_lr},
        {'params': model.qc2.parameters(), 'lr': init_qc_lr},
        # {'params': model.qc3.parameters(), 'lr': 1},
    ], lr=init_lr)
else:
    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)


milestones = [3, 7, 9]
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1)



import sys

sys.path.append("../interfae")
from qiskit_simulator import *

accur = []
accur_qc = []

def test_qc_sim():
    model.eval()
    test_loss = 0
    test_loss_qc = 0
    correct = 0
    correct_qc = 0

    # len(test_loader.dataset)
    num_test = 100
    idx_tes = 0


    for data, target in test_loader:
        logger.warning("{} iteration start".format(idx_tes))
        target, new_target = modify_target(target)
        data, target = data.to(device), target.to(device)

        # Theoretic Segmentation
        output = model(data, False)
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        test_loss += criterion(output, target)  # sum up batch loss
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        logger.warning("Theoretic Results {}, Target {}, Predict {}, Correct {}".format(output, target, pred, pred.eq(
            target.data.view_as(pred)).cpu().sum()))

        # QC Segmentation
        output_qc = run_simulator(model, data)
        pred_qc = output_qc.data.max(1, keepdim=True)[1]
        test_loss_qc += criterion(output_qc, target)
        correct_qc += pred_qc.eq(target.data.view_as(pred_qc)).cpu().sum()
        logger.warning("QC Sim Results {}, Target {}, Predict {}, Correct {}".format(output_qc,target,pred_qc,
            pred_qc.eq(target.data.view_as(pred_qc)).cpu().sum()))



        idx_tes += 1
        if idx_tes >= num_test:
            break


    accur.append(100. * correct / num_test)
    test_loss /= num_test
    print('Theoretic Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss, correct, num_test,
        100. * float(correct) / float(num_test)))

    accur_qc.append(100. * correct_qc / num_test)
    test_loss_qc /= num_test
    print('Quantum Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss_qc, correct_qc, num_test,
        100. * float(correct_qc) / float(num_test)))

    return float(correct) / len(test_loader.dataset)


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

test_loader = torch.utils.data.DataLoader(test_data, batch_size=1,
                                          num_workers=num_workers, shuffle=True, drop_last=True)
test_qc_sim()

