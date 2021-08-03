import torch.nn as nn
from lib_qf import *
from lib_bn import *
from lib_vqc import *
import torch
import sys

## Define the NN architecture
class Net(nn.Module):
    def __init__(self,img_size,layers,with_norm,training,binary,debug="False"):
        super(Net, self).__init__()

        self.in_size = img_size*img_size
        self.training = training
        self.with_norm = with_norm
        self.layer = len(layers)
        self.layers = layers
        self.binary = binary
        loop_in_size = self.in_size
        self.debug = debug
        for idx in range(self.layer):
            fc_name = "fc"+str(idx)
            if layers[idx][0]=='u':
                setattr(self, fc_name, BinaryLinearQuantumFirstLAYER(loop_in_size, layers[idx][1], bias=False))
            elif layers[idx][0]=='p':
                setattr(self, fc_name, BinaryLinear(loop_in_size, layers[idx][1], bias=False))
            elif layers[idx][0]=='v' and idx == 0:
                if(int(torch.log(torch.tensor(loop_in_size))/torch.log(torch.tensor(2)))!=torch.log(torch.tensor(loop_in_size))/torch.log(torch.tensor(2))):
                    print("Not support input size!")
                    sys.exit(0)
                setattr(self, fc_name, VQC_Net(int(torch.log(torch.tensor(loop_in_size))/torch.log(torch.tensor(2))), layers[idx][1]))
            elif layers[idx][0]=='v':
                setattr(self, fc_name, VQC_Net(loop_in_size, layers[idx][1]))
            loop_in_size = layers[idx][1]

        if self.with_norm:
            # quantum batch normal
            for idx in range(self.layer):
                if idx==0:
                    continue
                elif layers[idx][0]=='p' or layers[idx][0]=='u':
                    qca_name = "qca"+str(idx)
                    setattr(self, qca_name, QC_Norm_Correction_try2(num_features=layers[idx][1]))



    def forward(self, x, training=1):
        if self.layers[0][0]=="v":
            x = x.double()
        x = x.view(-1, self.in_size)
        for layer_idx in range(self.layer):
            if self.binary and layer_idx==0:
                x = (binarize(x - 0.5) + 1) / 2
            # print("input-",layer_idx, x)
            x = getattr(self, "fc" + str(layer_idx))(x)
            if self.layers[layer_idx][0]=='u':
            # if layer_idx == 'u':
                    x = pow(x,2)
            elif self.with_norm:
                if self.layers[layer_idx][0]=='p' or self.layers[layer_idx][0]=='u':
                    x = getattr(self, "qca"+str(layer_idx))(x,training=self.training)

        if self.layers[-1][1] == 1:
            x = torch.cat((x, 1 - x), -1)

        return x


