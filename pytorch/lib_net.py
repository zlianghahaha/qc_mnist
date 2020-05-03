import torch.nn as nn
from lib_util import *
from lib_qc import *
import torch

## Define the NN architecture
class Net(nn.Module):
    def __init__(self,img_size,layers,with_norm,given_ang,train_ang,training,binary,classic):
        super(Net, self).__init__()

        # self.fc = []
        # self.qc = []
        # self.qca = []
        self.in_size = img_size*img_size
        self.training = training
        self.with_norm = with_norm
        self.layer = len(layers)
        self.binary = binary
        self.classic = classic
        loop_in_size = self.in_size
        for idx in range(self.layer):
            fc_name = "fc"+str(idx)
            if classic:
                setattr(self, fc_name, BinaryLinearClassic(loop_in_size, layers[idx], bias=False))
            else:
                setattr(self, fc_name, BinaryLinear(loop_in_size, layers[idx], bias=False))
            loop_in_size = layers[idx]

        if self.with_norm:
            for idx in range(self.layer):
                qc_name = "qc"+str(idx)
                qca_name = "qca"+str(idx)
                setattr(self, qc_name, QC_Norm_try3(num_features=layers[idx], init_ang_inc=given_ang[idx], training=train_ang))
                setattr(self, qca_name, QC_Norm_Correction_try2(num_features=layers[idx]))
            for idx in range(self.layer):
                bn_name = "bn"+str(idx)
                setattr(self, bn_name,nn.BatchNorm1d(num_features=layers[idx]))


    def forward(self, x, training=1):
        x = x.view(-1, self.in_size)

        if self.classic == 1 and self.with_norm==0:
            for layer_idx in range(self.layer):
                if self.binary:
                    x = (binarize(x - 0.5) + 1) / 2
                x = getattr(self, "fc" + str(layer_idx))(x)
                x = x.pow(2)

        elif self.classic == 1 and self.with_norm==1:
            for layer_idx in range(self.layer):
                if self.binary:
                    x = (binarize(x - 0.5) + 1) / 2
                x = getattr(self, "fc" + str(layer_idx))(x)
                x = x.pow(2)
                x = getattr(self, "bn" + str(layer_idx))(x)
                x = clipfunc(x)





        elif self.classic == 0 and self.with_norm==0:
            for layer_idx in range(self.layer):
                if self.binary:
                    x = (binarize(x - 0.5) + 1) / 2
                x = getattr(self, "fc" + str(layer_idx))(x)

        else:   # Quantum Training
            if self.training == 1:
                for layer_idx in range(self.layer):
                    if self.binary:
                        x = (binarize(x-0.5)+1)/2
                    x = getattr(self, "fc"+str(layer_idx))(x)
                    x = getattr(self, "qca"+str(layer_idx))(x)
                    x = getattr(self, "qc"+str(layer_idx))(x)
            else:
                for layer_idx in range(self.layer):
                    if self.binary:
                        x = (binarize(x-0.5)+1)/2
                    x = getattr(self, "fc"+str(layer_idx))(x)
                    x = getattr(self, "qca"+str(layer_idx))(x, training=False)
                    x = getattr(self, "qc"+str(layer_idx))(x, training=False)

        # if num_f2 == 1:
        #     x = torch.cat((x, 1 - x), -1)

        return x


