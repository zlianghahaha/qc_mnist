import torch.nn as nn
from lib_util import *
from lib_qc import *
import torch

## Define the NN architecture
class Net(nn.Module):
    def __init__(self,img_size,layers,with_norm,given_ang,train_ang,training,binary,classic,debug="False"):
        super(Net, self).__init__()
        print(img_size,layers,with_norm,given_ang,train_ang,training,binary,classic,debug)
        # self.fc = []
        # self.qc = []
        # self.qca = []
        self.in_size = img_size*img_size
        self.training = training
        self.with_norm = with_norm
        self.layer = len(layers)
        self.layers = layers
        self.binary = binary
        self.classic = classic
        loop_in_size = self.in_size
        self.debug = debug
        for idx in range(self.layer):
            fc_name = "fc"+str(idx)
            if classic:
                setattr(self, fc_name, BinaryLinearClassic(loop_in_size, layers[idx], bias=False))
            elif idx==0:
                setattr(self, fc_name, BinaryLinearQuantumFirstLAYER(loop_in_size, layers[idx], bias=False))
            else:
                setattr(self, fc_name, BinaryLinearClassic(loop_in_size, layers[idx], bias=False))
                # setattr(self, fc_name, BinaryLinear(loop_in_size, layers[idx], bias=False))
            loop_in_size = layers[idx]

        if self.with_norm:
            if not classic:
                for idx in range(self.layer):
                    if idx==0:
                        continue
                    qc_name = "qc"+str(idx)
                    qca_name = "qca"+str(idx)
                    # setattr(self, qc_name, QC_Norm_try3(num_features=layers[idx], init_ang_inc=given_ang[idx], training=train_ang))
                    setattr(self, qca_name, QC_Norm_Correction_try2(num_features=layers[idx]))
            else:
                for idx in range(self.layer):
                    bn_name = "bn"+str(idx)
                    setattr(self, bn_name,nn.BatchNorm1d(num_features=layers[idx]))


    def forward(self, x, training=1):
        x = x.view(-1, self.in_size)

        if self.classic == 1 and self.with_norm==0:
            for layer_idx in range(self.layer):
                if self.binary and layer_idx==0:
                    # x = (binarize(x - 0.5) + 1) / 2
                    x = binarize(x-0.5)
                # if self.training == 0:
                #     print(x)
                x = getattr(self, "fc" + str(layer_idx))(x)
                # if self.training == 0:
                #     print(x)

                x = nn.ReLU()(x)
                # x = x.pow(2)

            #     if self.training == 0:
            #         print(x)
            #
            # if self.training == 0:
            #     sys.exit(0)
        elif self.classic == 1 and self.with_norm==1:
            for layer_idx in range(self.layer):
                if self.binary and layer_idx==0:
                    x = (binarize(x - 0.5) + 1) / 2
                x = getattr(self, "fc" + str(layer_idx))(x)
                # x = x.pow(2)
                x = nn.ReLU()(x)
                x = getattr(self, "bn" + str(layer_idx))(x)
                x = clipfunc(x)





        elif self.classic == 0 and self.with_norm==0:
            for layer_idx in range(self.layer):
                if self.binary and layer_idx==0:
                    x = (binarize(x - 0.5) + 1) / 2
                # if self.training == 0:
                #     print(x)
                x = getattr(self, "fc" + str(layer_idx))(x)
                # if self.training == 0:
                #     print(x)

            # if self.training == 0:
            #     sys.exit(0)

        else:   # Quantum Training
            if self.training == 1:
                for layer_idx in range(self.layer):
                    if self.binary and layer_idx==0:
                        x = (binarize(x-0.5)+1)/2
                    x = getattr(self, "fc"+str(layer_idx))(x)
                    if layer_idx==0:
                        x = x.pow(2)
                    if layer_idx!=0:
                        x = nn.ReLU()(x)
                        # x = getattr(self, "qca"+str(layer_idx))(x)
                        # x = getattr(self, "qc"+str(layer_idx))(x)
            else:
                for layer_idx in range(self.layer):
                    if self.binary and layer_idx==0:
                        x = (binarize(x-0.5)+1)/2

                    if self.debug:
                        print("\t",x)
                    x = getattr(self, "fc"+str(layer_idx))(x)
                    if layer_idx==0:
                        x = x.pow(2)
                    if self.debug:
                        print("\t",x)

                    if layer_idx!=0:
                        x = nn.ReLU()(x)
                        # x = getattr(self, "qca"+str(layer_idx))(x, training=False)
                        if self.debug:
                            print("\t", x)

                        # x = getattr(self, "qc"+str(layer_idx))(x, training=False)
                        # if self.debug:
                        #     print("\t", x)

        if self.layers[-1] == 1:
            x = torch.cat((x, 1 - x), -1)

        return x


