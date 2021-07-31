import torch
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.nn as nn
from torch.nn.parameter import Parameter
import numpy as np
import sys

################ zhirui on 12-30-2020 ################
# this block is for buding the matrix of variant circuit
######################################################
import math
class VClassicCircuitMatrix:
    def __init__(self, n_qubits):
        # --- parameter definition ---
        self._n_qubits = n_qubits
        #state = state
        #constant matrix
        self.mat_cz = torch.tensor([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,-1]])
        self.mat_identity = torch.tensor([[1,0],[0,1]])
        self.mat_swap = torch.tensor([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])

    def set_value(self,mm,col,row,val):
        index = (torch.LongTensor([col]),torch.LongTensor([row]))#生成索引
        mm = mm.index_put(index ,val)
        return mm

    def get_ry_matrix(self,theta):
        mm = torch.zeros(2,2,dtype=torch.float64)
        mm = self.set_value(mm,0,0,torch.cos(theta/2))
        mm = self.set_value(mm,0,1,-torch.sin(theta/2))
        mm = self.set_value(mm,1,0,torch.sin(theta/2))
        mm = self.set_value(mm,1,1,torch.cos(theta/2))
        return mm

    #swap (index) and (index-1)        
    def qf_swap_dec(self,state,index):
        #generate the matrix
        temp_mat = torch.ones(1,dtype=torch.float64)
        for i in range(0,index-1):
            temp_mat = torch.kron(self.mat_identity,temp_mat)
        temp_mat =torch.kron(self.mat_swap,temp_mat)
        for i in range(0,self._n_qubits-1-index):
            temp_mat = torch.kron(self.mat_identity,temp_mat)
        #change state
        state = torch.mm(temp_mat,state) 
        return state 


    def qf_ry(self,state,theta,index):

        temp_mat = torch.ones(1,dtype=torch.float64)
        if isinstance(index,int): 
            for i in range(0,self._n_qubits):
                if i == index:
                    mm = self.get_ry_matrix(theta)
                    temp_mat = torch.kron(mm,temp_mat) 
                else:
                    temp_mat = torch.kron(self.mat_identity,temp_mat) 
        else:
            for i in range(0,self._n_qubits):
                if i in index:
                    select_theta = torch.index_select(theta,0,torch.tensor([i]))
                    select_mm = self.get_ry_matrix(select_theta)
                    temp_mat = torch.kron(select_mm,temp_mat)
  
                else:
                    temp_mat = torch.kron(self.mat_identity,temp_mat) 
        #change state

        state = torch.mm(temp_mat,state)   

        return state
     
            
    def qf_cz(self,state,index1,index2):

        #generate the matrix
        
        #swap the bottom one next to the up one
        for i in range(index2,index1+1,-1):
            state = self.qf_swap_dec(state,i)
        
        #generate cz matrix
        temp_mat = torch.ones(1,dtype=torch.float64)
        for i in range(0,index1):
            temp_mat = torch.kron(self.mat_identity,temp_mat)

        temp_mat =torch.kron(self.mat_cz,temp_mat)

        for i in range(0,self._n_qubits-2-index1):
            temp_mat = torch.kron(self.mat_identity,temp_mat)

        #change state
        state = torch.mm(temp_mat,state)   

        #swap back
        for i in range(index1+2,index2+1):
            state = self.qf_swap_dec(state,i)

        return state

    #get the matrix transforming 16 output to 4 output.     
    def qf_sum(self):
        sum_mat = []
        flag = "0"+str(self._n_qubits)+"b"
        for i in range(0,int(math.pow(2,self._n_qubits))):
            bit_str = format(i,flag)
            row = []
            for c in bit_str:
                row.append(float(c))
            sum_mat.append(row)
        return sum_mat


    #the code is similar to the VQuantumCircuit::vqc_10
    def vqc_10(self,state,thetas):
        
        #head ry part 
        state = self.qf_ry(state,thetas[0:self._n_qubits],range(0,self._n_qubits))

        #cz part
        for i in range(self._n_qubits-1):
            state = self.qf_cz(state,self._n_qubits-2-i,self._n_qubits-1-i)
        state = self.qf_cz(state,0,self._n_qubits-1)

        #tail ry part
        state = self.qf_ry(state,thetas[self._n_qubits:2*self._n_qubits],range(0,self._n_qubits))
        return state 


    def measurement(self,state):
        sum_mat = torch.tensor(self.qf_sum(),dtype=torch.float64)
        sum_mat = sum_mat.t()
        state = state * state
        state = torch.mm(sum_mat,state)
        return state

    def state_prob(self,state):
        state = state * state
        return state
    
    def resolve(self,state):
        state = state.double()
        state = torch.pow(state,2)
        mstate = torch.ones(int(math.pow(2,self._n_qubits )),1, dtype=torch.float64)
        sum_mat = torch.tensor(self.qf_sum(),dtype=torch.float64)
        # print("sum_mat:",sum_mat)
        for i in range(sum_mat.shape[0]):
            for j in range(sum_mat.shape[1]):
                if int(sum_mat[i][j]) == 0:
                    val = torch.mm(torch.index_select(mstate,0,torch.tensor([i])),1-torch.index_select(state,0,torch.tensor([j]))).squeeze()
                    mstate = self.set_value(mstate,i,0,torch.index_select(val,0,torch.tensor([0])))
                elif int(sum_mat[i][j]) == 1: 
                    val = torch.mm(torch.index_select(mstate,0,torch.tensor([i])),torch.index_select(state,0,torch.tensor([j]))).squeeze()
                    mstate = self.set_value(mstate,i,0,torch.index_select(val,0,torch.tensor([0])))
        # print("state_item: 1",mstate)
        mstate = torch.sqrt(mstate)
        return mstate

        
    def run(self,state, thetas):
        # print("state_item 0 :",state)
        if state.shape[0]==self._n_qubits:
            state= self.resolve(state)

        # sys.exit(0)
        state =self.vqc_10(state,thetas)
        state = self.state_prob(state)
        # state = self.measurement(state)
        return state 


class VQC_Net(nn.Module):
    def __init__(self,num_qubit,class_num):
        super(VQC_Net, self).__init__()

        #init parameter
        self.num_qubit = num_qubit
        self.theta= Parameter(torch.tensor([np.pi/3,np.pi/4,np.pi/3,np.pi/9,np.pi,np.pi/4,np.pi/10,np.pi/2],dtype=torch.float64,requires_grad=True))#np.random.randn(8)*np.pi
        self.class_num = class_num

        #init  VClassicCircuitMatrix
        self.vcm = VClassicCircuitMatrix(num_qubit)

    def prob2amp(self, input):
        size = input.shape
        shape = list(size)
        shape[-1] = 2**self.num_qubit
        output = torch.zeros(shape,dtype=torch.float64)

        for i in range(size[0]):
            output[i][0] = (1 - input[i][0]) * (1 - input[i][1]) * (1 - input[i][2]) * (1 - input[i][3])
            output[i][1] = (input[i][0]) * (1 - input[i][1]) * (1 - input[i][2]) * (1 - input[i][3])
            output[i][2] = (1 - input[i][0]) * (input[i][1]) * (1 - input[i][2]) * (1 - input[i][3])
            output[i][3] = (input[i][0]) * (input[i][1]) * (1 - input[i][2]) * (1 - input[i][3])
            output[i][4] = (1 - input[i][0]) * (1 - input[i][1]) * (input[i][2]) * (1 - input[i][3])
            output[i][5] = (input[i][0]) * (1 - input[i][1]) * (input[i][2]) * (1 - input[i][3])
            output[i][6] = (1 - input[i][0]) * (input[i][1]) * (input[i][2]) * (1 - input[i][3])
            output[i][7] = (input[i][0]) * (input[i][1]) * (input[i][2]) * (1 - input[i][3])
            output[i][8] = (1 - input[i][0]) * (1 - input[i][1]) * (1 - input[i][2]) * (input[i][3])
            output[i][9] = (input[i][0]) * (1 - input[i][1]) * (1 - input[i][2]) * (input[i][3])
            output[i][10] = (1 - input[i][0]) * (input[i][1]) * (1 - input[i][2]) * (input[i][3])
            output[i][11] = (input[i][0]) * (input[i][1]) * (1 - input[i][2]) * (input[i][3])
            output[i][12] = (1 - input[i][0]) * (1 - input[i][1]) * (input[i][2]) * (input[i][3])
            output[i][13] = (input[i][0]) * (1 - input[i][1]) * (input[i][2]) * (input[i][3])
            output[i][14] = (1 - input[i][0]) * (input[i][1]) * (input[i][2]) * (input[i][3])
            output[i][15] = (input[i][0]) * (input[i][1]) * (input[i][2]) * (input[i][3])

        return output

    def forward(self, x):
        # print("x:",x,self.num_qubit)
        # mstate_temp = torch.index_select(x, 0, torch.tensor([0]))
        # print(x.shape)
        # mstate_temp = self.prob2amp(x)
        # print(mstate_temp.t())

        mstate_temp = torch.index_select(x, 0, torch.tensor([0]))
        mstate_temp = self.prob2amp(mstate_temp)

        # size = mstate_temp.shape
        # shape = list(size)
        # shape.append(1)
        # mstate_temp = mstate_temp.view(shape)
        #
        mstate_temp = self.vcm.run(mstate_temp.t(),self.theta)
        # print(mstate_temp)
        # sys.exit(0)
        mstate_temp = torch.take(mstate_temp, torch.tensor(range(0,self.class_num)))
        mstate = mstate_temp.view(1,-1)

        # variance circuit matrix
        for i in range(1,x.shape[0]):
            mstate_temp = torch.index_select(x, 0, torch.tensor([i]))

            mstate_temp = self.vcm.run(mstate_temp.t(),self.theta)


            mstate_temp = torch.take(mstate_temp, torch.tensor(range(0,self.class_num)))
            mstate_temp = mstate_temp.view(1,-1)
            # print("mstate_temp:",mstate_temp)
            mstate =  torch.cat((mstate,mstate_temp),dim=0)

        return mstate






