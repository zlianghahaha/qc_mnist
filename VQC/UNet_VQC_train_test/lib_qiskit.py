

import torch
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
import sys
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import functools
from qiskit.tools.monitor import job_monitor
from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister
from qiskit.extensions import  UnitaryGate
from qiskit import Aer, execute,IBMQ
import qiskit
import math

################ Weiwen on 12-30-2020 ################
# Function: fire_ibmq from Listing 6
# Note: used for execute quantum circuit using 
#       simulation or ibm quantum processor
# Parameters: (1) quantum circuit; 
#             (2) number of shots;
#             (3) simulation or quantum processor;
#             (4) backend name if quantum processor.
######################################################
def fire_ibmq(circuit,shots,Simulation = True,backend_name='ibmq_essex'):     
    if not Simulation:
        provider = IBMQ.get_provider('ibm-q-academic')
        backend = provider.get_backend(backend_name)
    else:
        backend = Aer.get_backend('qasm_simulator')
    job_ibm_q = execute(circuit, backend, shots=shots)

    # job_monitor(job_ibm_q)
    result_ibm_q = job_ibm_q.result()

    counts = result_ibm_q.get_counts()
    return counts


################ Weiwen on 12-30-2020 ################
# Function: analyze from Listing 6
# Note: used for analyze the count on states to  
#       formulate the probability for each qubit
# Parameters: (1) counts returned by fire_ibmq; 
######################################################
def analyze(counts):
    mycount = {}
    for i in range(2):
        mycount[i] = 0
    for k,v in counts.items():
        bits = len(k) 
        for i in range(bits):            
            if k[bits-1-i] == "1":
                if i in mycount.keys():
                    mycount[i] += v
                else:
                    mycount[i] = v
    return mycount,bits

class ExtendGate():
    
    ################ Weiwen on 06-02-2021 ################
    # Function: ccz from Listing 3
    # Note: using the basic Toffoli gates and CZ gate
    #       to implement ccz gate, which will flip the
    #       sign of state |111>
    # Parameters: (1) quantum circuit; 
    #             (2-3) control qubits;
    #             (4) target qubits;
    #             (5) auxiliary qubits.
    ######################################################
    @classmethod
    def ccz(cls,circ, q1, q2, q3, aux1):
        # Apply Z-gate to a state controlled by 3 qubits
        circ.ccx(q1, q2, aux1)
        circ.cz(aux1, q3)
        # cleaning the aux bit
        circ.ccx(q1, q2, aux1)
        return circ

    @classmethod
    def cccx(cls,circ, q1, q2, q3, q4, aux1, aux2):
        # Apply Z-gate to a state controlled by 3 qubits
        circ.ccx(q1, q2, aux1)
        circ.ccx(q3, aux1, aux2)
        circ.cx(aux2, q4)
        # cleaning the aux bits
        circ.ccx(q3, aux1, aux2)
        circ.ccx(q1, q2, aux1)
        return circ

    ################ Weiwen on 12-30-2020 ################
    # Function: cccz from Listing 3
    # Note: using the basic Toffoli gates and CZ gate
    #       to implement cccz gate, which will flip the
    #       sign of state |1111>
    # Parameters: (1) quantum circuit; 
    #             (2-4) control qubits;
    #             (5) target qubits;
    #             (6-7) auxiliary qubits.
    ######################################################
    @classmethod
    def cccz(cls,circ, q1, q2, q3, q4, aux1, aux2):
        # Apply Z-gate to a state controlled by 4 qubits
        circ.ccx(q1, q2, aux1)
        circ.ccx(q3, aux1, aux2)
        circ.cz(aux2, q4)
        # cleaning the aux bits
        circ.ccx(q3, aux1, aux2)
        circ.ccx(q1, q2, aux1)
        return circ

    ################ Weiwen on 12-30-2020 ################
    # Function: cccz from Listing 4
    # Note: using the basic Toffoli gate to implement ccccx
    #       gate. It is used to switch the quantum states
    #       of |11110> and |11111>.
    # Parameters: (1) quantum circuit; 
    #             (2-5) control qubits;
    #             (6) target qubits;
    #             (7-8) auxiliary qubits.
    ######################################################
    @classmethod
    def ccccx(cls,circ, q1, q2, q3, q4, q5, aux1, aux2):
        circ.ccx(q1, q2, aux1)
        circ.ccx(q3, q4, aux2)
        circ.ccx(aux2, aux1, q5)
        # cleaning the aux bits
        circ.ccx(q3, q4, aux2)
        circ.ccx(q1, q2, aux1)
        return circ

    ################ Weiwen on 12-30-2020 ################
    # Function: neg_weight_gate from Listing 3
    # Note: adding NOT(X) gate before the qubits associated
    #       with 0 state. For example, if we want to flip 
    #       the sign of |1101>, we add X gate for q2 before
    #       the cccz gate, as follows.
    #       --q3-----|---
    #       --q2----X|X--
    #       --q1-----|---
    #       --q0-----z---
    # Parameters: (1) quantum circuit; 
    #             (2) all qubits, say q0-q3;
    #             (3) the auxiliary qubits used for cccz
    #             (4) states, say 1101
    ######################################################
    @classmethod
    def neg_weight_gate(cls,circ,qubits,aux,state):
        idx = 0
        # The index of qubits are reversed in terms of states.
        # As shown in the above example: we put X at q2 not the third position.
        print("state 1",state)
        state = state[::-1]
        print("state 2",state)
        for idx in range(len(state)):
            if state[idx]=='0':
                circ.x(qubits[idx])
        cls.cccz(circ,qubits[0],qubits[1],qubits[2],qubits[3],aux[0],aux[1])
        for idx in range(len(state)):
            if state[idx]=='0':
                circ.x(qubits[idx])

class ULayerCircuit():
################ Weiwen on 06-02-2021 ################
# QuantumFlow Weight Generation for U-Layer
######################################################
    def __init__(self,input_num,output_num):
        self.n_qubit = int (math.log2(input_num))
        self.n_class = output_num
        if self.n_qubit > 4:
            print('UNetCircuit: The input size is too big. Qubits should be less than 4.')
            sys.exit(0)
        print("UNetCircuit: n_qubit =",self.n_qubit,",n_class =",self.n_class)
        
    
    def add_aux(self,circuit):
        if self.n_qubit == 3:
            aux = QuantumRegister(1,"aux_qbit")
        elif self.n_qubit == 4:
            aux = QuantumRegister(2,"aux_qbit")
        circuit.add_register(aux)
        return aux

    def add_in_qubits(self,circuit):
        inps = []
        for i in range(self.n_class):
            inp = QuantumRegister(self.n_qubit,"in"+str(i)+"_qbit")
            circuit.add_register(inp)
            inps.append(inp)
        return inps
    
    def add_out_qubits(self,circuit):
        out_qubits = QuantumRegister(self.n_class,"u_layer_qbits")
        circuit.add_register(out_qubits)
        return out_qubits

    def forward(self,circuit,data_matrix,weight,aux,in_qubits,out_qubit):
        for i in range(self.n_class):
            n_q_gates,n_idx = self.qf_map_extract_from_weight(weight[i])
            circuit.append(UnitaryGate(data_matrix[n_idx], label="Input"), in_qubits[i][0:self.n_qubit])
            qbits = in_qubits[i]
            for gate in n_q_gates:
                z_count = gate.count("1")
                # z_pos = get_index_list(gate,"1")
                z_pos = self.get_index_list(gate[::-1],"1")
                if z_count==1:
                    circuit.z(qbits[z_pos[0]])
                elif z_count==2:
                    circuit.cz(qbits[z_pos[0]],qbits[z_pos[1]])
                elif z_count==3:
                    ExtendGate.ccz(circuit,qbits[z_pos[0]],qbits[z_pos[1]],qbits[z_pos[2]],aux[0])
                elif z_count==4:
                    ExtendGate.cccz(circuit,qbits[z_pos[0]],qbits[z_pos[1]],qbits[z_pos[2]],qbits[z_pos[3]],aux[0],aux[1])
        circuit.barrier()
        for i in range(self.n_class):
            circuit.h(in_qubits[i])
            circuit.x(in_qubits[i])
        circuit.barrier()
        for i in range(self.n_class):
            qbits = in_qubits[i]
            if self.n_qubit==1:
                circuit.cx(qbits[0],qbits[1],out_qubit[i])
            elif self.n_qubit==2:
                circuit.ccx(qbits[0],qbits[1],out_qubit[i])
            elif self.n_qubit==3:
                ExtendGate.cccx(circuit,qbits[0],qbits[1],qbits[2],out_qubit[i],aux[0])
            elif self.n_qubit==4:
                ExtendGate.ccccx(circuit,qbits[0],qbits[1],qbits[2],qbits[3],out_qubit[i],aux[0],aux[1])


    @classmethod
    def get_index_list(self,input,target):
        index_list = []
        try:
            beg_pos = 0
            while True:
                find_pos = input.index(target,beg_pos)
                index_list.append(find_pos)
                beg_pos = find_pos+1
        except Exception as exception:        
            pass    
        return index_list
    @classmethod   
    def change_sign(self,sign,bin):
        affect_num = [bin]
        one_positions = []
        try:
            beg_pos = 0
            while True:
                find_pos = bin.index("1",beg_pos)
                one_positions.append(find_pos)
                beg_pos = find_pos+1
        except Exception as exception:
            # print("Not Found")
            pass
        for k,v in sign.items():
            change = True
            for pos in one_positions:
                if k[pos]=="0":                
                    change = False
                    break
            if change:
                sign[k] = -1*v
    
    @classmethod
    def find_start(self,affect_count_table,target_num):
        for k in list(affect_count_table.keys())[::-1]:
            if target_num<=k:
                return k

    @classmethod
    def recursive_change(self,direction,start_point,target_num,sign,affect_count_table,quantum_gates):
        
        if start_point == target_num:
            # print("recursive_change: STOP")
            return
        
        gap = int(math.fabs(start_point-target_num))    
        step = self.find_start(affect_count_table,gap)
        self.change_sign(sign,affect_count_table[step])
        quantum_gates.append(affect_count_table[step])
        
        if direction=="r": 
            # print("recursive_change: From",start_point,"Right(-):",step)
            start_point = start_point - step
            direction = "l"
            self.recursive_change(direction,start_point,target_num,sign,affect_count_table,quantum_gates)
            
        else:        
            # print("recursive_change: From",start_point,"Left(+):",step)
            start_point = start_point + step
            direction = "r"
            self.recursive_change(direction,start_point,target_num,sign,affect_count_table,quantum_gates)
        
    
    @classmethod
    def guarntee_upper_bound_algorithm(self,sign,target_num,total_len,digits):        
        flag = "0"+str(digits)+"b"
        pre_num = 0
        affect_count_table = {}
        quantum_gates = []
        for i in range(digits):
            cur_num = pre_num + pow(2,i)
            pre_num = cur_num
            binstr_cur_num = format(cur_num,flag) 
            affect_count_table[int(pow(2,binstr_cur_num.count("0")))] = binstr_cur_num   
        
        if target_num in affect_count_table.keys():
            quantum_gates.append(affect_count_table[target_num])
            self.change_sign(sign,affect_count_table[target_num])  
      
        else:
            direction = "r"
            start_point = self.find_start(affect_count_table,target_num)
            quantum_gates.append(affect_count_table[start_point])
            self.change_sign(sign,affect_count_table[start_point])
            self.recursive_change(direction,start_point,target_num,sign,affect_count_table,quantum_gates)
        
        return quantum_gates

    @classmethod
    def qf_map_extract_from_weight(self,weights):    
        # Find Z control gates according to weights
        w = (weights.detach().cpu().numpy())
        total_len = len(w)
        target_num = np.count_nonzero(w == -1)
        if target_num > total_len/2:
            w = w*-1
        target_num = np.count_nonzero(w == -1)    
        digits = int(math.log(total_len,2))
        flag = "0"+str(digits)+"b"
        max_num = int(math.pow(2,digits))
        sign = {}
        for i in range(max_num):        
            sign[format(i,flag)] = +1

        quantum_gates = self.guarntee_upper_bound_algorithm(sign,target_num,total_len,digits)
        
        # Build the mapping from weight to final negative num 
        fin_sign = list(sign.values())
        fin_weig = [int(x) for x in list(w)]
        sign_neg_index = []    
        try:
            beg_pos = 0
            while True:
                find_pos = fin_sign.index(-1,beg_pos)            
                # qiskit_position = int(format(find_pos,flag)[::-1],2)                            
                sign_neg_index.append(find_pos)
                beg_pos = find_pos+1
        except Exception as exception:        
            pass  
    

        weight_neg_index = []
        try:
            beg_pos = 0
            while True:
                find_pos = fin_weig.index(-1,beg_pos)
                weight_neg_index.append(find_pos)
                beg_pos = find_pos+1
        except Exception as exception:        
            pass    
    
        map = {}
        for i in range(len(sign_neg_index)):
            map[sign_neg_index[i]] = weight_neg_index[i]
    
        ret_index = list([-1 for i in range(len(fin_weig))])
        
        
        for k,v in map.items():
            ret_index[k]=v
        
        
        for i in range(len(fin_weig)):
            if ret_index[i]!=-1:
                continue
            for j in range(len(fin_weig)):
                if j not in ret_index:
                    ret_index[i]=j
                    break  
        return quantum_gates,ret_index
    

# just for temp 
class PLayerCircuit():
    def __init__(self,input_num,output_num):
        self.input_num = input_num
        self.output_num = output_num
        if self.input_num != 2 or self.output_num != 2:
            print('PLayerCircuit: The input size or output size is not 2. Now thet p-layer only support 2 inputs and 2 outputs!')
            sys.exit(0)
        print("PLayerCircuit: input_num =",self.input_num,",output_num =",self.output_num)

    def add_out_qubits(self,circuit):
        out_qubits = QuantumRegister(self.output_num,"p_layer_qbits")
        circuit.add_register(out_qubits)
        return out_qubits
    
    def forward(self,circuit,weight,in_qubits,out_qubits):
        for i in range(self.output_num):
            #mul weight
            if weight[i].sum()<0:
                weight[i] = weight[i]*-1
            idx = 0
            for idx in range(weight[i].flatten().size()[0]):
                if weight[i][idx]==-1:
                    circuit.x(in_qubits[idx])
            #sum and pow2
        
            circuit.h(out_qubits[i])
            circuit.cz(in_qubits[0],out_qubits[i])
            circuit.x(out_qubits[i])
            circuit.cz(in_qubits[1],out_qubits[i])
            circuit.x(out_qubits[i])
            circuit.h(out_qubits[i])
            circuit.x(out_qubits[i])
            #recover
            for idx in range(weight[i].flatten().size()[0]):
                if weight[i][idx]==-1:
                    circuit.x(in_qubits[idx])
            circuit.barrier(in_qubits,out_qubits)

class NormerlizeCircuit():
    def __init__(self,n_qubit):
        self.n_qubit = n_qubit
        print("NormerlizeCircuit: n_qubit =",self.n_qubit)

    def add_norm_qubits(self,circuit):
        norm_qubits = QuantumRegister(self.n_qubit,"norm_qbits")
        circuit.add_register(norm_qubits)
        return norm_qubits

    def add_out_qubits(self,circuit):
        out_qubits = QuantumRegister(self.n_qubit,"norm_output_qbits")
        circuit.add_register(out_qubits)
        return out_qubits
    
    def forward(self,circuit,input_qubits,norm_qubits,out_qubits,norm_flag,norm_para):
        for i in range(self.n_qubit):
            norm_init_rad = float(norm_para[i].sqrt().arcsin()*2)
            circuit.ry(norm_init_rad,norm_qubits[i])
            if norm_flag[i]:
                circuit.cx(input_qubits[i],out_qubits[i])
                circuit.x(input_qubits[i])
                circuit.ccx(input_qubits[i],norm_qubits[i],out_qubits[i])
            else:
                circuit.ccx(input_qubits[i],norm_qubits[i],out_qubits[i])

