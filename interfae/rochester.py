# %%

import sys

sys.path.append("../qiskit")
sys.path.append("../pytorch")
from qiskit_library import *
# from mnist import *
import numpy as np
from random import randrange
import qiskit as qk
from qiskit import Aer
from qiskit import execute
import math
import sys
import random
import time
from tqdm import tqdm
import torch
import numpy as np
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
from lib_model_summary import summary
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np
import math
import os
import shutil
import os
import time
import sys
from pathlib import Path
import functools
from QC_SELF import Net
from collections import Counter

from qiskit import IBMQ

# IBMQ.delete_accounts()
IBMQ.save_account(
    '62d0e14364f490e45b5b5e0f6eebdbc083270ffffb660c7054219b15c7ce99ab4aa3b321309c0a9d0c3fc20086baece1376297dcdb67c7b715f9de1e4fa79efb')
IBMQ.load_account()


def analyze(counts):
    mycount = {}
    for i in range(num_c_reg):
        mycount[i] = 0
    for k, v in counts.items():
        bits = len(k)
        for i in range(bits):
            if k[bits - 1 - i] == "1":
                if i in mycount.keys():
                    mycount[i] += v
                else:
                    mycount[i] = v
    return mycount, bits


def fire_ibmq(circuit, shots, iter, Simulation=False, printable=True, backend_name='ibmq_essex'):
    if printable:
        print(circuit)

    count_set = []
    start = time.time()
    for it in range(iter):
        if not Simulation:
            provider = IBMQ.get_provider('ibm-q-academic')
            # ibm-q-academic backends:
            #  5 qubits: ibmq_valencia
            # 20 qubits: ibmq_poughkeepsie, ibmq_johannesburg,ibmq_boeblingen, ibmq_20_tokyo
            # 53 qubits: ibmq_rochester

            # To get a specific qubit backend:
            backend = provider.get_backend(backend_name)
        else:
            backend = Aer.get_backend('qasm_simulator')
        job_ibm_q = execute(circuit, backend, shots=shots)
        # job_monitor(job_ibm_q)
        result_ibm_q = job_ibm_q.result()
        counts = result_ibm_q.get_counts()
        count_set.append(counts)
    end = time.time()
    print("Simulation time:", end - start)

    return count_set


# %%

# INPUTs
import torch
from torch import tensor

milestones = [6, 10, 14]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net().to(device)

optimizer = torch.optim.Adam([
    {'params': model.fc1.parameters()},
    {'params': model.fc2.parameters()},
    {'params': model.qc1.parameters(), 'lr': 1},
    {'params': model.qc2.parameters(), 'lr': 1},
], lr=1)

scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [0, 1], gamma=0.1)

resume_path = "../pytorch/model/ipykernel_launcher.py_2020_05_04-14_22_13/checkpoint_0_0.9714.pth.tar"
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

print("====")
print(model(torch.tensor([0.95, 0.95]), False))
print("====")

# %%
'''
q = qk.QuantumRegister(12,"q")
c = qk.ClassicalRegister(1,"c")

circuit = qk.QuantumCircuit(q, c)


circuit.ry(0.7954,q[0])
circuit.ry(2.4981,q[1])
circuit.ry(2.3766,q[3])
circuit.ry(1.7236,q[5])
circuit.ry(0.7954,q[6])
circuit.ry(2.4981,q[7])
circuit.ry(0.8371,q[9])
circuit.ry(2.5933,q[10])

circuit.x(q[3])
circuit.x(q[10])
circuit.barrier(q[3],q[11],q[10],q[9],q[8],q[7],q[6],q[5],q[4],q[1],q[0],q[2])
circuit.h(q[1])
circuit.h(q[7])
circuit.cz(q[0],q[1])
circuit.cz(q[6],q[7])
circuit.z(q[1])
circuit.h(q[7])
circuit.h(q[1])
circuit.x(q[7])
circuit.x(q[1])
circuit.cx(q[7],q[8])
circuit.cx(q[1],q[2])
circuit.ccx(q[7],q[9],q[8])
circuit.x(q[1])
circuit.ccx(q[1],q[3],q[2])
circuit.cx(q[2],q[4])
circuit.ccx(q[2],q[5],q[4])
circuit.barrier(q[3],q[11],q[10],q[9],q[8],q[7],q[6],q[5],q[4],q[1],q[0],q[2])
circuit.h(q[8])
circuit.cz(q[4],q[8])
circuit.h(q[8])
circuit.x(q[8])
circuit.cx(q[8],q[11])
circuit.x(q[8])
circuit.ccx(q[8],q[10],q[11])
circuit.barrier(q[3],q[11],q[10],q[9],q[8],q[7],q[6],q[5],q[4],q[1],q[0],q[2])
circuit.measure(q[11],c[0])



print(circuit)
'''

# %%
'''
q = qk.QuantumRegister(15,"q")
c = qk.ClassicalRegister(2,"c")

circuit = qk.QuantumCircuit(q, c)


circuit.ry(0.4510,q[0])
circuit.ry(0.4510,q[1])
circuit.ry(1.7412,q[3])
circuit.ry(0.4510,q[4])
circuit.ry(0.4510,q[5])
circuit.ry(2.3766,q[7])
circuit.ry(1.5956,q[9])

circuit.h(q[1])
circuit.h(q[5])
circuit.x(q[7])
circuit.cz(q[0],q[1])
circuit.cz(q[4],q[5])
circuit.z(q[1])
circuit.z(q[5])
circuit.h(q[1])
circuit.h(q[5])
circuit.x(q[1])
circuit.x(q[5])
circuit.cx(q[1],q[2])
circuit.cx(q[5],q[6])
circuit.ccx(q[1],q[3],q[2])
circuit.x(q[5])
circuit.ccx(q[5],q[7],q[6])
circuit.cx(q[6],q[8])
circuit.ccx(q[6],q[9],q[8])
circuit.barrier(q[7],q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[8],q[9],q[10],q[11],q[12],q[13],q[14])
circuit.cx(q[2],q[8])
circuit.h(q[10])
circuit.h(q[11])
circuit.ry(2.2800,q[13])
circuit.ccx(q[1],q[3],q[2])
circuit.cz(q[8],q[10])
circuit.x(q[13])
circuit.cx(q[1],q[2])
circuit.cz(q[8],q[11])
circuit.ry(0.8632,q[2])
circuit.z(q[10])
circuit.h(q[11])
circuit.h(q[10])
circuit.x(q[11])
circuit.x(q[10])
circuit.cx(q[10],q[12])
circuit.x(q[10])
circuit.ccx(q[10],q[13],q[12])
circuit.cx(q[11],q[14])
circuit.ccx(q[11],q[2],q[14])
circuit.barrier(q[7],q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[8],q[9],q[10],q[11],q[12],q[13],q[14])
circuit.measure(q[12],c[0])
circuit.measure(q[14],c[1])

print(circuit)

'''
# %%

'''

qc_shots = 10000
num_c_reg = 2

print("="*50)
print("Start simulation:")
start = time.time()        
iters = 1
counts = fire_ibmq(circuit,qc_shots,iters,True,False)
end = time.time()
qc_time = end - start

(mycount,bits) = analyze(counts[0])
for b in range(bits):
    print (b,float(mycount[b])/qc_shots)

print("From QC:",counts)
print("Simulation elasped time:",qc_time)

'''
# %%
'''
qc_shots = 8192
print("="*50)
print("Start run:")
start = time.time()        
iters = 1
counts = fire_ibmq(circuit,qc_shots,iters,False,False,backend_name="ibmq_rochester")
end = time.time()
qc_time = end - start

(mycount,bits) = analyze(counts[0])
for b in range(bits):
    print (b,float(mycount[b])/qc_shots)

print("From QC:",counts)
print("Simulation elasped time:",qc_time)

'''


# %%


def generate_circuit(model, input, num_layer, arch, opt=0):
    print("=" * 100)
    print("Generating Quantum Circuit from Machine Learning Model")
    print("=" * 100)

    in_size = input.shape[0]
    # print(in_size)

    # for name,para in model.named_parameters():
    #     print(name,torch.tensor(para))
    #
    #

    c = qk.ClassicalRegister(2, "reg")
    circuit = qk.QuantumCircuit(c)

    q_in_whole = {}
    q_aux_whole = {}
    q_out = {}

    for out in range(arch[1]):
        # for i in range(num_layer):
        i = 0
        for sub_qc in range(arch[i]):
            q_in = qk.QuantumRegister(in_size, "io" + "-" + str(out) + "-" + str(i) + "-" + str(sub_qc))
            circuit.add_register(q_in)

            q_in_whole[out, i, sub_qc] = q_in
            w = model.state_dict()["fc" + str(i + 1) + ".weight"][sub_qc]
            w = (w > 0).int()

            if model.state_dict()["qc" + str(i + 1) + "a.x_l_0_5"][sub_qc] == 0:
                q_aux = qk.QuantumRegister(2, "aux" + "-" + str(out) + "-" + str(i) + "-" + str(sub_qc))
                circuit.add_register(q_aux)
                circuit.barrier()
                q_aux_whole[out, i, sub_qc] = q_aux
                # print("qc"+str(i)+str(sub_qc))
                # qcircuit["qc"+str(i)+str(sub_qc)] =
                # circuit = qcircuit["qc"+str(i)+str(sub_qc)]
                if i == 0:
                    init(circuit, input, q_in)
                circuit.h(q_in[1])
                circuit.cz(q_in[0], q_in[1])
                if w[0] != w[1]:
                    circuit.z(q_in[1])
                circuit.h(q_in[1])
                circuit.x(q_in[1])
                circuit.cx(q_in[1], q_aux[0])
                mul = model.state_dict()["qc" + str(i + 1) + "a.x_running_rot"][sub_qc]
                if i != num_layer - 1:
                    mul = mul * model.state_dict()["qc" + str(i + 1) + ".x_running_rot"][sub_qc]
                init(circuit, mul.view(1) * 2 - 1, q_aux[1])
                circuit.ccx(q_in[1], q_aux[1], q_aux[0])
                q_out[out, i, sub_qc] = q_aux[0]
                # print(qcircuit["qc"+str(i)+str(sub_qc)])
            elif model.state_dict()["qc" + str(i + 1) + "a.x_l_0_5"][sub_qc] == 1 and i != num_layer - 1:
                q_aux = qk.QuantumRegister(4, "aux" + "-" + str(out) + "-" + str(i) + "-" + str(sub_qc))
                circuit.add_register(q_aux)
                circuit.barrier()
                q_aux_whole[out, i, sub_qc] = q_aux
                # print("qc"+str(i)+str(sub_qc))
                # qcircuit["qc"+str(i)+str(sub_qc)] = qk.QuantumCircuit(q_in, q_aux)
                # circuit = qcircuit["qc"+str(i)+str(sub_qc)]
                if i == 0:
                    init(circuit, input, q_in)
                circuit.h(q_in[1])
                circuit.cz(q_in[0], q_in[1])
                if w[0] != w[1]:
                    circuit.z(q_in[1])
                circuit.h(q_in[1])
                circuit.x(q_in[1])
                circuit.cx(q_in[1], q_aux[0])

                mul = model.state_dict()["qc" + str(i + 1) + "a.x_running_rot"][sub_qc]
                init(circuit, mul.view(1) * 2 - 1, q_aux[1])
                circuit.x(q_in[1])
                circuit.x(q_aux[1])
                circuit.ccx(q_in[1], q_aux[1], q_aux[0])

                circuit.cx(q_aux[0], q_aux[2])

                mul = model.state_dict()["qc" + str(i + 1) + ".x_running_rot"][sub_qc]
                init(circuit, mul.view(1) * 2 - 1, q_aux[3])
                circuit.ccx(q_aux[0], q_aux[3], q_aux[2])
                q_out[out, i, sub_qc] = q_aux[2]
                # print(qcircuit["qc"+str(i)+str(sub_qc)])

            else:
                q_aux = qk.QuantumRegister(2, "aux" + "-" + str(out) + "-" + str(i) + "-" + str(sub_qc))
                circuit.add_register(q_aux)
                circuit.barrier()
                q_aux_whole[out, i, sub_qc] = q_aux
                # print("qc"+str(i)+str(sub_qc))
                # qcircuit["qc"+str(i)+str(sub_qc)] = qk.QuantumCircuit(q_in, q_aux)
                # circuit = qcircuit["qc"+str(i)+str(sub_qc)]
                if i == 0:
                    init(circuit, input, q_in)
                circuit.h(q_in[1])
                circuit.cz(q_in[0], q_in[1])
                if w[0] != w[1]:
                    circuit.z(q_in[1])
                circuit.h(q_in[1])
                circuit.x(q_in[1])
                circuit.cx(q_in[1], q_aux[0])
                mul = model.state_dict()["qc" + str(i + 1) + "a.x_running_rot"][sub_qc]
                init(circuit, mul.view(1) * 2 - 1, q_aux[1])
                circuit.x(q_in[1])
                circuit.x(q_aux[1])
                circuit.ccx(q_in[1], q_aux[1], q_aux[0])
                # print(qcircuit["qc"+str(i)+str(sub_qc)])
                q_out[out, i, sub_qc] = q_aux[0]

        circuit.cx(q_out[out, i, 0], q_out[out, i, 1])

        i = 1
        q_in = qk.QuantumRegister(1, "io" + "-" + str(out) + "-" + str(i))
        circuit.add_register(q_in)
        q_aux = qk.QuantumRegister(2, "aux" + "-" + str(out) + "-" + str(i))
        circuit.add_register(q_aux)
        circuit.barrier()
        q_in_whole[out, i] = q_in
        w = model.state_dict()["fc" + str(i + 1) + ".weight"][out]
        w = (w > 0).int()

        circuit.h(q_in)
        circuit.cz(q_out[out, 0, 1], q_in)
        if w[0] != w[1]:
            circuit.z(q_in)
        circuit.h(q_in)
        circuit.x(q_in)
        circuit.cx(q_in, q_aux[0])

        if model.state_dict()["qc" + str(i + 1) + "a.x_l_0_5"][out] == 0:
            mul = model.state_dict()["qc" + str(i + 1) + "a.x_running_rot"][out]
            init(circuit, mul.view(1) * 2 - 1, q_aux[1])
            circuit.ccx(q_in, q_aux[1], q_aux[0])
            q_out[out, i] = q_aux[0]
        else:
            mul = model.state_dict()["qc" + str(i + 1) + "a.x_running_rot"][out]
            init(circuit, mul.view(1) * 2 - 1, q_aux[1])
            circuit.x(q_in)
            circuit.x(q_aux[1])
            circuit.ccx(q_in, q_aux[1], q_aux[0])
            q_out[out, i] = q_aux[0]

    circuit.measure(q_out[0, 1], c[0])
    circuit.measure(q_out[1, 1], c[1])

    return circuit


import random

x_set = [x / 20.0 for x in range(1, 20)]

results = []
fh = open("ibmq_rochester.res", "w")
for i in range(100):

    x = random.choice(x_set)
    y = random.choice(x_set)

    circuit = generate_circuit(model, torch.tensor([x, y]) * 2 - 1, 2, [2, 2])

    qc_shots = 10000
    num_c_reg = 2

    print("***************Start:", [x, y], "*****************")
    #
    # print("="*50)
    # print("Start simulation:")
    # start = time.time()
    # iters = 1
    # counts = fire_ibmq(circuit,qc_shots,iters,True,False)
    # end = time.time()
    # qc_time = end - start
    #
    # (mycount,bits) = analyze(counts[0])
    #
    # res = {}
    # for b in range(bits):
    #     print (b,float(mycount[b])/qc_shots)
    #     res[b] = float(mycount[b])/qc_shots
    #
    #
    # print("From QC:",counts)
    # print("Simulation elasped time:",qc_time)

    qc_shots = 8192
    print("=" * 50)
    print("Start run:")
    start = time.time()
    iters = 1
    counts = fire_ibmq(circuit, qc_shots, iters, False, False, backend_name="ibmq_rochester")
    end = time.time()
    qc_time = end - start

    qc_res = {}
    (mycount, bits) = analyze(counts[0])
    for b in range(bits):
        print(b, float(mycount[b]) / qc_shots)
        qc_res[b] = float(mycount[b]) / qc_shots

    results.append([(x, y), qc_res])

    print("From QC:", counts)
    print("Simulation elasped time:", qc_time)

    print([(x, y), qc_res], file=fh, flush=True)

    if i % 5 == 0:
        print("=" * 10, "PRINT RESULTS", "=" * 10)
        print("-" * 40)
        for k in results:
            print(k)
        print("-" * 40)

#
# print(circuit)


