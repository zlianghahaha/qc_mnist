# %%

import sys

sys.path.append("../qiskit")
sys.path.append("../pytorch")
from qiskit_library import *
from lib_util import *
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
# INPUTs
import torch
from torch import tensor

def do_slp_via_th(input_ori, w_ori):
    p = input_ori
    d = 4 * p * (1 - p)
    e = (2 * p - 1)
    # e_sq = torch.tensor(1)
    w = w_ori

    sum_of_sq = (d + e.pow(2)).sum(-1)
    sum_of_sq = sum_of_sq.unsqueeze(-1)
    sum_of_sq = sum_of_sq.expand(p.shape[0], w.shape[0])

    diag_p = torch.diag_embed(e)

    p_w = torch.matmul(w, diag_p)
    # print(diag_p)
    # print(w)
    # print(p_w)
    #
    z_p_w = torch.zeros_like(p_w)
    shft_p_w = torch.cat((p_w, z_p_w), -1)

    sum_of_cross = torch.zeros_like(p_w)
    length = p.shape[1]

    for shft in range(1, length):
        sum_of_cross += shft_p_w[:, :, 0:length] * shft_p_w[:, :, shft:length + shft]

    sum_of_cross = sum_of_cross.sum(-1)

    # print(sum_of_sq,sum_of_cross)
    return (sum_of_sq + 2 * sum_of_cross) / (length ** 2)


def simulate_one_step(I, W, test_L1):
    if test_L1:
        W1 = W
        IFM = I.clone().detach()
        input = IFM[0] * 2 - 1

        q_in = qk.QuantumRegister(16, "io")
        q_enc = qk.QuantumRegister(4, "encoder")
        q_out = qk.QuantumRegister(len(W1), "output")
        c = qk.ClassicalRegister(len(W1), "reg")
        aux = qk.QuantumRegister(4, "aux")

        maxIndex = len(input)
        circuit = qk.QuantumCircuit(q_in, q_enc, q_out, aux, c)

        for idx in range(len(W1)):
            SLP_16_encoding(circuit, q_in, q_enc, input, aux)
            SLP_16_Uw(circuit, q_enc, W1[idx], aux)
            circuit.barrier()

            for qbit in q_enc[0:4]:
                circuit.h(qbit)
                circuit.x(qbit)
            ccccx(circuit, q_enc[0], q_enc[1], q_enc[2], q_enc[3], q_out[idx], aux[0], aux[1])
            circuit.barrier()

            # reset_qbits(circuit,q_in)
            # reset_qbits(circuit,q_enc)

        for idx in range(len(W1)):
            circuit.measure(q_out[idx], c[idx])


    else:
        W2 = W
        OFM1_QC = I
        input = OFM1_QC[0] * 2 - 1

        q_in = qk.QuantumRegister(4, "io")
        q_enc = qk.QuantumRegister(2, "encoder")
        q_out = qk.QuantumRegister(len(W2), "output")
        c = qk.ClassicalRegister(len(W2), "reg")
        aux = qk.QuantumRegister(4, "aux")

        maxIndex = len(input)
        circuit = qk.QuantumCircuit(q_in, q_enc, q_out, aux, c)

        for idx in range(len(W2)):
            SLP_4_encoding(circuit, q_in, q_enc, input, aux)

            SLP_4_Uw(circuit, q_enc, W2[idx], aux)
            circuit.barrier()

            for qbit in q_enc[0:2]:
                circuit.h(qbit)
                circuit.x(qbit)
            circuit.ccx(q_enc[0], q_enc[1], q_out[idx])
            circuit.barrier()

            # reset_qbits(circuit,q_in)
            # reset_qbits(circuit,q_enc)

        for idx in range(len(W2)):
            circuit.measure(q_out[idx], c[idx])

    # print(circuit)

    # %%

    qc_shots = 1000
    num_c_reg = 1

    # print("=" * 50)
    # print("Start simulation:")
    start = time.time()
    iters = 1
    counts = simulate(circuit, qc_shots, iters, False)

    end = time.time()
    qc_time = end - start

    # print("From QC:", counts)
    # print("Simulation elasped time:", qc_time)

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

    # for k,v in counts[0].items():
    #     print(k,v)
    (mycount, bits) = analyze(counts[0])

    # for b in range(bits):
    #     print(b, float(mycount[b]) / qc_shots)
    return float(mycount[0]) / qc_shots



def run_simulator(model,IFM):
    OFM = {}
    input_data = IFM.view(1,-1, )
    layer_idx = 0
    for layer_name, layer in model.named_modules():
        if isinstance(layer, BinaryLinear):
            W = (binarize(model.state_dict()[layer_name+".weight"]))
            OFM_QC = torch.zeros((1,layer.out_features))

            idx = 0
            for w in W:
                w = w.unsqueeze(0)
                # print("\t\tInputs:",input_data)
                # print("\t\tWeights:", w)
                OFM_QC[0][idx] = simulate_one_step(input_data, w, layer_idx==0)
                # print("\t\tResults:",OFM_QC[0][idx])
                idx += 1



            if layer.out_features == 1:
                OFM_QC = torch.cat((OFM_QC, 1 - OFM_QC), -1)

            # print("="*100)
            # print(layer,OFM_QC)

            OFM[layer_name] = OFM_QC
            input_data = OFM_QC
            layer_idx += 1

    # print(OFM["fc2"])
    return OFM["fc2"]
    #
    # idx = 0
    # for w in W1:
    #     w = w.unsqueeze(0)
    #     OFM1_QC[0][idx] = simulate_one_step(IFM, w, True)
    #     idx += 1
    #
    # idx = 0
    # for w in W2:
    #     w = w.unsqueeze(0)
    #     OFM2_QC[0][idx] = simulate_one_step(OFM1_QC, w, False)
    #     idx += 1


if __name__ == "__main__":


    res = tensor([0])

    IFM = tensor([[0.0235, 0.2314, 0.1490, 0.0000, 0.0078, 0.2549, 0.3176, 0.0157, 0.0235,
             0.1686, 0.3412, 0.1216, 0.0549, 0.2431, 0.2157, 0.0353]])
    W1 = tensor([[-1., 1., -1., 1., 1., -1., 1., -1., 1., -1., -1., -1., -1., 1.,
             1., -1.],
            [-1., -1., -1., -1., -1., 1., -1., -1., -1., 1., 1., -1., -1., -1.,
             -1., -1.],
            [1., 1., 1., 1., 1., 1., -1., 1., 1., 1., 1., 1., 1., -1.,
             1., 1.],
            [1., 1., -1., -1., 1., 1., -1., -1., 1., 1., -1., -1., -1., 1.,
             -1., 1.]])

    OFM1 = tensor([[0.0384, 0.3188, 0.4043, 0.0272]])
    W2 = tensor([[1., 1., -1., 1.],
            [-1., 1., -1., -1.]])

    OFM2 = tensor([[0.3904, 0.3105]])


    OFM1_QC = torch.zeros_like(OFM1)
    OFM2_QC = torch.zeros_like(OFM2)

    idx = 0
    for w in W1:
        w = w.unsqueeze(0)

        print("\t\tInputs:", IFM)
        print("\t\tWeights:", w)
        OFM1_QC[0][idx] = simulate_one_step(IFM, w, True)
        print("\t\tResults:", OFM1_QC[0][idx])
        idx += 1

    idx = 0
    for w in W2:
        w = w.unsqueeze(0)
        OFM2_QC[0][idx] = simulate_one_step(OFM1_QC, w, False)
        idx += 1

    print(OFM1_QC)
    print(OFM2_QC)
