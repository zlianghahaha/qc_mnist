# %%

from qiskit_library import *
import torch
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

from qiskit import IBMQ
from qiskit.providers.aer.noise import NoiseModel
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


def fire_ibmq(circuit, shots, iter, Simulation=False, printable=True, backend_name='ibmq_essex', mapping={}):
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
            provider = IBMQ.get_provider('ibm-q-academic')
            backend = provider.get_backend('ibmq_burlington')
            coupling_map = backend.configuration().coupling_map

            # Generate an Aer noise model for device
            noise_model = NoiseModel.from_backend(backend)
            basis_gates = noise_model.basis_gates
            backend = Aer.get_backend('qasm_simulator')

        if Simulation:
            job_ibm_q = execute(circuit, backend,
                          coupling_map=coupling_map,
                          noise_model=noise_model,
                          basis_gates=basis_gates,
                          initial_layout=mapping,
                          shots=shots)
        elif len(mapping.keys()) != 0:
            job_ibm_q = execute(circuit, backend, shots=shots, initial_layout=mapping)
        else:
            job_ibm_q = execute(circuit, backend, shots=shots)

        job_monitor(job_ibm_q)
        result_ibm_q = job_ibm_q.result()
        counts = result_ibm_q.get_counts()
        count_set.append(counts)
    end = time.time()
    print("Simulation time:", end - start)

    return count_set


def do_slp_via_th(input_ori, w_ori):
    p = input_ori
    d = 4 * p * (1 - p)
    e = (2 * p - 1)
    # e_sq = torch.tensor(1)
    w = w_ori

    sum_of_sq = (d + e.pow(2)).sum(-1)
    sum_of_sq = sum_of_sq.unsqueeze(-1)
    sum_of_sq = sum_of_sq.expand(p.shape[0], w.shape[0])

    diag_p = torch.diag_embed(p)

    p_w = torch.matmul(w, diag_p)

    z_p_w = torch.zeros_like(p_w)
    shft_p_w = torch.cat((p_w, z_p_w), -1)

    sum_of_cross = torch.zeros_like(p_w)
    # print(p,p.shape)
    length = p.shape[1]

    for shft in range(1, length):
        sum_of_cross += shft_p_w[:, :, 0:length] * shft_p_w[:, :, shft:length + shft]

    sum_of_cross = sum_of_cross.sum(-1)

    return (sum_of_sq + 2 * sum_of_cross) / (length ** 2)


# %%



def original_design(input, w):
    # data preparation
    angle = input.acos()
    # circuit dsign
    q_io = qk.QuantumRegister(3, "io")
    # q_n = qk.QuantumRegister(1,"neural")
    c = qk.ClassicalRegister(1, "reg")

    circuit = qk.QuantumCircuit(q_io, c)
    circuit.ry(angle[0].item(), q_io[0])
    circuit.ry(angle[1].item(), q_io[1])
    circuit.h(q_io[2])
    circuit.barrier()
    if w[0] == 1:
        circuit.x(q_io[0])
    if w[1] == 1:
        circuit.x(q_io[1])
    circuit.barrier()
    circuit.x(q_io[2])
    circuit.cz(q_io[0], q_io[2])
    circuit.x(q_io[2])
    circuit.cz(q_io[1], q_io[2])
    circuit.barrier()
    circuit.h(q_io[2])
    circuit.x(q_io[2])
    circuit.measure(q_io[2], c)

    mapping = {q_io[0]: 1, q_io[1]: 3, q_io[2]: 4}

    return circuit, mapping


def opt_1_design(input, w):
    # data preparation
    angle = input.acos()
    # circuit dsign
    q_io = qk.QuantumRegister(3, "io")
    # q_n = qk.QuantumRegister(1,"neural")
    c = qk.ClassicalRegister(1, "reg")

    circuit = qk.QuantumCircuit(q_io, c)
    circuit.ry(angle[0].item(), q_io[0])
    circuit.ry(angle[1].item(), q_io[1])
    circuit.h(q_io[2])
    circuit.barrier()
    if w[0] == 1:
        circuit.x(q_io[0])
    if w[1] == 1:
        circuit.x(q_io[1])
    circuit.barrier()
    circuit.cx(q_io[0], q_io[1])
    circuit.cz(q_io[1], q_io[2])
    circuit.barrier()
    circuit.h(q_io[2])
    circuit.x(q_io[2])
    circuit.measure(q_io[2], c)

    mapping = {q_io[0]: 1, q_io[1]: 3, q_io[2]: 4}

    return circuit, mapping


def opt_2_design(input, w):
    # data preparation
    angle = input.acos()
    # circuit dsign
    q_io = qk.QuantumRegister(2, "io")
    # q_n = qk.QuantumRegister(1,"neural")
    c = qk.ClassicalRegister(1, "reg")

    circuit = qk.QuantumCircuit(q_io, c)
    circuit.ry(angle[0].item(), q_io[0])
    circuit.ry(angle[1].item(), q_io[1])

    circuit.barrier()
    if w[0] == 1:
        circuit.x(q_io[0])
    if w[1] == 1:
        circuit.x(q_io[1])
    circuit.barrier()
    circuit.cx(q_io[0], q_io[1])

    circuit.barrier()
    circuit.x(q_io[1])
    circuit.measure(q_io[1], c)
    mapping = {q_io[0]: 3, q_io[1]: 4}

    return circuit, mapping


def opt_3_design(input, w):
    # data preparation
    input = (1 - input) / 2
    angle = (1 - (input[0] + input[1] - 2 * input[0] * input[1]) * 2).acos()
    # circuit dsign
    q_io = qk.QuantumRegister(1, "io")
    # q_n = qk.QuantumRegister(1,"neural")
    c = qk.ClassicalRegister(1, "reg")

    circuit = qk.QuantumCircuit(q_io, c)
    circuit.ry(angle.item(), q_io[0])

    circuit.barrier()
    circuit.measure(q_io[0], c)
    mapping = {q_io[0]: 0}

    return circuit, mapping


def test_design():
    # circuit dsign
    q_io = qk.QuantumRegister(1, "io")
    # q_n = qk.QuantumRegister(1,"neural")
    c = qk.ClassicalRegister(1, "reg")

    circuit = qk.QuantumCircuit(q_io, c)
    circuit.x(q_io[0])
    circuit.measure(q_io[0], c)
    mapping = {q_io[0]: 4}

    return circuit, mapping



results = {}

for i in range(11):


    input_ori = [i/10,(10-i)/10]
    input = 1 - torch.tensor(input_ori) * 2
    w = [0.0, 1.0]

    # print(input_ori,"start")



    ori_circuit, ori_mapping = original_design(input, w)
    opt1_circuit, opt1_mapping = opt_1_design(input, w)
    opt2_circuit, opt2_mapping = opt_2_design(input, w)
    opt3_circuit, opt3_mapping = opt_3_design(input, w)

    test_circuit, test_mapping = test_design()

    # print(test_circuit)

    qc_shots = 8192
    num_c_reg = 1


    t_input = 1 - torch.tensor([input_ori]) * 2
    t_w = 1 - torch.tensor([w]) * 2

    theoretic_output = do_slp_via_th(t_input, t_w)

    results[tuple(input_ori)] = []
    results[tuple(input_ori)].append(theoretic_output)
    print(theoretic_output)

    #
    #
    counts = fire_ibmq(opt3_circuit, qc_shots, 1, True, False,mapping=opt3_mapping)
    (mycount, bits) = analyze(counts[0])
    for b in range(bits):
        sim_res = float(mycount[b]) / qc_shots
        results[tuple(input_ori)].append(sim_res)
        print("qiskit_sim",sim_res)
    print()

    #
    # counts = fire_ibmq(ori_circuit, qc_shots, 1, False, False, backend_name="ibmq_valencia", mapping=ori_mapping)
    # (mycount, bits) = analyze(counts[0])
    # for b in range(bits):
    #     ori_res = float(mycount[b]) / qc_shots
    #     results[tuple(input_ori)].append(float(mycount[b]) / qc_shots)
    #
    #
    # counts = fire_ibmq(opt1_circuit, qc_shots, 1, False, False, backend_name="ibmq_valencia", mapping=opt1_mapping)
    # (mycount, bits) = analyze(counts[0])
    # for b in range(bits):
    #     opt1_res =  float(mycount[b]) / qc_shots
    #     results[tuple(input_ori)].append(float(mycount[b]) / qc_shots)
    #
    # counts = fire_ibmq(opt2_circuit, qc_shots, 1, False, False, backend_name="ibmq_valencia", mapping=opt2_mapping)
    # (mycount, bits) = analyze(counts[0])
    # for b in range(bits):
    #     opt2_res = float(mycount[b]) / qc_shots
    #     results[tuple(input_ori)].append(float(mycount[b]) / qc_shots)
    #
    # counts = fire_ibmq(opt3_circuit, qc_shots, 1, False, False, backend_name="ibmq_valencia", mapping=opt3_mapping)
    # (mycount, bits) = analyze(counts[0])
    # for b in range(bits):
    #     opt3_res = float(mycount[b]) / qc_shots
    #     results[tuple(input_ori)].append(float(mycount[b]) / qc_shots)

    # counts = fire_ibmq(test_circuit, qc_shots, 1, False, False, backend_name="ibmq_valencia", mapping=test_mapping)
    # (mycount, bits) = analyze(counts[0])
    # for b in range(bits):
    #     print(float(mycount[b]) / qc_shots)




    # print(input_ori,theoretic_output,ori_res,opt1_res,opt2_res,opt3_res)
    # print(input_ori, theoretic_output, opt3_res)
    # print(input_ori, theoretic_output, sim_res)
