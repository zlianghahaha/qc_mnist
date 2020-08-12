# %%
import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys
import copy
import argparse
import time


class ToQuantumData(object):
    def __call__(self, tensor):
        # torch.set_printoptions(profile="full")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data = tensor.to(device)
        input_vec = data.view(-1)

        input_vec = input_vec.to(device)
        vec_len = input_vec.size()[0]
        input_matrix = torch.zeros(vec_len, vec_len)
        input_matrix = input_matrix.to(device)
        input_matrix[0] = input_vec
        input_matrix = input_matrix.transpose(0, 1)

        u, s, v = torch.svd(input_matrix)
        u = u.to(device)
        v = v.to(device)
        output_matrix = u.matmul(v)
        # print(output_matrix.shape)
        # u, s, v = np.linalg.svd(input_matrix)
        # output_matrix = torch.tensor(np.dot(u, v))
        # print(output_matrix.shape)
        # sys.exit(0)
        output_data = output_matrix[:, 0].view(tensor.shape)
        return output_data



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='QuantumFlow Qiskit Simulation')
    parser.add_argument('-s','--size',default="4",help="image resize",)
    parser.add_argument('-t', "--test", help="Only Test without Training", action="store_true", )
    args = parser.parse_args()

    print("=" * 100)
    print("Data Convert for MNIST. This script is for converting data to unitray matrix.")
    print("\tStart at:", time.strftime("%m/%d/%Y %H:%M:%S"))
    print("\tProblems and issues, please contact Dr. Weiwen Jiang (wjiang2@nd.edu)")
    print("\tEnjoy and Good Luck!")
    print("=" * 100)
    print()

    size = int(args.size)
    istest = args.test

    data_str = "training"
    if istest:
        data_str = "test"

    res = torch.load("processed/"+data_str+".pt")
    print(res[0].shape, res[1].shape)

    trans_to_tensor = transforms.ToTensor()
    trans_to_qc_data = ToQuantumData()
    trans_resize = transforms.Resize((size, size))

    data = torch.zeros(res[0].shape[0], size, size, dtype=torch.float32)

    for i in range(res[0].shape[0]):
        # print(res[0][i].shape)
        if i % 500 == 0:
            print(i)
        npimg = res[0][i].numpy()

        im = Image.fromarray(npimg, mode="L")
        qc_data = trans_to_qc_data(trans_to_tensor(trans_resize(im)))[0]

        data[i] = qc_data

    # %%
    x = (data, res[1])
    torch.save(x, 'processed/qc_'+data_str+'_'+str(size)+'_'+str(size)+'.pt')
    print("Convert Done!!!!")
    print("stored at:",'processed/'+data_str+'_'+str(size)+'_'+str(size)+'.pt')