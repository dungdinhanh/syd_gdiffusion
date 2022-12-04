import numpy as np
import argparse
import os
from evaluations.svd_cal import *

parser = argparse.ArgumentParser("Input for svd visulaization")
parser.add_argument("--f1", type=str, default="../outputhfai/local_runs_info/mnist_info2_classifier_training/")
parser.add_argument("--f2", type=str, default="../outputhfai/local_runs_info/mnist_info3_classifier_training/")
parser.add_argument("--f3", type=str, default="../outputhfai/local_runs_info/mnist_info4_classifier_training/")
parser.add_argument("--n1", type=str, default="info2")
parser.add_argument("--n2", type=str, default="info3")
parser.add_argument("--n3", type=str, default="info4")
parser.add_argument("--save", type=str, default="../outputhfai/runs_analyse/svd")
parser.add_argument("--name", type=str, default="SVD visualization")

FILE_NAME="logits_10000x10.npz"


def process_file_name(folder, name_args):
    if folder is not None:
        if os.path.isdir(folder):
            file = os.path.join(folder, "tsne", FILE_NAME)
            if name_args == "default":
                name = os.path.basename(folder)
            else:
                name = name_args
            return file, name
        return None
    return None


if __name__ == '__main__':
    args = parser.parse_args()
    list_logits = []
    list_names = []
    file1, name1 = process_file_name(args.f1, args.n1)
    file2, name2 = process_file_name(args.f2, args.n2)
    file3, name3 = process_file_name(args.f3, args.n3)

    logits1 = np.load(file1)['arr_0']
    logits2 = np.load(file2)['arr_0']
    logits3 = np.load(file3)['arr_0']

    os.makedirs(args.save, exist_ok=True)
    save_svd_file = os.path.join(args.save, "svd.png")

    svd_visulaize([logits1, logits2, logits3], [name1, name2, name3], save_svd_file)








