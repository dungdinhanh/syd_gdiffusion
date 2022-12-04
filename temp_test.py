import glob
import os
from pathlib import Path

def remove_temp_mark(save_folder):
    list_temp_files = list(glob.glob(os.path.join(save_folder, "*_temp.pt")))
    list_new_files = remove_temp_mark_str(list_temp_files)
    for i in range(len(list_new_files)):
        os.rename(list_temp_files[i], list_new_files[i])


def remove_temp_mark_str(list_files):
    new_list_files = []
    for file in list_files:
        new_file = file[:-8] + ".pt"
        new_list_files.append(new_file)
    return new_list_files
    pass

import blobfile as bf
if __name__ == '__main__':
    # os.rename("runs/temp12.txt", "runs/temp13.txt")
    first_file = "runs/temp12.txt"
    print(bf.exists(first_file))


