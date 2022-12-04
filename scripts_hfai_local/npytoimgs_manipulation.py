import numpy as np
from PIL import Image
import argparse
import os


def read_file_to_numpy(file):
    npzfile = np.load(file)
    n = len(npzfile)
    if n == 1:
        return list(npzfile['arr_0'])
    if n == 2:
        all_images = list(npzfile['arr_0'])
        all_labels = list(npzfile['arr_1'])
        return all_images, all_labels
    if n == 3:
        all_images = list(npzfile['arr_0'])
        all_labels = list(npzfile['arr_1'])
        all_cleans = npzfile['arr_2']


        all_cleans = np.clip((all_cleans + 1) * 127.5, 0, 255).astype(np.uint8)
        all_cleans = all_cleans.transpose((0, 2, 3, 1))
        all_cleans = list(all_cleans)
        return all_images, all_labels, all_cleans
    else:
        print("weird qua ba noi")
        return None
    pass


def load_numpy_to_image(array, folder_path):
    for i in range(len(array)):
        im = Image.fromarray(array[i], "L")
        im.save(os.path.join(folder_path, f"image%d.png"%i))
    pass


def load_numpy_labels_to_image(array1, array2, folder_path):
    for i in range(len(array1)):
        img = np.squeeze(array1[i], axis=-1)
        im = Image.fromarray(img, "L")
        im.save(os.path.join(folder_path, f"%d_image%d.png"%(int(array2[i]), i)))

def load_numpy_labels_to_image2(array1, array2, array3, folder_path):
    gen_path = os.path.join(folder_path, "gen_images")
    real_path = os.path.join(folder_path, "real_images")
    os.makedirs(gen_path, exist_ok=True)
    os.makedirs(real_path, exist_ok=True)
    for i in range(len(array1)):
        img = np.squeeze(array1[i], axis=-1)
        im = Image.fromarray(img, "L")
        im.save(os.path.join(folder_path, "gen_images", f"%d_image%d.png"%(int(array2[i]), i)))
        img_clean = np.squeeze(array3[i])
        im_clean = Image.fromarray(img_clean, "L")
        im_clean.save(os.path.join(folder_path, "real_images", f"%d_image%d.png"%(int(array2[i]), i)))



parser = argparse.ArgumentParser("numpy to images")
parser.add_argument("--numpy", help="numpy files", default="../outputhfai/runs/")



if __name__ == '__main__':
    args = parser.parse_args()
    file_numpy = args.numpy
    folder_images = os.path.join(os.path.dirname(file_numpy), "images")
    os.makedirs(folder_images, exist_ok=True)
    images_array, labels, images_clean = read_file_to_numpy(file_numpy)
    load_numpy_labels_to_image2(images_array, labels, images_clean, folder_images)