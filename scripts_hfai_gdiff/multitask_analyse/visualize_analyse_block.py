import os.path

import numpy as np
import matplotlib.pyplot as plt


def angle_distribution(angle_matrix):
    angle_shape = angle_matrix.shape
    time_steps = angle_shape[1]
    angle_binning_vectors = []
    for i in range(time_steps):
        angle_timestepi = np.histogram(angle_matrix[:, i], bins=180, range=[0.0, 180.0])
        angle_binning_vectors.append(angle_timestepi)
    return angle_binning_vectors
    pass


def magnitude_distribution(magnitude_matrix):
    magnitude_shape = magnitude_matrix.shape
    time_steps = magnitude_shape[1]
    magnitude_binning_vectors = []
    min = 0.0
    max = 112
    for i in range(time_steps):
        magnitude_timestepi = np.histogram(magnitude_matrix[:, i], bins=30, range=[min, max])
        magnitude_binning_vectors.append(magnitude_timestepi)
    return magnitude_binning_vectors
    pass


def plot_vector_magnitude(list_vector, file_name):
    n = len(list_vector)
    figure, axis = plt.subplots(n, 1)
    list_values = [list_vector[i][0] for i in range(n)]
    list_x = [list_vector[i][1] for i in range(n)]
    list_x_name = []
    for i in range(list_x[0].shape[0] -1):
        list_x_name.append("{:0.1f}-{:0.1f}".format(float(list_x[0][i]), list_x[0][i+1]))
    for i in range(n):
        axis[i].bar(list_x_name, list_values[i], align='center', width=0.2)
        # if i == 0:
        #     axis[i].set_title(f"timestep_{i * 25}")
        # else:
        #     axis[i].set_title(f"timestep_{(i+5) * 25}")
        axis[i].set_title(f"timestep_{225 + i * 5}")
        for tick in axis[i].get_xticklabels():
            tick.set_rotation(45)
    plt.subplots_adjust(left=0.04,
                    bottom=0.1,
                    right=0.980,
                    top=0.921,
                    wspace=0.4,
                    hspace=0.929)
    figure.set_size_inches(20, 12)
    # plt.xticks(rotation=45, ha="right")

    plt.savefig(file_name, dpi=300)
    plt.close()



def plot_vector_angle(list_vector, file_name):
    n = len(list_vector)
    figure, axis = plt.subplots(n, 1)
    list_values = [list_vector[i][0] for i in range(n)]
    x_axis = np.arange(180)
    for i in range(n):
        axis[i].bar(x_axis, list_values[i], align='center', width=0.2)
        # axis[i].set_title(f"timestep_{i*25}")
        axis[i].set_title(f"timestep_{225 + i * 5}")
    # figure.tight_layout(pad = 10.0)
    plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.921,
                    top=0.921,
                    wspace=0.4,
                    hspace=0.929)
    figure.set_size_inches(20, 12)
    plt.savefig(file_name, dpi=300)
    plt.close()


def plot_magnitude(magnitude_matrix, file_name):
    mag_distribution = magnitude_distribution(magnitude_matrix)
    plot_vector_magnitude(mag_distribution, file_name)


def plot_angle(angle_matrix, file_name):
    angle_dist = angle_distribution(angle_matrix)
    plot_vector_angle(angle_dist, file_name)


def visualize_1block_3elements(npz_file, block_name):
    npzfile = np.load(npz_file)
    out_folder = os.path.dirname((npz_file))
    block_folder = os.path.join(out_folder, block_name)
    os.makedirs(block_folder, exist_ok=True)

    magnitude_cls = npzfile['arr_0']
    plot_magnitude(magnitude_cls, os.path.join(block_folder, "mag_cls.png"))

    magnitude_div = npzfile['arr_1']
    plot_magnitude(magnitude_div, os.path.join(block_folder, "magnitude_div.png"))

    magnitude_diff = npzfile['arr_2']
    plot_magnitude(magnitude_diff, os.path.join(block_folder, "magnitude_diff.png"))

    angle_cls_div = npzfile['arr_3']
    plot_angle(angle_cls_div, os.path.join(block_folder, "angle_cls_div.png"))

    angle_cls_diff = npzfile['arr_4']
    plot_angle(angle_cls_diff, os.path.join(block_folder, 'angle_cls_diff.png'))

    angle_div_diff = npzfile['arr_5']
    plot_angle(angle_div_diff, os.path.join(block_folder, "angle_div_diff.png"))


def visualize_1block_2elements(npz_file, block_name):
    npzfile = np.load(npz_file)
    out_folder = os.path.dirname((npz_file))
    block_folder = os.path.join(out_folder, block_name)
    os.makedirs(block_folder, exist_ok=True)


    magnitude_div = npzfile['arr_0']
    plot_magnitude(magnitude_div, os.path.join(block_folder, "magnitude_div.png"))

    magnitude_diff = npzfile['arr_1']
    plot_magnitude(magnitude_diff, os.path.join(block_folder, "magnitude_diff.png"))

    angle_div_diff = npzfile['arr_2']
    plot_angle(angle_div_diff, os.path.join(block_folder, "angle_div_diff.png"))


def visualize_analyse_3elements(analyse_folder):
    before_block = os.path.join(analyse_folder, "reference", 'analyse_before.npz')
    after_block = os.path.join(analyse_folder, "reference", "analyse_after.npz")

    visualize_1block_3elements(before_block, "before")
    visualize_1block_3elements(after_block, "after")

def visualize_analyse_3elements_no_mlt(analyse_folder):
    analyse_block = os.path.join(analyse_folder, "reference", 'analyse.npz')
    visualize_1block_3elements(analyse_block, "analyse")

def visualize_analyse_2elements(analyse_folder):
    before_block = os.path.join(analyse_folder, "reference", 'analyse_before.npz')
    after_block = os.path.join(analyse_folder, "reference", "analyse_after.npz")

    visualize_1block_2elements(before_block, "before")
    visualize_1block_2elements(after_block, "after")

def visualize_analyse_2elements_no_mlt(analyse_folder):
    analyse_block = os.path.join(analyse_folder, "reference", "analyse.npz")

    visualize_1block_2elements(analyse_block, "analyse")



if __name__ == '__main__':
    # test_folder = "/home/dzung/unisyddev/output/test_draft2/"
    # visualize_analyse(test_folder)
    log1 = "/home/dzung/unisyddev/output/analyse_diversity/IMN64_225250/unconditional_mlt2_ddiv/scale0.5"
    visualize_analyse_3elements(log1)

    log2 = "/home/dzung/unisyddev/output/analyse_diversity/IMN64_225250/unconditional_mlt2_cdiv/scale0.5"
    visualize_analyse_3elements(log2)

    log3 = "/home/dzung/unisyddev/output/analyse_diversity/IMN64_225250/unconditional_cls/scale0.5"
    visualize_analyse_3elements_no_mlt(log3)
    pass