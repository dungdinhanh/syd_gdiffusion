import argparse

import torch
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import copy

parser = argparse.ArgumentParser()
parser.add_argument('--load_path', required=True, help='Checkpoint to load path from')
parser.add_argument('--batch', default=256, help="Batch size")
parser.add_argument('--logdir', default="../output/infogan_analyse/celeba32_analyse")
args = parser.parse_args()

from guided_diffusion_hfai.infogan.celeba_model import Generator


def noise_sample(n_dis_c, dis_c_dim, n_con_c, z, idx, change_cat ,device):
    samples_per_row = 10
    batch_size = dis_c_dim * samples_per_row
    z = z

    new_idx = copy.deepcopy(idx)
    if (n_dis_c != 0):
        dis_c = torch.zeros(batch_size, n_dis_c, dis_c_dim, device=device)

        for i in range(dis_c_dim):
            new_idx[change_cat][i * samples_per_row: (i+1) * samples_per_row] = i
        for i in range(n_dis_c):
            dis_c[torch.arange(0, batch_size), i, idx[i]] = 1.0

        dis_c = dis_c.view(batch_size, -1, 1, 1)

    if (n_con_c != 0):
        # Random uniform between -1 and 1.
        con_c = torch.rand(batch_size, n_con_c, 1, 1, device=device) * 2 - 1
    noise = z
    if (n_dis_c != 0):
        noise = torch.cat((z, dis_c), dim=1)
    return noise, new_idx

if __name__ == '__main__':
    # Load the checkpoint file

    state_dict = torch.load(args.load_path)

    # Set the device to run on: GPU or CPU.
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    # Get the 'params' dictionary from the loaded state_dict.
    params = state_dict['params']
    num_z = params['num_z']
    num_dis_c = params['num_dis_c']
    dis_c_dim = params['dis_c_dim']
    # print(params)
    idx = np.zeros((num_dis_c, 10))
    for i in range(num_dis_c):
        idx[i] = np.random.randint(dis_c_dim, size=1)
    idx = np.repeat(idx, 10, axis=1)
    z = torch.randn(2, num_z, 1, 1, device=device)
    z = z.repeat(10, 1, 1, 1)
    print(torch.squeeze(z))
    exit(0)
    # noise, idx = noise_sample(num_dis_c, dis_c_dim, 0, num_z, device)
    # temp_noise = np.squeeze(noise)
    # print(temp_noise[10:20, 128:])
    # print(idx)
    # exit(0)

    # Create the generator network.
    netG = Generator().to(device)
    # Load the trained generator weights.
    netG.load_state_dict(state_dict['netG'])
    print(netG)

    zeros = torch.zeros(100, 1, 1, 1, device=device)

    idx = np.arange(10).repeat(10)
    dis_c = torch.zeros(100, 10, 1, 1, device=device)
    dis_c[torch.arange(0, 100), idx] = 1.0
    # Discrete latent code.
    c1 = dis_c.view(100, -1, 1, 1)

    z = torch.randn(100, 62, 1, 1, device=device)

    # To see variation along c2 (Horizontally) and c1 (Vertically)
    noise1 = torch.cat((z, c1), dim=1)

    # Generate image.
    with torch.no_grad():
        generated_img1 = netG(noise1).detach().cpu()
    # Display the generated image.
    fig = plt.figure(figsize=(10, 10))
    plt.axis("off")
    plt.imshow(np.transpose(vutils.make_grid(generated_img1, nrow=10, padding=2, normalize=True), (1,2,0)))
    plt.savefig()

    # Generate image.
