import argparse
import os

import torch
import torchvision.utils as vutils
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scripts_infogan.utils import noise_sample

parser = argparse.ArgumentParser()
parser.add_argument('--load_path', required=True, help='Checkpoint to load path from')
parser.add_argument('--batch', default=512, help="Batch size")
parser.add_argument('--num', default=50000, help="number of generated images")
parser.add_argument('--save', default="../output/infogan_analyse/celeba32/", help="saved folder")
args = parser.parse_args()

from guided_diffusion_hfai.infogan.celeba_model import Generator

# Load the checkpoint file
state_dict = torch.load(args.load_path)

# Set the device to run on: GPU or CPU.
device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
# Get the 'params' dictionary from the loaded state_dict.
params = state_dict['params']

# Create the generator network.
netG = Generator().to(device)
# Load the trained generator weights.
netG.load_state_dict(state_dict['netG'])
print(netG)

# c = np.linspace(-2, 2, 10).reshape(1, -1)
# c = np.repeat(c, 10, 0).reshape(-1, 1)
# c = torch.from_numpy(c).float().to(device)
# c = c.view(-1, 1, 1, 1)
#
# zeros = torch.zeros(100, 1, 1, 1, device=device)
#
# # Continuous latent code.
# c2 = torch.cat((c, zeros), dim=1)
# c3 = torch.cat((zeros, c), dim=1)
#
# idx = np.arange(10).repeat(10)
# dis_c = torch.zeros(100, 10, 1, 1, device=device)
# dis_c[torch.arange(0, 100), idx] = 1.0
# # Discrete latent code.
# c1 = dis_c.view(100, -1, 1, 1)


# # To see variation along c3 (Horizontally) and c1 (Vertically)
# noise2 = torch.cat((z, c1, c3), dim=1)
save_folder_images = os.path.join(args.save, "images")
os.makedirs(save_folder_images, exist_ok=True)
# Generate image.
count_image = 0
gen_image = True
num_z = 128
num_dis_c = 10
dis_c_dim = 10
num_con_c = 0
all_images =[]
all_labels = []

while gen_image:
    noise, idx = noise_sample(num_dis_c, dis_c_dim, num_con_c, num_z, args.batch, device)
    with torch.no_grad():
        generated_img = netG(noise).detach().cpu().numpy()
        generated_img = np.clip((generated_img + 1) * 127.5, 0, 255).astype(np.uint8)
        generated_img = generated_img.transpose((0, 2, 3, 1))
    idx = np.transpose(idx)
    all_images.append(generated_img)
    all_labels.append(idx)

    for i, img in enumerate(generated_img):
        # print(img.shape)
        # exit()
        # img = np.squeeze(img)
        im = Image.fromarray(img, "RGB")
        cat_str = ""
        for i_dim in range(len(idx[i])):
            cat_str += f"{int(idx[i][i_dim])}"
        im.save(os.path.join(save_folder_images, f"{cat_str}_image%d.png"%( count_image)))
        count_image += 1
        im.close()
        if count_image >= args.num:
            gen_image = False
            all_images = np.concatenate(all_images, axis=0)
            all_labels  = np.concatenate(all_labels, axis=0)
            all_images = all_images[:args.num]
            all_labels = all_labels[:args.num]
            shape_str = "x".join([str(int(ix)) for ix in all_images.shape])
            save_npz_path = os.path.join(args.save, f"samples_{shape_str}.npz")
            np.savez(save_npz_path, all_images, all_labels)
            print("Complete sampling")
            break
        pass




# Display the generated image.
# fig = plt.figure(figsize=(10, 10))
# plt.axis("off")
# plt.imshow(np.transpose(vutils.make_grid(generated_img1, nrow=10, padding=2, normalize=True), (1,2,0)))
# plt.show()
#
# # Generate image.
# with torch.no_grad():
#     generated_img2 = netG(noise2).detach().cpu()
# # Display the generated image.
# fig = plt.figure(figsize=(10, 10))
# plt.axis("off")
# plt.imshow(np.transpose(vutils.make_grid(generated_img2, nrow=10, padding=2, normalize=True), (1,2,0)))
# plt.show()