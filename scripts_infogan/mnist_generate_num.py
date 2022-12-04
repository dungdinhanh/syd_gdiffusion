import argparse
import os

import torch
import torchvision.utils as vutils
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--load_path', required=True, help='Checkpoint to load path from')
parser.add_argument('--batch', default=256, help="Batch size")
parser.add_argument('--num', default=10000, help="number of generated images")
parser.add_argument('--save', default="../output/infogan_analyse/mnist/", help="saved folder")
args = parser.parse_args()

from guided_diffusion_hfai.infogan.mnist_model import Generator

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
save_folder = args.save
os.makedirs(save_folder, exist_ok=True)
# Generate image.
count_image = 0
gen_image = True
while gen_image:
    z = torch.randn(args.batch, 62, 1, 1, device=device)
    idx = torch.randint(low=0, high=10, size=(args.batch,), device=device)
    dis_c = torch.zeros(args.batch, 10, 1, 1, device=device)
    dis_c[torch.arange(0, args.batch), idx] = 1.0
    c = torch.randn(args.batch, 2, 1, 1, device=device)

    # To see variation along c2 (Horizontally) and c1 (Vertically)
    noise = torch.cat((z, dis_c, c), dim=1)
    with torch.no_grad():
        generated_img = netG(noise).detach().cpu().numpy()
        generated_img = np.clip((generated_img + 1) * 127.5, 0, 255).astype(np.uint8)
    for i, img in enumerate(generated_img):
        img = np.squeeze(img)
        im = Image.fromarray(img, "L")
        im.save(os.path.join(save_folder, f"%d_image%d.png"%(idx[i], count_image)))
        count_image += 1
        im.close()
        if count_image >= args.num:
            gen_image=False
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