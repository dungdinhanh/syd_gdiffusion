from guided_diffusion_hfai.image_datasets import *


def get_data(dataset, batch_size):
    if dataset == 'MNIST':

        dataset = MNISTHF(split='train', classes=False)
        num_worker=8
    elif dataset == 'CelebA':
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                (0.5, 0.5, 0.5))])
        num_worker=1
        dataset = CelebALocal(root="../data_local/", classes=False, split='train', resolution=32)
    else:
        return None
    loader = DataLoader(dataset, batch_size, num_workers=num_worker, shuffle=True)
    return loader
    # while True:
    #     yield from loader
    pass