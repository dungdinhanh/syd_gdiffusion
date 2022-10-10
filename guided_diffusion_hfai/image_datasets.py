import math
import random
from typing import Callable, Optional
from PIL import Image
import pickle
import blobfile as bf
# from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset
from hfai.datasets.imagenet import ImageNet
from torchvision import transforms, models, datasets
from torch.utils.data.distributed import DistributedSampler


# def load_data(
#     *,
#     data_dir,
#     batch_size,
#     image_size,
#     class_cond=False,
#     deterministic=False,
#     random_crop=False,
#     random_flip=True,
# ):
#     """
#     For a dataset, create a generator over (images, kwargs) pairs.
#
#     Each images is an NCHW float tensor, and the kwargs dict contains zero or
#     more keys, each of which map to a batched Tensor of their own.
#     The kwargs dict can be used for class labels, in which case the key is "y"
#     and the values are integer tensors of class labels.
#
#     :param data_dir: a dataset directory.
#     :param batch_size: the batch size of each returned pair.
#     :param image_size: the size to which images are resized.
#     :param class_cond: if True, include a "y" key in returned dicts for class
#                        label. If classes are not available and this is true, an
#                        exception will be raised.
#     :param deterministic: if True, yield results in a deterministic order.
#     :param random_crop: if True, randomly crop the images for augmentation.
#     :param random_flip: if True, randomly flip the images for augmentation.
#     """
#     if not data_dir:
#         raise ValueError("unspecified data directory")
#     all_files = _list_image_files_recursively(data_dir)
#     classes = None
#     if class_cond:
#         # Assume classes are the first part of the filename,
#         # before an underscore.
#         class_names = [bf.basename(path).split("_")[0] for path in all_files]
#         sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
#         classes = [sorted_classes[x] for x in class_names]
#     dataset = ImageDataset(
#         image_size,
#         all_files,
#         classes=classes,
#         shard=MPI.COMM_WORLD.Get_rank(),
#         num_shards=MPI.COMM_WORLD.Get_size(),
#         random_crop=random_crop,
#         random_flip=random_flip,
#     )
#     if deterministic:
#         loader = DataLoader(
#             dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
#         )
#     else:
#         loader = DataLoader(
#             dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
#         )
#     while True:
#         yield from loader


def load_data_imagenet_hfai(
    *,
    train=True,
    batch_size,
    image_size,
    random_crop=False,
    random_flip=True,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    # if not data_dir:
    #     raise ValueError("unspecified data directory")
    # all_files = _list_image_files_recursively(data_dir)
    # classes = None
    # if class_cond:
    #     # Assume classes are the first part of the filename,
    #     # before an underscore.
    #     class_names = [bf.basename(path).split("_")[0] for path in all_files]
    #     sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
    #     classes = [sorted_classes[x] for x in class_names]
    # dataset = ImageDataset(
    #     image_size,
    #     all_files,
    #     classes=classes,
    #     shard=MPI.COMM_WORLD.Get_rank(),
    #     num_shards=MPI.COMM_WORLD.Get_size(),
    #     random_crop=random_crop,
    #     random_flip=random_flip,
    # )
    if train:
        dataset = ImageNetHF(image_size, random_crop=random_crop, random_flip=random_flip, split='train')
    else:
        dataset = ImageNetHF(image_size, random_crop=random_crop, random_flip=random_flip, split='val')
    # if deterministic:
    #     loader = DataLoader(
    #         dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
    #     )
    # else:
    #     loader = DataLoader(
    #         dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
    #     )
    data_sampler = DistributedSampler(dataset, shuffle=True)
    loader = dataset.loader(batch_size, num_workers=8, sampler=data_sampler, pin_memory=True)
    while True:
        yield from loader # put all items of loader into list and concat all list infinitely
    # return loader

def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        classes=None,
        shard=0,
        num_shards=1,
        random_crop=False,
        random_flip=True,
    ):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")

        if self.random_crop:
            arr = random_crop_arr(pil_image, self.resolution)
        else:
            arr = center_crop_arr(pil_image, self.resolution)

        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]

        arr = arr.astype(np.float32) / 127.5 - 1

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        return np.transpose(arr, [2, 0, 1]), out_dict


class ImageNetHF(ImageNet):
    def __init__(self, resolution, random_crop=False, random_flip=True, split='train'):
        to_tensor_transform = transforms.Compose([transforms.ToTensor()])
        super(ImageNetHF, self).__init__(split=split, transform=to_tensor_transform, check_data=True, miniset=False)
        self.resolution = resolution
        self.random_crop = random_crop
        self.random_flip = random_flip

    def __getitem__(self, indices):
        imgs_bytes = self.reader.read(indices)
        samples = []
        for i, bytes_ in enumerate(imgs_bytes):
            img = pickle.loads(bytes_).convert("RGB")
            label = self.meta["targets"][indices[i]]
            samples.append((img, int(label)))

        transformed_samples = []
        for img, label in samples:
            if self.random_crop:
                arr = random_crop_arr(img, self.resolution)
            else:
                arr = center_crop_arr(img, self.resolution)

            if self.random_flip and random.random() < 0.5:
                arr = arr[:, ::-1]

            img = arr.astype(np.float32) / 127.5 - 1
            # img = np.transpose(img, [2, 0, 1]) # might not need to transpose
            if self.transform:
                img = self.transform(img)
            out_dict = {}
            # if self.local_classes is not None:
            out_dict["y"] = label

            transformed_samples.append((img, out_dict))
        return transformed_samples


def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
