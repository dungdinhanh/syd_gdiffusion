"""
Like image_sample.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.
"""

import argparse
import os

import numpy as np
import torch as th
# import torch.distributed as dist
import hfai.nccl.distributed as dist
import torch.nn.functional as F
import hfai
from guided_diffusion_hfai.image_datasets import load_dataset_MNIST_nosampler
from guided_diffusion_hfai import dist_util, logger
from guided_diffusion_hfai.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    add_dict_to_argparser,
    args_to_dict,
)
import datetime
from PIL import Image
from evaluations.tsne_cal import *


def main():
    args = create_argparser().parse_args()

    # dist_util.setup_dist()
    if os.path.isfile(args.classifier_path):
        save_folder = os.path.abspath(os.path.join(os.path.dirname(args.classifier_path), os.pardir))
        log_folder = os.path.join(save_folder, "tsne_log")
    else:
        print("No classifier path valid")
        exit(0)
    # save_folder = os.path.join(
    #     args.logdir,
    #     "logs",
    # )

    logger.configure(log_folder, rank=0)

    output_images_folder = os.path.join(save_folder, "tsne")
    os.makedirs(output_images_folder, exist_ok=True)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()


    logger.log("loading classifier...")
    classifier = create_classifier(**args_to_dict(args, classifier_defaults().keys()))
    classifier.load_state_dict(
        dist_util.load_state_dict(args.classifier_path, map_location="cpu")
    )
    classifier.to(dist_util.dev())
    if args.classifier_use_fp16:
        classifier.convert_to_fp16()
    classifier.eval()

    logger.log("creating data loader...")
    # data = load_data(
    #     data_dir=args.data_dir,
    #     batch_size=args.batch_size,
    #     image_size=args.image_size,
    #     class_cond=True,
    #     random_crop=True,
    # )
    data = load_dataset_MNIST_nosampler(
        train=True, batch_size=args.batch_size, class_cond=True
    )

    def cond_fn(x, t):
        x_in = x.detach()
        logits = classifier(x_in, t)
        return logits

    all_images = []
    all_labels = []
    logger.log("sampling...")

    while len(all_images) * args.batch_size < args.num_samples:
        batch, extra = next(data)
        batch = batch.to(dist_util.dev())
        classes = extra["y"].numpy()

        t_clean = th.zeros(batch.shape[0], dtype=th.long, device=dist_util.dev())

        logits = cond_fn(batch, t_clean).detach().cpu().numpy()

        all_images.append(logits)
        all_labels.append(classes)

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    label_arr = np.concatenate(all_labels, axis=0)
    label_arr = label_arr[: args.num_samples]

    tsne_array = logits_labels_tsne(arr, label_arr, os.path.join(output_images_folder, "tsne.png"))

    shape_str = "x".join([str(x) for x in arr.shape])
    out_path = os.path.join(output_images_folder, f"logits_{shape_str}.npz")
    logger.log(f"saving to {out_path}")
    np.savez(out_path, arr, tsne_array[0], label_arr)

    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
        classifier_path="",
        classifier_scale=1.0,
        logdir=""
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser



if __name__ == "__main__":
    # ngpus = th.cuda.device_count()
    # hfai.multiprocessing.spawn(main, args=(), nprocs=ngpus, bind_numa=True)
    main()
