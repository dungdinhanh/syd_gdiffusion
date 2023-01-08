"""
Like image_sample.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.
"""

import argparse
import os

import numpy as np
import math
import torch as th
# import torch.distributed as dist
import hfai.nccl.distributed as dist
import torch.nn.functional as F
import hfai
from guided_diffusion_hfai import dist_util, logger
from guided_diffusion_hfai.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    # create_model_and_diffusion,
    create_classifier,
    add_dict_to_argparser,
    args_to_dict,
)
import hfai.client
import hfai.multiprocessing

from guided_diffusion_hfai.script_util_mlt import create_model_and_diffusion_mlt2
import datetime
from PIL import Image
import hfai.client


def main(local_rank):
    args = create_argparser().parse_args()

    dist_util.setup_dist(local_rank)

    save_folder = os.path.join(
        args.logdir,
        "logs",
    )

    logger.configure(save_folder, rank=dist.get_rank())

    output_images_folder = os.path.join(args.logdir, "reference")
    os.makedirs(output_images_folder, exist_ok=True)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion_mlt2(
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

    def cond_fn(x, t, y=None):
        assert y is not None
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            return th.autograd.grad(selected.sum(), x_in)[0] * args.classifier_scale

    def model_fn(x, t, y=None):
        assert y is not None
        return model(x, t, y if args.class_cond else None)

    logger.log("Looking for previous file")
    checkpoint = os.path.join(output_images_folder, "samples_last.npz")
    temp_checkpoint = os.path.join(output_images_folder, "temp_checkpoint.pt")
    half_save = args.half_save
    if half_save:
        mark = "_part1" # always start with part 1
        num_samples = int(args.num_samples / 2)
        final_file1 = os.path.join(output_images_folder,
                                  f"samples_{num_samples}x{args.image_size}x{args.image_size}x3_part1.npz")
        if os.path.isfile(final_file1):
            logger.log("Complete half_save")
            half_save = False
            mark = "_part2" # if part 1 complete then start with part 2
        final_file2 = os.path.join(output_images_folder,
                                  f"samples_{num_samples}x{args.image_size}x{args.image_size}x3_part2.npz")
        if os.path.isfile(final_file2):
            dist.barrier()
            logger.log("sampling complete")
            return
    else:
        num_samples = args.num_samples
        final_file = os.path.join(output_images_folder,
                                  f"samples_{num_samples}x{args.image_size}x{args.image_size}x3.npz")
        if os.path.isfile(final_file):
            dist.barrier()
            logger.log("sampling complete")
            return
        mark=""
    if os.path.isfile(checkpoint):
        npzfile = np.load(checkpoint)
        all_images = list(npzfile['arr_0'])
        all_labels = list(npzfile['arr_1'])
        # this is to fix the previous problem where no half split
        if half_save:
            if len(all_images) * args.batch_size >= num_samples:
                index_part1 = int(math.ceil(float(num_samples)/float(args.batch_size)))
                all_images_p1 = all_images[:index_part1]
                all_labels_p1 = all_labels[:index_part1]
                if hfai.client.receive_suspend_command():
                    print("Receive suspend - good luck next run ^^")
                    hfai.client.go_suspend()
                save_to_file(all_images_p1, all_labels_p1, num_samples, output_images_folder, mark)
                all_images = all_images[index_part1:]
                all_labels = all_labels[index_part1:]
                mark = "_part2"
                logger.log("sampling part 1 complete")
                logger.log("sampling part 2")
                half_save = False
    else:
        all_images = []
        all_labels = []
    logger.log(f"Number of current images: {len(all_images) * args.batch_size}")
    logger.log("sampling...")
    if args.image_size == 28:
        img_channels = 1
        num_class = 10
    else:
        img_channels = 3
        num_class = NUM_CLASSES

    while len(all_images) * args.batch_size < num_samples:
        model_kwargs = {}
        classes = th.randint(
            low=0, high=num_class, size=(args.batch_size,), device=dist_util.dev()
        )
        model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model_fn,
            (args.batch_size, img_channels, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn,
            device=dist_util.dev(),
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        batch_images = [sample.cpu().numpy() for sample in gathered_samples]
        all_images.extend(batch_images)
        gathered_labels = [th.zeros_like(classes) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_labels, classes)
        batch_labels = [labels.cpu().numpy() for labels in gathered_labels]
        all_labels.extend(batch_labels)
        if len(all_images) * args.batch_size >= num_samples and half_save:
            half_save = False
            if hfai.client.receive_suspend_command():
                print("Receive suspend - good luck next run ^^")
                hfai.client.go_suspend()
            save_to_file(all_images, all_labels, num_samples, output_images_folder, "_part1")
            mark="_part2"
            del all_images, all_labels
            all_images = []
            all_labels = []
            if dist.get_rank() == 0:
                os.remove(checkpoint)
            continue

        if dist.get_rank() == 0:
            if hfai.client.receive_suspend_command():
                print("Receive suspend - good luck next run ^^")
                hfai.client.go_suspend()
            logger.log(f"created {len(all_images) * args.batch_size} samples")

            np.savez(temp_checkpoint, np.stack(all_images), np.stack(all_labels))
            if hfai.client.receive_suspend_command():
                print("Receive suspend - good luck next run ^^")
                hfai.client.go_suspend()
            os.remove(checkpoint)
            os.rename(temp_checkpoint, checkpoint)


    save_to_file(all_images, all_labels, num_samples, output_images_folder, mark)
    if dist.get_rank() == 0:
        os.remove(checkpoint)

    dist.barrier()
    logger.log("sampling complete")


def save_to_file(all_images, all_labels, num_samples, output_images_folder, mark=""):
    arr = np.concatenate(all_images, axis=0)
    arr = arr[: num_samples]
    label_arr = np.concatenate(all_labels, axis=0)
    label_arr = label_arr[: num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(output_images_folder, f"samples_{shape_str}{mark}.npz")
        logger.log(f"saving to {out_path}")
        np.savez(out_path, arr, label_arr)


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=50000,
        batch_size=16,
        use_ddim=False,
        model_path="",
        classifier_path="",
        classifier_scale=1.0,
        logdir="",
        half_save=False
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser



if __name__ == "__main__":
    ngpus = th.cuda.device_count()
    hfai.multiprocessing.spawn(main, args=(), nprocs=ngpus, bind_numa=False)
