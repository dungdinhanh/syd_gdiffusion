"""
Like image_sample.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.

Analyse conflicts between diversity - diffusion -classification
"""

import argparse
import os

import numpy as np
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
import hfai.multiprocessing

from guided_diffusion_hfai.script_util_mlt import create_model_and_diffusion_mlt_analyse
import datetime
from PIL import Image
import matplotlib.pyplot as plt
#
# th.manual_seed(36)
# th.cuda.manual_seed_all(36)
#
# if th.cuda.is_available() :
#     th.backends.cudnn.benchmark = False
#     th.backends.cudnn.deterministic = True


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
    model, diffusion = create_model_and_diffusion_mlt_analyse(
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
    final_file = os.path.join(output_images_folder,
                              f"samples_{args.num_samples}x{args.image_size}x{args.image_size}x3.npz")
    # if os.path.isfile(final_file):
    #     dist.barrier()
    #     logger.log("sampling complete")
    #     return
    if os.path.isfile(checkpoint):
        npzfile = np.load(checkpoint)
        all_images = list(npzfile['arr_0'])
        all_labels = list(npzfile['arr_1'])
    else:
        all_images = []
        all_labels = []
    logger.log(f"Number of current images: {len(all_images)}")
    logger.log("sampling...")
    if args.image_size == 28:
        img_channels = 1
        num_class = 10
    else:
        img_channels = 3
        num_class = NUM_CLASSES
    # while len(all_images) * args.batch_size < args.num_samples:
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

    timestep_coss_dc = diffusion.timestep_coss_dc
    timestep_mags_dc = diffusion.timestep_mags_dc
    save_and_plot(timestep_coss_dc, timestep_mags_dc, output_images_folder, "diffusion_classification")
    # distilled_coss_dc, distilled_mags_dc = distill_information(timestep_coss_dc, timestep_mags_dc)
    print("dc done")
    timestep_coss_ddv = diffusion.timestep_coss_ddv
    timestep_mags_ddv = diffusion.timestep_mags_ddv
    save_and_plot(timestep_coss_ddv, timestep_mags_ddv, output_images_folder, "diffusion_diversity")
    # distilled_coss_ddv, distilled_mags_ddv = distill_information(timestep_coss_ddv, timestep_mags_ddv)
    print("ddv done")
    timestep_coss_cdv = diffusion.timestep_coss_cdv
    timestep_mags_cdv = diffusion.timestep_mags_cdv
    save_and_plot(timestep_coss_cdv, timestep_mags_cdv, output_images_folder, "classification_diversity")
    # distilled_coss_cdv, distilled_mags_cdv = distill_information(timestep_coss_cdv, timestep_mags_cdv)
    print("cdv done")


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
    if dist.get_rank() == 0:
        logger.log(f"created {len(all_images) * args.batch_size} samples")
        np.savez(checkpoint, np.stack(all_images), np.stack(all_labels))

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    label_arr = np.concatenate(all_labels, axis=0)
    label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(output_images_folder, f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        np.savez(out_path, arr, label_arr)
        os.remove(checkpoint)

    dist.barrier()
    logger.log("sampling complete")


def save_and_plot(timestep_coss, timestep_mags, folder, name):
    save_folder = os.path.join(folder, name)
    os.makedirs(save_folder, exist_ok=True)
    n = len(timestep_coss)
    m = len(timestep_mags)
    print(n)
    print(m)
    assert n == m
    for i in range(n):
        timestep_coss[i] = timestep_coss[i].cpu().numpy()
        timestep_mags[i] = timestep_mags[i].cpu().numpy()
    # save matrix first
    distilled_coss, distilled_mags = distill_information(timestep_coss, timestep_mags)
    timestep_coss = np.stack(timestep_coss)
    timestep_mags = np.stack(timestep_mags)

    # save to file
    matrix_file = os.path.join(save_folder, "coss_mags.npz")
    np.savez(matrix_file, timestep_coss, timestep_mags)

    # plot
    file_coss = os.path.join(save_folder, "cossine_similarity.png")
    x_axis = np.arange(0, n)
    plt.plot(x_axis, distilled_coss)
    plt.xlabel("timesteps")
    plt.ylabel("percentage of conflict samples")
    plt.savefig(file_coss)
    plt.close()

    file_mags = os.path.join(save_folder, "magnitude_similarity.png")
    plt.plot(x_axis, distilled_mags)
    plt.xlabel("timesteps")
    plt.ylabel("average magnitude similarity (closer to 0 means more conflict)")
    plt.savefig(file_mags)
    plt.close()




def distill_information(timestep_coss, timestep_mags):
    n = len(timestep_mags)
    m = len(timestep_coss)
    assert m == n
    distilled_coss = [] # number of negative /number of >=0
    distilled_mags = [] # average magnitudes
    for i in range(n):
        neg_coss_idx = timestep_coss[i] < 0.0
        neg_coss = timestep_coss[i][neg_coss_idx]
        neg_mags = timestep_mags[i][neg_coss_idx]

        no_samples = timestep_coss[i].shape[0]
        no_neg = neg_coss.shape[0]
        if no_neg == 0:
            avg_mags = 0
        else:
            avg_mags = np.mean(neg_mags)
        rate_neg = no_neg/no_samples * 100
        distilled_coss.append(rate_neg)
        distilled_mags.append(avg_mags)
    return distilled_coss, distilled_mags
    pass

def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=50000,
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
    ngpus = th.cuda.device_count()
    hfai.multiprocessing.spawn(main, args=(), nprocs=ngpus, bind_numa=False)
