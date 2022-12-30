"""
Like image_sample.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.

PCGrad on classificaiton - div + classification - diffusion

analyse on magnitude and angles
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

from guided_diffusion_hfai.script_util_mlt_analyse import create_model_and_diffusion_cdiv_analyse # diffusion - classification conflict + classification - diversity conflict
import datetime
from PIL import Image


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
    model, diffusion = create_model_and_diffusion_cdiv_analyse(
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
    checkpoint_analyse_before = os.path.join(output_images_folder, "analyse_before_last.npz")
    checkpoint_analyse_after = os.path.join(output_images_folder, "analyse_after_last.npz")
    final_file = os.path.join(output_images_folder,
                              f"samples_{args.num_samples}x{args.image_size}x{args.image_size}x3.npz")
    if os.path.isfile(final_file):
        dist.barrier()
        logger.log("sampling complete")
        return
    if os.path.isfile(checkpoint):
        npzfile = np.load(checkpoint)
        all_images = list(npzfile['arr_0'])
        all_labels = list(npzfile['arr_1'])

        npzafter = np.load(checkpoint_analyse_after)
        all_magnitude_cls_after = list(npzafter['arr_0'])
        all_magnitude_div_after = list(npzafter['arr_1'])
        all_magnitude_diff_after = list(npzafter['arr_2'])

        all_angle_cls_div_after = list(npzafter['arr_3'])
        all_angle_cls_diff_after = list(npzafter['arr_4'])
        all_angle_div_diff_after = list(npzafter['arr_5'])

        npzbefore = np.load(checkpoint_analyse_before)
        all_magnitude_cls_before = list(npzbefore['arr_0'])
        all_magnitude_div_before = list(npzbefore['arr_1'])
        all_magnitude_diff_before = list(npzbefore['arr_2'])

        all_angle_cls_div_before = list(npzbefore['arr_3'])
        all_angle_cls_diff_before = list(npzbefore['arr_4'])
        all_angle_div_diff_before = list(npzbefore['arr_5'])

    else:
        all_images = []
        all_labels = []

        all_magnitude_cls_after = []
        all_magnitude_div_after = []
        all_magnitude_diff_after = []

        all_angle_cls_div_after  = []
        all_angle_cls_diff_after = []
        all_angle_div_diff_after = []

        all_magnitude_cls_before = []
        all_magnitude_div_before = []
        all_magnitude_diff_before = []

        all_angle_cls_div_before = []
        all_angle_cls_diff_before = []
        all_angle_div_diff_before = []

    logger.log(f"Number of current images: {len(all_images)}")
    logger.log("sampling...")
    if args.image_size == 28:
        img_channels = 1
        num_class = 10
    else:
        img_channels = 3
        num_class = NUM_CLASSES
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        classes = th.randint(
            low=0, high=num_class, size=(args.batch_size,), device=dist_util.dev()
        )
        model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample, analyse_before, analyse_after = sample_fn(
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

        # after analyse ______________________________________________________-
        batch_magnitude_cls_after = gather_analyse_matrix(analyse_after, "magnitude", "cls")
        all_magnitude_cls_after.extend(batch_magnitude_cls_after)

        batch_magnitude_div_after = gather_analyse_matrix(analyse_after, "magnitude", "div")
        all_magnitude_div_after.extend(batch_magnitude_div_after)

        batch_magnitude_diff_after = gather_analyse_matrix(analyse_after, "magnitude", "diff")
        all_magnitude_diff_after.extend(batch_magnitude_diff_after)

        batch_angle_cls_div_after = gather_analyse_matrix(analyse_after, "angle", "cls_div")
        all_angle_cls_div_after.extend(batch_angle_cls_div_after)

        batch_angle_cls_diff_after = gather_analyse_matrix(analyse_after, "angle", "cls_diff")
        all_angle_cls_diff_after.extend(batch_angle_cls_diff_after)

        batch_angle_div_diff_after = gather_analyse_matrix(analyse_after, "angle", "div_diff")
        all_angle_div_diff_after.extend(batch_angle_div_diff_after)
        list_analyse_after = [all_magnitude_cls_after, all_magnitude_div_after, all_magnitude_diff_after,
                              all_angle_cls_div_after, all_angle_cls_diff_after, all_angle_div_diff_after]

        # before analyse _______________________________________________________

        batch_magnitude_cls_before = gather_analyse_matrix(analyse_before, "magnitude", "cls")
        all_magnitude_cls_before.extend(batch_magnitude_cls_before)

        batch_magnitude_div_before = gather_analyse_matrix(analyse_before, "magnitude", "div")
        all_magnitude_div_before.extend(batch_magnitude_div_before)

        batch_magnitude_diff_before = gather_analyse_matrix(analyse_before, "magnitude", "diff")
        all_magnitude_diff_before.extend(batch_magnitude_diff_before)

        batch_angle_cls_div_before = gather_analyse_matrix(analyse_before, "angle", "cls_div")
        all_angle_cls_div_before.extend(batch_angle_cls_div_before)

        batch_angle_cls_diff_before = gather_analyse_matrix(analyse_before, "angle", "cls_diff")
        all_angle_cls_diff_before.extend(batch_angle_cls_diff_before)

        batch_angle_div_diff_before = gather_analyse_matrix(analyse_before, "angle", "div_diff")
        all_angle_div_diff_before.extend(batch_angle_div_diff_before)
        list_analyse_before = [all_magnitude_cls_before, all_magnitude_div_before, all_magnitude_diff_before,
                               all_angle_cls_div_before, all_angle_cls_diff_before, all_angle_div_diff_before]

        if dist.get_rank() == 0:
            logger.log(f"created {len(all_images) * args.batch_size} samples")
            np.savez(checkpoint, np.stack(all_images), np.stack(all_labels))

            np.savez(checkpoint_analyse_before,
                     np.stack(all_magnitude_cls_before),
                     np.stack(all_magnitude_div_before),
                     np.stack(all_magnitude_diff_before),
                     np.stack(all_angle_cls_div_before),
                     np.stack(all_angle_cls_diff_before),
                     np.stack(all_angle_div_diff_before))

            np.savez(checkpoint_analyse_after,
                     np.stack(all_magnitude_cls_after),
                     np.stack(all_magnitude_div_after),
                     np.stack(all_magnitude_diff_after),
                     np.stack(all_angle_cls_div_after),
                     np.stack(all_angle_cls_diff_after),
                     np.stack(all_angle_div_diff_after))

    analyse_after = concat_analyse(analyse_matrix=list_analyse_after, num_samples= args.num_samples)
    analyse_before = concat_analyse(analyse_matrix=list_analyse_before, num_samples=args.num_samples)

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

        out_analyse_after = os.path.join(output_images_folder, "analyse_after.npz")
        logger.log(f"saving to {out_analyse_after}")
        np.savez(out_analyse_after, analyse_after[0], analyse_after[1], analyse_after[2],
                 analyse_after[3], analyse_after[4], analyse_after[5])
        os.remove(checkpoint_analyse_after)

        out_analyse_before = os.path.join(output_images_folder, "analyse_before.npz")
        logger.log(f"saving to {out_analyse_after}")
        np.savez(out_analyse_before, analyse_before[0], analyse_before[1], analyse_before[2],
                 analyse_before[3], analyse_before[4], analyse_before[5])
        os.remove(checkpoint_analyse_before)


    dist.barrier()
    logger.log("sampling complete")


def gather_analyse_matrix(analyse_dict, key1, key2):
    gather_infos  = [th.zeros_like(analyse_dict[key1][key2]) for _ in range(dist.get_world_size())]
    dist.all_gather(gather_infos, analyse_dict[key1][key2])
    batch_info = [gather_info.cpu().numpy() for gather_info in gather_infos]
    return batch_info

def concat_analyse(analyse_matrix, num_samples):
    n_measures = len(analyse_matrix)
    for i in range(n_measures):
        analyse_matrix[i] = np.concatenate(analyse_matrix[i], axis=0)
        analyse_matrix[i] = analyse_matrix[i][:num_samples]
    return analyse_matrix


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
