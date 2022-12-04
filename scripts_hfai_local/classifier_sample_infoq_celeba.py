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
from guided_diffusion_hfai import dist_util, logger
from guided_diffusion_hfai.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    create_classifier_infodiff,
    add_dict_to_argparser,
    args_to_dict,
)
import datetime
from PIL import Image


def main():
    args = create_argparser().parse_args()
    num_cat = args.cat_num
    cat_dim = args.cat_dim
    num_con = args.con_num
    total_cat_dim = num_cat * cat_dim
    output_channels = num_con * 2 + cat_dim * num_cat


    save_folder = os.path.join(
        args.logdir,
        "logs",
    )

    logger.configure(save_folder, rank=0)

    output_images_folder = os.path.join(args.logdir, "reference")
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
    classifier = create_classifier_infodiff(out_channels=output_channels,
                                            **args_to_dict(args, classifier_defaults().keys()))
    classifier.load_state_dict(
        dist_util.load_state_dict(args.classifier_path, map_location="cpu")
    )
    classifier.to(dist_util.dev())
    if args.classifier_use_fp16:
        classifier.convert_to_fp16()
    classifier.eval()

    def cond_fn(x, t, y=None, con=None):
        assert y is not None
        # assert con is not None
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = classifier(x_in, t)
            # log_probs = 0.0
            selected = 0.0
            for cat_index in range(num_cat):
                log_probs = F.log_softmax(logits[:, cat_dim * cat_index : cat_dim * (cat_index + 1)], dim=-1)
                selected += log_probs[range(len(log_probs)), y[:, cat_index].view(-1)]
            if num_con != 0:
                q_mu = logits[:, total_cat_dim: (total_cat_dim + num_con)]
                q_var = th.exp(logits[:, (total_cat_dim + num_con):])
                # con_selected = nll_loss(con, q_mu, q_var)
                con_selected = 0.0
            else:
                con_selected = 0.0
            selected = selected + con_selected
            return th.autograd.grad(selected.sum(), x_in)[0] * args.classifier_scale

    def model_fn(x, t, y=None, con=None):
        assert y is not None
        # assert con is not None
        return model(x, t, y if args.class_cond else None)

    logger.log("Looking for previous file")
    checkpoint = os.path.join(output_images_folder, "samples_last.npz")
    final_file = os.path.join(output_images_folder, f"samples_{args.num_samples}x{args.image_size}x{args.image_size}x3.npz")
    if os.path.isfile(final_file):
        dist.barrier()
        logger.log("sampling complete")
        return
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
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        classes = th.randint(
            low=0, high=cat_dim, size=(args.batch_size, num_cat), device=dist_util.dev()
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

        gathered_samples = [sample]
        batch_images = [sample.cpu().numpy() for sample in gathered_samples]
        all_images.extend(batch_images)
        gathered_labels = [classes]
        batch_labels = [labels.cpu().numpy() for labels in gathered_labels]
        all_labels.extend(batch_labels)

        logger.log(f"created {len(all_images) * args.batch_size} samples")
        np.savez(checkpoint, np.stack(all_images), np.stack(all_labels))

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    label_arr = np.concatenate(all_labels, axis=0)
    label_arr = label_arr[: args.num_samples]

    shape_str = "x".join([str(x) for x in arr.shape])
    out_path = os.path.join(output_images_folder, f"samples_{shape_str}.npz")
    logger.log(f"saving to {out_path}")
    np.savez(out_path, arr, label_arr)
    os.remove(checkpoint)

    dist.barrier()
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
        logdir="",
        cat_num=10,
        cat_dim=10,
        con_num=0
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser



if __name__ == "__main__":
    main()
