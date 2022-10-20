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
    add_dict_to_argparser,
    args_to_dict,
)
import datetime
from PIL import Image
from guided_diffusion_hfai.image_datasets import load_dataset_MNIST_nosampler


def main():
    args = create_argparser().parse_args()

    # dist_util.setup_dist()

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
    classifier = create_classifier(**args_to_dict(args, classifier_defaults().keys()))
    classifier.load_state_dict(
        dist_util.load_state_dict(args.classifier_path, map_location="cpu")
    )
    classifier.to(dist_util.dev())
    if args.classifier_use_fp16:
        classifier.convert_to_fp16()
    classifier.eval()

    def cond_fn(x, t, y=None, x_clean=None):
        assert x_clean is not None
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = classifier(x_in, t)
            logits_clean = classifier(x_clean, t)
            log_probs = F.log_softmax(logits, dim=-1) * logits_clean
            selected = log_probs.sum(dim=-1)
            return th.autograd.grad(selected.sum(), x_in)[0] * args.classifier_scale

    def model_fn(x, t, y=None, x_clean=None):
        assert y is not None
        return model(x, t, y if args.class_cond else None)

    logger.log("Looking for previous file")
    checkpoint = os.path.join(output_images_folder, "samples_last.npz")
    if os.path.isfile(checkpoint):
        npzfile = np.load(checkpoint)
        all_images = list(npzfile['arr_0'])
        all_labels = list(npzfile['arr_1'])
        all_cleans = list(npzfile['arr_3'])
    else:
        all_images = []
        all_labels = []
        all_cleans = []
    logger.log("sampling...")

    data = load_dataset_MNIST_nosampler(
        train=True, batch_size=args.batch_size, class_cond=True
    )

    if args.image_size == 28:
        img_channels = 1
        num_class = 10
    else:
        img_channels = 3
        num_class = NUM_CLASSES
    while len(all_images) * args.batch_size < args.num_samples:
        batch, extra = next(data)
        classes = extra["y"].to(dist_util.dev())
        batch = batch.to(dist_util.dev())
        model_kwargs = {}
        # classes = th.randint(
        #     low=0, high=num_class, size=(args.batch_size,), device=dist_util.dev()
        # )
        model_kwargs["y"] = classes
        model_kwargs["x_clean"] = batch
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
        # dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        batch_images = [sample.cpu().numpy() for sample in gathered_samples]
        all_images.extend(batch_images)
        gathered_labels = [classes]
        batch_labels = [labels.cpu().numpy() for labels in gathered_labels]
        all_labels.extend(batch_labels)

        gathered_batches = [batch]
        clean_images = [batch.cpu().numpy() for batch in gathered_batches]
        all_cleans.extend(clean_images)

        logger.log(f"created {len(all_images) * args.batch_size} samples")
        np.savez(checkpoint, np.stack(all_images), np.stack(all_labels), np.stack(all_cleans))

        #     for i in range(len(batch_images)):
        #         # print(batch_images[0].shape)
        #         # print(len(batch_images))
        #         # exit(0)
        #         for j in range(len(batch_images[i])):
        #             im = Image.fromarray(batch_images[i][j], "RGB")
        #             im.save(os.path.join(output_images_folder, "%d_image%d.png"%(batch_labels[i][j], count_image)))
        #             count_image += 1

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    label_arr = np.concatenate(all_labels, axis=0)
    label_arr = label_arr[: args.num_samples]
    clean_i = np.concatenate(all_cleans, axis=0)
    clean_i = clean_i[: args.num_samples]

    shape_str = "x".join([str(x) for x in arr.shape])
    out_path = os.path.join(output_images_folder, f"samples_{shape_str}.npz")
    logger.log(f"saving to {out_path}")
    np.savez(out_path, arr, label_arr, clean_i)
    os.remove(checkpoint)

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
