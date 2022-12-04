"""
Like image_sample.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.
"""

import argparse
import os

import numpy as np
import torch
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
    create_model_and_diffusion_det,
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
    model, diffusion = create_model_and_diffusion_det(
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

    def cond_fn(x, t, y=None, con=None, pnoise=None):
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

    def model_fn(x, t, y=None, con=None, pnoise=None):
        assert y is not None
        # assert con is not None
        return model(x, t, y if args.class_cond else None)


    all_images = []
    all_labels = []
    logger.log("sampling...")
    if args.image_size == 28:
        img_channels = 1
        num_class = 10
    else:
        img_channels = 3
        num_class = NUM_CLASSES

    fixnoise_folder = os.path.join(args.logdir, "fixnoise")
    os.makedirs(fixnoise_folder, exist_ok=True)
    init_noise_path = os.path.join(fixnoise_folder, "init_noise.pt")
    sample_noise_path = os.path.join(fixnoise_folder, "sample_noise.pt")
    class_path = os.path.join(fixnoise_folder, "class.pt")
    if os.path.isfile(init_noise_path):
        init_noise = torch.load(init_noise_path)
        print("load init")
    else:
        init_noise = th.randn((25, img_channels, args.image_size, args.image_size))
        torch.save(init_noise, init_noise_path)
        print("sample init")
    init_noise = init_noise.to(dist_util.dev())

    if os.path.isfile(sample_noise_path):
        sample_noise = torch.load(sample_noise_path)
        print("load sample")
    else:
        sample_noise = th.randn((25, 300, img_channels, args.image_size, args.image_size))
        torch.save(sample_noise, sample_noise_path)
        print("sample sample")
    sample_noise = sample_noise.to(dist_util.dev())

    if os.path.isfile(class_path):
        classes = torch.load(class_path)
        print("load class")
    else:
        classes = th.randint(
            low=0, high=cat_dim, size=(args.batch_size, num_cat)
        )
        torch.save(classes, class_path)
        print("sample class")
    classes = classes.to(dist_util.dev())

    #__________________________________________________________________________________________________
    # while len(all_images) * args.batch_size < args.num_samples:
    model_kwargs = {}

    model_kwargs["y"] = classes
    model_kwargs["pnoise"] = sample_noise
    sample_fn = (
        diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
    )
    sample = sample_fn(
        model_fn,
        (25, img_channels, args.image_size, args.image_size),
        clip_denoised=args.clip_denoised,
        model_kwargs=model_kwargs,
        cond_fn=cond_fn,
        device=dist_util.dev(),
        noise=init_noise
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
    #_________________________________________________________________________________________________________________

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: 25]
    fixnoise_image_folder = os.path.join(fixnoise_folder, "images")
    os.makedirs(fixnoise_image_folder, exist_ok=True)
    for i in range(25):
        im = Image.fromarray(arr[i], "RGB")
        img_name = "image"
        im.save(os.path.join(fixnoise_image_folder, f"{img_name}_{i}.png"))

    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=25,
        batch_size=25,
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
