"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py

Docstrings have been added, as well as DDIM sampling and a new collection of beta schedules.
"""
import copy

# import torch

from guided_diffusion_hfai.gaussian_diffusion import *
from guided_diffusion_hfai.respace import *
import random
import torch
import torch.nn as nn
import torch.nn.functional
from guided_diffusion_hfai.gaussian_diffusion_mlt import GaussianDiffusionMLT2
import math


class GaussianDiffusionMLTCDiv_Analyse(GaussianDiffusionMLT2):
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """
    def __int__(self,
        use_timesteps,
        *,
        betas,
        model_mean_type,
        model_var_type,
        loss_type,
        rescale_timesteps=False,
    ):
        super(GaussianDiffusionMLTCDiv_Analyse, self).__init__(use_timesteps=use_timesteps, betas=betas,
                                                   model_mean_type=model_mean_type,
                                                   model_var_type=model_var_type,
                                                   loss_type=loss_type,
                                                   rescale_timesteps=rescale_timesteps)

    def p_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
    ):
        """
        Batch consider

        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        self.t = t

        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0

        if cond_fn is not None:
            out["mean"], gradient_div, log_before, log_after = self.condition_mean(
                cond_fn, out, x, t, nonzero_mask, model_kwargs=model_kwargs
            )
        else:
            noise = th.randn_like(x)
            gradient_div = nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
            log_before = None
            log_after = None
        sample = out["mean"] + gradient_div
        return {"sample": sample, "pred_xstart": out["pred_xstart"], "before": log_before, "after": log_after}

    def p_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None
        a_before = {
            "magnitude": {
                "cls": [],
                "diff": [],
                "div": []
            },

            "angle": {
                "cls_div": [],
                "cls_diff": [],
                "div_diff": []
            }
        }

        a_after = {
            "magnitude": {
                "cls": [],
                "diff": [],
                "div": []
            },

            "angle": {
                "cls_div": [],
                "cls_diff": [],
                "div_diff": []
            }
        }


        count_iter = 0
        for sample in self.p_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
        ):
            final = sample
            if (count_iter % 5 == 0 or count_iter == 249) and (count_iter > 224) :
                analyse_before_step = sample["before"]
                analyse_after_step = sample["after"]
                self.integrate_analyse(a_before, analyse_before_step)
                self.integrate_analyse(a_after, analyse_after_step)
            count_iter += 1
        a_after = self.stack_analyse(analyse_dict=a_after)
        a_before = self.stack_analyse(analyse_dict=a_before)
        return final["sample"], a_before, a_after

    def integrate_analyse(self, analyse_all:dict, analyse_step: dict):
        # append magnitude
        for key in analyse_all.keys():
            for key_in in analyse_all[key].keys():
                analyse_all[key][key_in].append(analyse_step[key][key_in])
        return analyse_all

    def stack_analyse(self, analyse_dict: dict):
        for key in analyse_dict.keys():
            for key_in in analyse_dict[key].keys():
                analyse_dict[key][key_in] = torch.stack(analyse_dict[key][key_in], dim=-1)
        return analyse_dict



    def condition_mean_mtl(self, cond_fn, p_mean_var, x, t, nonzero_mask, model_kwargs=None):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.

        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        gradient = cond_fn(x, self._scale_timesteps(t), **model_kwargs)
        gradient_cls = p_mean_var["variance"] * gradient.float()
        gradient_gen = _extract_into_tensor(self.posterior_mean_coef1, t, x.shape) * p_mean_var['pred_xstart']

        noise = th.randn_like(x)
        gradient_div = nonzero_mask * th.exp(0.5 * p_mean_var["log_variance"]) * noise

        g_cls_shape = gradient_cls.shape
        g_gen_shape = gradient_gen.shape
        assert g_cls_shape == g_gen_shape

        log_before_project = self.get_information_analyse(gradient_cls.clone(), gradient_gen.clone(), gradient_div.clone())

        #diffusion - classification conflict

        new_gradient_cls = self.project_conflict(gradient_cls.clone(), gradient_gen.clone(), g_cls_shape) # check if gradient_cls change / changed -> using clone -> does not change
        new_gradient_gen = self.project_conflict(gradient_gen.clone(), gradient_cls.clone(), g_gen_shape) # check if gradient_gen change / changed -> using clone -> does not change

        # classification - diversity conflict
        final_gradient_cls = self.project_conflict(new_gradient_cls.clone(), gradient_div.clone(), g_cls_shape)
        final_gradient_div = self.project_conflict(gradient_div.clone(), gradient_cls.clone(), g_cls_shape)

        log_after_project  = self.get_information_analyse(final_gradient_cls.clone(), new_gradient_gen.clone(),
                                                          final_gradient_div.clone())

        new_mean = (
            p_mean_var["mean"].float() - gradient_gen.float() + new_gradient_gen.float() + final_gradient_cls.float()
        )
        del gradient_cls, gradient_gen, new_gradient_gen, new_gradient_cls, final_gradient_cls
        return new_mean, final_gradient_div, log_before_project, log_after_project

    def get_information_analyse(self, grad_cls, grad_diff, grad_div, grad_divnow=None):
        grad_cls_flatten = torch.flatten(grad_cls, start_dim=1)
        grad_diff_flatten = torch.flatten(grad_diff, start_dim=1)
        grad_div_flatten = torch.flatten(grad_div, start_dim=1)
        if grad_divnow is not None:
            grad_divnow_flatten = torch.flatten(grad_divnow, start_dim=1)
        else:
            grad_divnow_flatten = torch.zeros_like(grad_div_flatten)

        grad_cls_norm = nn.functional.normalize(grad_cls_flatten, dim=1)
        grad_diff_norm = nn.functional.normalize(grad_diff_flatten, dim=1)
        grad_div_norm = nn.functional.normalize(grad_div_flatten, dim=1)

        # magnitude
        magnitude_cls =  grad_cls_flatten.norm(dim=1)
        magnitude_diff = grad_diff_flatten.norm(dim=1)
        magnitude_div = grad_div_flatten.norm(dim=1)
        magnitude_divnow = grad_divnow_flatten.norm(dim=1)

        # angles

        angle_cls_div = (torch.arccos((grad_cls_norm * grad_div_norm).sum(dim=1)) * 180)/math.pi
        angle_cls_diff = (torch.arccos((grad_cls_norm * grad_diff_norm).sum(dim=1)) * 180)/math.pi
        angle_div_diff = (torch.arccos((grad_diff_norm * grad_div_norm).sum(dim=1))* 180)/math.pi

        output_dict = {
            "magnitude": {
                "cls": magnitude_cls,
                "diff": magnitude_diff,
                "div": magnitude_div,
                "divnow": magnitude_divnow
            },

            "angle": {
                "cls_div": angle_cls_div,
                "cls_diff": angle_cls_diff,
                "div_diff": angle_div_diff
            }
        }
        return output_dict
        pass


class GaussianDiffusion_Analyse(GaussianDiffusionMLTCDiv_Analyse):
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """
    def __int__(self,
        use_timesteps,
        *,
        betas,
        model_mean_type,
        model_var_type,
        loss_type,
        rescale_timesteps=False,
    ):
        super(GaussianDiffusion_Analyse, self).__init__(use_timesteps=use_timesteps, betas=betas,
                                                   model_mean_type=model_mean_type,
                                                   model_var_type=model_var_type,
                                                   loss_type=loss_type,
                                                   rescale_timesteps=rescale_timesteps)

    def p_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
    ):
        """
        Batch consider

        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        self.t = t

        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0

        if cond_fn is not None:
            out["mean"], gradient_div, log_before = self.condition_mean(
                cond_fn, out, x, t, nonzero_mask, model_kwargs=model_kwargs
            )
        else:
            noise = th.randn_like(x)
            gradient_div = nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
            log_before = None
        sample = out["mean"] + gradient_div
        return {"sample": sample, "pred_xstart": out["pred_xstart"], "before": log_before}

    def p_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None
        a_before = {
            "magnitude": {
                "cls": [],
                "diff": [],
                "div": [],
                "diffnow": []
            },

            "angle": {
                "cls_div": [],
                "cls_diff": [],
                "div_diff": []
            }
        }



        count_iter = 0
        for sample in self.p_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
        ):
            final = sample
            if (count_iter % 5 == 0  or count_iter == 249) and (count_iter > 224):
                analyse_before_step = sample["before"]
                self.integrate_analyse(a_before, analyse_before_step)
            count_iter += 1
        a_before = self.stack_analyse(analyse_dict=a_before)
        return final["sample"], a_before


    def condition_mean_mtl(self, cond_fn, p_mean_var, x, t, nonzero_mask, model_kwargs=None):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.

        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        gradient = cond_fn(x, self._scale_timesteps(t), **model_kwargs)
        gradient_cls = p_mean_var["variance"] * gradient.float()
        gradient_gen = _extract_into_tensor(self.posterior_mean_coef1, t, x.shape) * p_mean_var['pred_xstart']

        noise = th.randn_like(x)
        gradient_div = nonzero_mask * th.exp(0.5 * p_mean_var["log_variance"]) * noise

        log_before_project = self.get_information_analyse(gradient_cls.clone(), gradient_gen.clone(), gradient_div.clone())

        new_mean = (
            p_mean_var["mean"].float() + gradient_cls.float()
        )
        del gradient_cls, gradient_gen
        return new_mean, gradient_div, log_before_project


class GaussianDiffusionMLTDDiv_Analyse(GaussianDiffusionMLTCDiv_Analyse):
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """
    def __int__(self,
        use_timesteps,
        *,
        betas,
        model_mean_type,
        model_var_type,
        loss_type,
        rescale_timesteps=False,
    ):
        super(GaussianDiffusionMLTDDiv_Analyse, self).__init__(use_timesteps=use_timesteps, betas=betas,
                                                   model_mean_type=model_mean_type,
                                                   model_var_type=model_var_type,
                                                   loss_type=loss_type,
                                                   rescale_timesteps=rescale_timesteps)

    def p_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
    ):
        """
        Batch consider

        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        self.t = t

        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0

        if cond_fn is not None:
            out["mean"], gradient_div, log_before, log_after = self.condition_mean(
                cond_fn, out, x, t, nonzero_mask, model_kwargs=model_kwargs
            )
        else:
            noise = th.randn_like(x)
            gradient_div = nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
            log_before = None
            log_after = None
        sample = out["mean"] + gradient_div
        return {"sample": sample, "pred_xstart": out["pred_xstart"], "before": log_before, "after": log_after}

    def condition_mean_mtl(self, cond_fn, p_mean_var, x, t, nonzero_mask, model_kwargs=None):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.

        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        gradient = cond_fn(x, self._scale_timesteps(t), **model_kwargs)
        gradient_cls = p_mean_var["variance"] * gradient.float()
        gradient_gen = _extract_into_tensor(self.posterior_mean_coef1, t, x.shape) * p_mean_var['pred_xstart']

        noise = th.randn_like(x)
        gradient_div = nonzero_mask * th.exp(0.5 * p_mean_var["log_variance"]) * noise

        g_cls_shape = gradient_cls.shape
        g_gen_shape = gradient_gen.shape
        assert g_cls_shape == g_gen_shape

        log_before_project = self.get_information_analyse(gradient_cls.clone(), gradient_gen.clone(), gradient_div.clone())

        new_gradient_gen = self.project_conflict(gradient_gen.clone(), gradient_div.clone(), gradient_gen.shape)
        new_gradient_div = self.project_conflict(gradient_div.clone(), gradient_gen.clone(),gradient_gen.shape)

        log_after_project  = self.get_information_analyse(gradient_cls.clone(), new_gradient_gen.clone(),
                                                          new_gradient_div.clone())

        new_mean = (
            p_mean_var["mean"].float() - gradient_gen.float() + new_gradient_gen.float() + gradient_cls.float()
        )
        del gradient_cls, gradient_gen, new_gradient_gen
        return new_mean, new_gradient_div, log_before_project, log_after_project

    def project_conflict(self, grad1, grad2, shape):
        new_grad1 = torch.flatten(grad1, start_dim=1)
        new_grad2 = torch.flatten(grad2, start_dim=1)

        # g1 * g2 --------------- (batchsize,)
        g_1_g_2 = torch.sum(new_grad1 * new_grad2, dim=1)
        g_1_g_2 = torch.clamp(g_1_g_2, max=0.0)

        # ||g2||^2 ----------------- (batchsize,)
        norm_g2 = new_grad2.norm(dim=1) **2
        if torch.any(norm_g2 == 0.0):
            return new_grad1.view(shape)

        # (g1 * g2)/||g2||^2 ------------------- (batchsize,)
        g12_o_normg2 = g_1_g_2/norm_g2
        g12_o_normg2 = torch.unsqueeze(g12_o_normg2, dim=1)
        # why zero has problem?
        # g1
        new_grad1 -= ((g12_o_normg2) * new_grad2)
        new_grad1 = new_grad1.view(shape)
        return new_grad1



def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)
