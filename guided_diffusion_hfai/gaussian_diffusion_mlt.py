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


class GaussianDiffusionMLT(SpacedDiffusion):
    """
    NO consider about batchsize
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
        super(GaussianDiffusionMLT, self).__init__(use_timesteps=use_timesteps, betas=betas,
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

        noise = th.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        if cond_fn is not None:
            out["mean"] = self.condition_mean(
                cond_fn, out, x, t, model_kwargs=model_kwargs
            )
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def condition_mean_mtl(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
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

        g_cls_shape = gradient_cls.shape
        g_gen_shape = gradient_gen.shape
        assert g_cls_shape == g_gen_shape

        new_gradient_cls = self.project_conflict(gradient_cls.clone(), gradient_gen.clone(), g_cls_shape) # check if gradient_cls change / changed -> using clone -> does not change
        new_gradient_gen = self.project_conflict(gradient_gen.clone(), gradient_cls.clone(), g_gen_shape) # check if gradient_gen change / changed -> using clone -> does not change

        new_mean = (
            p_mean_var["mean"].float() - gradient_gen.float() + new_gradient_gen.float() + new_gradient_cls.float()
        )
        del gradient_cls, gradient_gen, new_gradient_gen, new_gradient_cls
        return new_mean

    def project_conflict(self, grad1, grad2, shape):
        new_grad1 = torch.flatten(grad1)
        new_grad2 = torch.flatten(grad2)

        g_1_g_2 = torch.dot(new_grad1, new_grad2)

        if g_1_g_2 < 0: # later could change to partial conflict
            new_grad1 -= ((g_1_g_2) * new_grad2/ (new_grad2.norm() ** 2))
        new_grad1 = new_grad1.view(shape)
        return new_grad1

    def condition_mean(self, cond_fn, *args, **kwargs):
        return self.condition_mean_mtl(self._wrap_model(cond_fn), *args, **kwargs)


class GaussianDiffusionMLT2(SpacedDiffusion):
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
        super(GaussianDiffusionMLT2, self).__init__(use_timesteps=use_timesteps, betas=betas,
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
        noise = th.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        if cond_fn is not None:
            out["mean"] = self.condition_mean(
                cond_fn, out, x, t, model_kwargs=model_kwargs
            )
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def condition_mean_mtl(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
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

        g_cls_shape = gradient_cls.shape
        g_gen_shape = gradient_gen.shape
        assert g_cls_shape == g_gen_shape

        new_gradient_cls = self.project_conflict(gradient_cls.clone(), gradient_gen.clone(), g_cls_shape) # check if gradient_cls change / changed -> using clone -> does not change
        new_gradient_gen = self.project_conflict(gradient_gen.clone(), gradient_cls.clone(), g_gen_shape) # check if gradient_gen change / changed -> using clone -> does not change

        new_mean = (
            p_mean_var["mean"].float() - gradient_gen.float() + new_gradient_gen.float() + new_gradient_cls.float()
        )
        del gradient_cls, gradient_gen, new_gradient_gen, new_gradient_cls
        return new_mean

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

    def condition_mean(self, cond_fn, *args, **kwargs):
        return self.condition_mean_mtl(self._wrap_model(cond_fn), *args, **kwargs)

class GaussianDiffusionMLT_DiversityGrad(GaussianDiffusionMLT):
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
        super(GaussianDiffusionMLT_DiversityGrad, self).__init__(use_timesteps=use_timesteps, betas=betas,
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

        noise = th.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        gradient_gen = torch.zeros_like(out["mean"])
        gradient_cls = torch.zeros_like(out["mean"])
        if cond_fn is not None:
            out["mean"], gradient_gen, gradient_cls = self.condition_mean(
                cond_fn, out, x, t, model_kwargs=model_kwargs
            )

        gradient_diversity = nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        (gradient_gen, gradient_cls, gradient_diversity) = self.project_conflict(gradient_gen, gradient_cls,
                                                                                 gradient_diversity, gradient_gen.shape)

        out["mean"] = (_extract_into_tensor(self.posterior_mean_coef2, t, x.shape) * x + gradient_gen).float() + \
                      gradient_cls.float()
        sample = out["mean"] + gradient_diversity.float()
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def condition_mean_mtl(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
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

        g_cls_shape = gradient_cls.shape
        g_gen_shape = gradient_gen.shape
        assert g_cls_shape == g_gen_shape

        new_mean = p_mean_var["mean"]
        return new_mean, gradient_gen, gradient_cls

    def project_conflict(self, grad1, grad2, grad3, shape):
        grad1 = torch.flatten(grad1)
        grad2 = torch.flatten(grad2)
        grad3 = torch.flatten(grad3)
        grads = [grad1, grad2, grad3]
        pc_grad, num_grads = copy.deepcopy(grads), 3
        for g_i in pc_grad:
            random.shuffle(grads)
            for g_j in grads:
                g_i_g_j = torch.dot(g_i, g_j)
                if g_i_g_j < 0:
                    g_i -= (g_i_g_j) * g_j / (g_j.norm() ** 2)
        for i in range(num_grads):
            pc_grad[i] = pc_grad[i].view(shape)
        return pc_grad


class GaussianDiffusionMLT_Analyse(GaussianDiffusionMLT):
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
    timestep_coss_dc = []  # diffusion classification
    timestep_mags_dc = []  # diff - class

    timestep_coss_ddv = []  # diff - div
    timestep_mags_ddv = []  # diff - dv

    timestep_coss_cdv = []  # class = div
    timestep_mags_cdv = []  # class - div


    def __int__(self,
        use_timesteps,
        *,
        betas,
        model_mean_type,
        model_var_type,
        loss_type,
        rescale_timesteps=False,
    ):

        super(GaussianDiffusionMLT_Analyse, self).__init__(use_timesteps=use_timesteps, betas=betas,
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
        noise = th.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0

        gradient_cls = torch.zeros_like(x)
        if cond_fn is not None:
            out["mean"], gradient_cls = self.condition_mean(
                cond_fn, out, x, t, model_kwargs=model_kwargs
            )
        gradient_gen = _extract_into_tensor(self.posterior_mean_coef1, t, x.shape) * out['pred_xstart']
        gradient_div = nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise

        sample = out["mean"] + gradient_div

        coss_dc = self.direction_conflict(gradient_gen.clone(), gradient_cls.clone())
        self.timestep_coss_dc.append(coss_dc)
        self.timestep_mags_dc.append(self.magnitude_conflict(gradient_gen.clone(), gradient_cls.clone()))

        coss_ddiv = self.direction_conflict(gradient_gen.clone(), gradient_div.clone())
        self.timestep_coss_ddv.append(coss_ddiv)
        self.timestep_mags_ddv.append(self.magnitude_conflict(gradient_gen.clone(), gradient_div.clone()))

        coss_cdv = self.direction_conflict(gradient_cls.clone(), gradient_div.clone())
        self.timestep_coss_cdv.append(coss_cdv)
        self.timestep_mags_cdv.append(self.magnitude_conflict(gradient_cls.clone(), gradient_div.clone()))

        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def condition_mean_mtl(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.

        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        gradient = cond_fn(x, self._scale_timesteps(t), **model_kwargs)
        new_mean = (
                p_mean_var["mean"].float() + p_mean_var["variance"] * gradient.float()
        )
        return new_mean, p_mean_var["variance"] * gradient.float()

    def direction_conflict(self, grad1, grad2):
        grad1_norm = nn.functional.normalize(torch.flatten(grad1, start_dim=1), dim=1)
        grad2_norm = nn.functional.normalize(torch.flatten(grad2, start_dim=1), dim=1)
        return (grad1_norm * grad2_norm).sum(dim=1)
        pass

    def magnitude_conflict(self, grad1, grad2):
        fgrad1 = torch.flatten(grad1, start_dim=1)
        fgrad2 = torch.flatten(grad2, start_dim=1)
        fgrad1_norm = fgrad1.norm(dim=1)
        fgrad2_norm = fgrad2.norm(dim=1)
        magnitude = (2 * fgrad1_norm * fgrad2_norm)/(fgrad1_norm**2 + fgrad2_norm**2)
        return magnitude
        pass


class GaussianDiffusionMLTCDiv(GaussianDiffusionMLT2):
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
        super(GaussianDiffusionMLTCDiv, self).__init__(use_timesteps=use_timesteps, betas=betas,
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
            out["mean"], gradient_div = self.condition_mean(
                cond_fn, out, x, t, nonzero_mask, model_kwargs=model_kwargs
            )
        else:
            noise = th.randn_like(x)
            gradient_div = nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        sample = out["mean"] + gradient_div
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

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

        #diffusion - classification conflict
        new_gradient_cls = self.project_conflict(gradient_cls.clone(), gradient_gen.clone(), g_cls_shape) # check if gradient_cls change / changed -> using clone -> does not change
        new_gradient_gen = self.project_conflict(gradient_gen.clone(), gradient_cls.clone(), g_gen_shape) # check if gradient_gen change / changed -> using clone -> does not change

        # classification - diversity conflict
        final_gradient_cls = self.project_conflict(new_gradient_cls.clone(), gradient_div.clone(), g_cls_shape)
        final_gradient_div = self.project_conflict(gradient_div.clone(), new_gradient_cls.clone(), g_cls_shape)

        new_mean = (
            p_mean_var["mean"].float() - gradient_gen.float() + new_gradient_gen.float() + final_gradient_cls.float()
        )
        del gradient_cls, gradient_gen, new_gradient_gen, new_gradient_cls, final_gradient_cls
        return new_mean, final_gradient_div


class GaussianDiffusionMLTChannelConflict(GaussianDiffusionMLT2):
    def __int__(self,
        use_timesteps,
        *,
        betas,
        model_mean_type,
        model_var_type,
        loss_type,
        rescale_timesteps=False,
    ):
        super(GaussianDiffusionMLTChannelConflict, self).__init__(use_timesteps=use_timesteps, betas=betas,
                                                                    model_mean_type=model_mean_type,
                                                                    model_var_type=model_var_type,
                                                                    loss_type=loss_type,
                                                                    rescale_timesteps=rescale_timesteps)

    def project_conflict(self, grad1, grad2, shape):
        """project conflict considering channel conflicts"""
        new_grad1 = torch.flatten(grad1, start_dim=2)
        new_grad2 = torch.flatten(grad2, start_dim=2)

        # g1 * g2 --------------- (batchsize, channels)
        g_1_g_2 = torch.sum(new_grad1 * new_grad2, dim=2)
        g_1_g_2 = torch.clamp(g_1_g_2, max=0.0)

        # ||g2||^2 ----------------- (batchsize, channels)
        norm_g2 = new_grad2.norm(dim=2) ** 2
        if torch.any(norm_g2 == 0.0):
            return new_grad1.view(shape)

        # (g1 * g2)/||g2||^2 ------------------- (batchsize, channels)
        g12_o_normg2 = g_1_g_2 / norm_g2
        g12_o_normg2 = torch.unsqueeze(g12_o_normg2, dim=2)
        # why zero has problem?
        # g1
        new_grad1 -= ((g12_o_normg2) * new_grad2)
        new_grad1 = new_grad1.view(shape)
        return new_grad1


class GaussianDiffusionMLTDDiv(GaussianDiffusionMLT2):
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
        super(GaussianDiffusionMLTDDiv, self).__init__(use_timesteps=use_timesteps, betas=betas,
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
        noise = th.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        if cond_fn is not None:
            out["mean"] = self.condition_mean(
                cond_fn, out, x, t, model_kwargs=model_kwargs
            )
        else:
            gradient_diversity = nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
            gradient_gen = _extract_into_tensor(self.posterior_mean_coef1, t, x.shape) * out['pred_xstart']
            new_gradient_gen = self.project_conflict(gradient_gen.clone(), gradient_diversity.clone(), gradient_gen.shape)
            new_gradient_diversity = self.project_conflict(gradient_diversity.clone(), gradient_gen.clone(), gradient_gen.shape)
            sample = out["mean"] - gradient_gen + new_gradient_gen + new_gradient_diversity
            del gradient_gen, new_gradient_gen, new_gradient_diversity, gradient_diversity
            return {"sample": sample, "pred_xstart": out["pred_xstart"]}

        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

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
