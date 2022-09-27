import os
import itertools
import numpy as np

from PIL import Image
from tqdm import tqdm
from scipy import linalg
from os import path as osp

import torch
import torch.nn as nn

from torch.nn import functional as F
from torchvision.datasets import ImageFolder
from torchvision.models import vgg19

from .gaussian_diffusion import _extract_into_tensor


# =======================================================
# Functions
# =======================================================


def load_from_DDP_model(state_dict):

    new_state_dict = {}
    for k, v in state_dict.items():
        if k[:7] == 'module.':
            k = k[7:]
        new_state_dict[k] = v
    return new_state_dict


# =======================================================
# Gradient Extraction Functions
# =======================================================


@torch.enable_grad()
def clean_class_cond_fn(x_t, y, classifier,
                        s, use_logits):
    '''
    Computes the classifier gradients for the guidance

    :param x_t: clean instance
    :param y: target
    :param classifier: classification model
    :param s: scaling classifier gradients parameter
    :param use_logits: compute the loss over the logits
    '''
    
    x_in = x_t.detach().requires_grad_(True)
    logits = classifier(x_in)

    y = y.to(logits.device).float()
    # Select the target logits,
    # for those of target 1, we take the logits as they are (sigmoid(logits) = p(y=1 | x))
    # for those of target 0, we take the negative of the logits (sigmoid(-logits) = p(y=0 | x))
    selected = y * logits - (1 - y) * logits
    if use_logits:
        selected = -selected
    else:
        selected = -F.logsigmoid(selected)

    selected = selected * s
    grads = torch.autograd.grad(selected.sum(), x_in)[0]

    return grads



@torch.enable_grad()
def clean_multiclass_cond_fn(x_t, y, classifier,
                             s, use_logits):
    
    x_in = x_t.detach().requires_grad_(True)
    selected = classifier(x_in)

    # Select the target logits
    if not use_logits:
        selected = F.log_softmax(selected, dim=1)
    selected = -selected[range(len(y)), y]
    selected = selected * s
    grads = torch.autograd.grad(selected.sum(), x_in)[0]

    return grads


@torch.enable_grad()
def dist_cond_fn(x_tau, z_t, x_t, alpha_t,
                 l1_loss, l2_loss,
                 l_perc):

    '''
    Computes the distance loss between x_t, z_t and x_tau
    :x_tau: initial image
    :z_t: current noisy instance
    :x_t: current clean instance
    :alpha_t: time dependant constant
    '''

    z_in = z_t.detach().requires_grad_(True)
    x_in = x_t.detach().requires_grad_(True)

    m1 = l1_loss * torch.norm(z_in - x_tau, p=1, dim=1).sum() if l1_loss != 0 else 0
    m2 = l2_loss * torch.norm(z_in - x_tau, p=2, dim=1).sum() if l2_loss != 0 else 0
    mv = l_perc(x_in, x_tau) if l_perc is not None else 0
    
    if isinstance(m1 + m2 + mv, int):
        return 0

    if isinstance(m1 + m2, int):
        grads = 0
    else:
        grads = torch.autograd.grad(m1 + m2, z_in)[0]

    if isinstance(mv, int):
        return grads
    else:
        return grads + torch.autograd.grad(mv, x_in)[0] / alpha_t


# =======================================================
# Sampling Function
# =======================================================


def get_DiME_iterative_sampling(use_sampling=False):
    '''
    Returns DiME's main algorithm to construct counterfactuals.
    The returned function computes x_t in a recursive way.
    Easy way to set the optional parameters into the sampling
    function such as the use_sampling flag.

    :param use_sampling: use mu + sigma * N(0,1) when computing
     the next iteration when estimating x_t
    '''
    @torch.no_grad()
    def p_sample_loop(diffusion,
                      model,
                      shape,
                      num_timesteps,
                      img,
                      t,
                      z_t=None,
                      clip_denoised=True,
                      model_kwargs=None,
                      device=None,
                      class_grad_fn=None,
                      class_grad_kwargs=None,
                      dist_grad_fn=None,
                      dist_grad_kargs=None,
                      x_t_sampling=True,
                      is_x_t_sampling=False,
                      guided_iterations=9999999):

        '''
        :param :
        :param diffusion: diffusion algorithm
        :param model: DDPM model
        :param num_timesteps: tau, or the depth of the noise chain
        :param img: instance to be explained
        :param t: time variable
        :param z_t: noisy instance. If z_t is instantiated then the model
                    will denoise z_t
        :param clip_denoised: clip the noised data to [-1, 1]
        :param model_kwargs: useful when the model is conditioned
        :param device: torch device
        :param class_grad_fn: class function to compute the gradients of the classifier
                              has at least an input, x_t.
        :param class_grad_kwargs: Additional arguments for class_grad_fn
        :param dist_grad_fn: Similar as class_grad_fn, uses z_t, x_t, x_tau, and alpha_t as inputs
        :param dist_grad_kwargs: Additional args fot dist_grad_fn
        :param x_t_sampling: use sampling when computing x_t
        :param is_x_t_sampling: useful flag to distinguish when x_t is been generated
        :param guided_iterations: Early stop the guided iterations
        '''

        x_t = img.clone()
        z_t = diffusion.q_sample(img, t) if z_t is None else z_t

        x_t_steps = []
        z_t_steps = []
        indices = list(range(num_timesteps))[::-1]

        for jdx, i in enumerate(indices):

            t = torch.tensor([i] * shape[0], device=device)
            x_t_steps.append(x_t.detach())
            z_t_steps.append(z_t.detach())

            # out is a dictionary with the following (self-explanatory) keys:
            # 'mean', 'variance', 'log_variance'
            out = diffusion.p_mean_variance(
                model,
                z_t,
                t,
                clip_denoised=clip_denoised,
                denoised_fn=None,
                model_kwargs=model_kwargs,
            )

            # extract sqrtalphacum
            alpha_t = _extract_into_tensor(diffusion.sqrt_alphas_cumprod,
                                       t, shape)

            nonzero_mask = (
                (t != 0).float().view(-1, *([1] * (len(shape) - 1)))
            )  # no noise when t == 0

            grads = 0
            
            if (class_grad_fn is not None) and (guided_iterations > jdx):
                grads = grads + class_grad_fn(x_t=x_t,
                                              **class_grad_kwargs) / alpha_t

            if (dist_grad_fn is not None) and (guided_iterations > jdx):
                grads = grads + dist_grad_fn(z_t=z_t,
                                             x_tau=img,
                                             x_t=x_t,
                                             alpha_t=alpha_t,
                                             **dist_grad_kargs)

            out["mean"] = (
                out["mean"].float() -
                out["variance"] * grads
            )

            if not x_t_sampling:
                z_t = out["mean"]

            else:
                z_t = (
                    out["mean"] +
                    nonzero_mask * torch.exp(0.5 * out["log_variance"]) * torch.randn_like(img)
                )

            # produce x_t in a brute force manner
            if (num_timesteps - (jdx + 1) > 0) and (class_grad_fn is not None) and (dist_grad_fn is not None) and (guided_iterations > jdx):
                x_t = p_sample_loop(
                    diffusion=diffusion,
                    model=model,
                    model_kwargs=model_kwargs,
                    shape=shape,
                    num_timesteps=num_timesteps - (jdx + 1),
                    img=img,
                    t=None,
                    z_t=z_t,
                    clip_denoised=True,
                    device=device,
                    x_t_sampling=use_sampling,
                    is_x_t_sampling=True,
                )[0]

        return z_t, x_t_steps, z_t_steps

    return p_sample_loop


# =======================================================
# Classes
# =======================================================


class ChunkedDataset:
    def __init__(self, dataset, chunk=0, num_chunks=1):
        self.dataset = dataset
        self.indexes = [i for i in range(len(dataset)) if (i % num_chunks) == chunk]

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        i = [self.indexes[idx]]
        i += list(self.dataset[i[0]])
        return i


class ImageSaver():
    def __init__(self, output_path, exp_name, extention='.jpg'):
        self.output_path = output_path
        self.exp_name = exp_name
        self.idx = 0
        self.extention = extention
        with open('imagenet1000_clsidx_to_labels.txt') as f: 
            self.dic = eval(f.read()) 
        self.construct_directory()

    def construct_directory(self):

        os.makedirs(osp.join(self.output_path, 'Original', 'Correct'), exist_ok=True)
        os.makedirs(osp.join(self.output_path, 'Original', 'Incorrect'), exist_ok=True)

        for clst, cf, subf in itertools.product(['CC', 'IC'],
                                                ['CCF', 'ICF'],
                                                ['CF', 'Noise', 'Info', 'SM']):
            os.makedirs(osp.join(self.output_path, 'Results',
                                 self.exp_name, clst,
                                 cf, subf),
                        exist_ok=True)

    def __call__(self, imgs, cfs, noises, target, label,
                 pred, pred_cf, bkl, l_1, indexes=None, masks=None):

        for idx in range(len(imgs)):
            current_idx = indexes[idx].item() if indexes is not None else idx + self.idx
            mask = None if masks is None else masks[idx]
            self.save_img(img=imgs[idx],
                          cf=cfs[idx],
                          noise=noises[idx],
                          idx=current_idx,
                          target=target[idx].item(),
                          label=label[idx].item(),
                          pred=pred[idx].item(),
                          pred_cf=pred_cf[idx].item(),
                          bkl=bkl[idx].item(),
                          l_1=l_1[idx].item(),
                          mask=mask)

        self.idx += len(imgs)

    @staticmethod
    def select_folder(label, target, pred, pred_cf):
        folder = osp.join('CC' if label == pred else 'IC',
                          'CCF' if target == pred_cf else 'ICF')
        return folder

    @staticmethod
    def preprocess(img):
        '''
        remove last dimension if it is 1
        '''
        if img.shape[2] > 1:
            return img
        else:
            return np.squeeze(img, 2)

    def save_img(self, img, cf, noise, idx, target, label,
                 pred, pred_cf, bkl, l_1, mask):
        folder = self.select_folder(label, target, pred, pred_cf)
        output_path = osp.join(self.output_path, 'Results',
                               self.exp_name, folder)
        img_name = f'{idx}'.zfill(7)
        orig_path = osp.join(self.output_path, 'Original',
                             'Correct' if label == pred else 'Incorrect',
                             img_name + self.extention)

        if mask is None:
            l0 = np.abs(img.astype('float') - cf.astype('float'))
            l0 = l0.sum(2, keepdims=True)
            l0 = 255 * l0 / l0.max()
            l0 = np.concatenate([l0] * img.shape[2], axis=2).astype('uint8')
            l0 = Image.fromarray(self.preprocess(l0))
            l0.save(osp.join(output_path, 'SM', img_name + self.extention))
        else:
            mask = mask.astype('uint8') * 255
            mask = Image.fromarray(mask)
            mask.save(osp.join(output_path, 'SM', img_name + self.extention))

        img = Image.fromarray(self.preprocess(img))
        img.save(orig_path)

        cf = Image.fromarray(self.preprocess(cf))
        cf.save(osp.join(output_path, 'CF', img_name + self.extention))

        noise = Image.fromarray(self.preprocess(noise))
        noise.save(osp.join(output_path, 'Noise', img_name + self.extention))


        to_write = (f'label: {label}' +
                    f'\npred: {pred}' +
                    f'\ntarget: {target}' +
                    f'\ncf pred: {pred_cf}' +
                    f'\nBKL: {bkl}' +
                    f'\nl_1: {l_1}')
        with open(osp.join(output_path, 'Info', img_name + '.txt'), 'w') as f:
            f.write(to_write)


class Normalizer(nn.Module):
    def __init__(self, classifier):
        super().__init__()
        self.classifier = classifier
        self.register_buffer('mu', torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1))
        self.register_buffer('sigma', torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1))

    def forward(self, x):
        x = (torch.clamp(x, -1, 1) + 1) / 2
        x = (x - self.mu) / self.sigma
        return self.classifier(x)


class SingleLabel(ImageFolder):
    def __init__(self, query_label, **kwargs):
        super().__init__(**kwargs)
        self.query_label = query_label

        # remove those instances that do no have the
        # query label

        old_len = len(self)
        instances = [self.targets[i] == query_label
                     for i in range(old_len)]
        self.samples = [self.samples[i]
                        for i in range(old_len) if instances[i]]
        self.targets = [self.targets[i]
                        for i in range(old_len) if instances[i]]
        self.imgs = [self.imgs[i]
                     for i in range(old_len) if instances[i]]


class SlowSingleLabel():
    def __init__(self, query_label, dataset, maxlen=float('inf')):
        self.dataset = dataset
        self.indexes = []
        if isinstance(dataset, ImageFolder):
            self.indexes = np.where(np.array(dataset.targets) == query_label)[0]
            self.indexes = self.indexes[:maxlen]
        else:
            print('Slow route. This may take some time!')
            if query_label != -1:
                for idx, (_, l) in enumerate(tqdm(dataset)):

                    l = l['y'] if isinstance(l, dict) else l
                    if l == query_label:
                        self.indexes.append(idx)

                    if len(self.indexes) == maxlen:
                        break
            else:
                self.indexes = list(range(min(maxlen, len(dataset))))

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        return self.dataset[self.indexes[idx]]


class PerceptualLoss(nn.Module):
    def __init__(self, layer, c):
        super().__init__()
        self.c = c
        vgg19_model = vgg19(pretrained=True)
        vgg19_model = nn.Sequential(*list(vgg19_model.features.children())[:layer])
        self.model = Normalizer(vgg19_model)
        self.model.eval()

    def forward(self, x0, x1):
        B = x0.size(0)

        l = F.mse_loss(self.model(x0).view(B, -1), self.model(x1).view(B, -1),
                       reduction='none').mean(dim=1)
        return self.c * l.sum()


class extra_data_saver():
    def __init__(self, output_path, exp_name):
        self.idx = 0
        self.exp_name = exp_name

    def __call__(self, x_ts, indexes=None):
        n_images = x_ts[0].size(0)
        n_steps = len(x_ts)

        for i in range(n_images):
            current_idx = indexes[i].item() if indexes is not None else i + self.idx
            os.makedirs(osp.join(self.output_path, self.exp_name, str(current_idx).zfill(6)), exist_ok=True)

            for j in range(n_steps):
                cf = x_ts[j][i, ...]

                # renormalize the image
                cf = ((cf + 1) * 127.5).clamp(0, 255).to(torch.uint8)
                cf = cf.permute(1, 2, 0)
                cf = cf.contiguous().cpu().numpy()
                cf = Image.fromarray(cf)
                cf.save(osp.join(self.output_path, self.exp_name, str(current_idx).zfill(6), str(j).zfill(4) + '.jpg'))

        self.idx += n_images


class X_T_Saver(extra_data_saver):
    def __init__(self, output_path, exp_path, extention='.jpg'):
        super().__init__(output_path, exp_path)
        self.output_path = osp.join(output_path, 'x_t')


class Z_T_Saver(extra_data_saver):
    def __init__(self, output_path, exp_path, extention='.jpg'):
        super().__init__(output_path, exp_path)
        self.output_path = osp.join(output_path, 'z_t')


class Mask_Saver(extra_data_saver):
    def __init__(self, output_path, exp_path, extention='.jpg'):
        super().__init__(output_path, exp_path)
        self.output_path = osp.join(output_path, 'masks')

    def __call__(self, masks, indexes=None):
        '''
        Masks are non-binarized 
        '''
        n_images = masks[0].size(0)
        n_steps = len(masks)

        for i in range(n_images):
            current_idx = indexes[i].item() if indexes is not None else i + self.idx
            os.makedirs(osp.join(self.output_path, self.exp_name, str(current_idx).zfill(6)), exist_ok=True)

            for j in range(n_steps):
                cf = masks[j][i, ...]
                cf = torch.cat((cf, (cf > 0.5).to(cf.dtype)), dim=-1)

                # renormalize the image
                cf = (cf * 255).clamp(0, 255).to(torch.uint8)
                cf = cf.permute(1, 2, 0)
                cf = cf.squeeze(dim=-1)
                cf = cf.contiguous().cpu().numpy()
                cf = Image.fromarray(cf)
                cf.save(osp.join(self.output_path, self.exp_name, str(current_idx).zfill(6), str(j).zfill(4) + self.extention))

        self.idx += n_images
