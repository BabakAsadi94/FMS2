import copy
import os

import torch
from torchdyn.core import NeuralODE
from torchvision.utils import save_image

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay + source_dict[key].data * (1 - decay)
        )


def infiniteloop(dataloader):
    while True:
        for x, y in iter(dataloader):
            yield x


def infiniteloop_cond(dataloader):
    while True:
        for x, y in iter(dataloader):
            yield x, y


def generate_samples(model, parallel, savedir, step, net_="normal"):
    model.eval()
    model_ = copy.deepcopy(model)
    if parallel:
        model_ = model_.module.to(device)
    node_ = NeuralODE(model_, solver="euler", sensitivity="adjoint")
    with torch.no_grad():
        traj = node_.trajectory(
            torch.randn(64, 3, 32, 32, device=device),
            t_span=torch.linspace(0, 1, 100, device=device),
        )
        traj = traj[-1, :].view([-1, 3, 32, 32]).clip(-1, 1)
        traj = traj / 2 + 0.5
    save_image(traj, savedir + f"{net_}_generated_FM_images_step_{step}.png", nrow=8)
    model.train()


def generate_samples_cond_sfm(
    model, parallel, savedir, step,
    net_="normal", conditional=False, omega=0.8,
    mask=None, sample_num=64, sample_resolution=32,
):
    model.eval()
    model_ = copy.deepcopy(model)
    if parallel:
        model_ = model_.module.to(device)

    y = mask if (conditional and mask is not None) else None

    if y is not None:
        class CondModelWrapper(torch.nn.Module):
            def __init__(self, model, y, omega):
                super().__init__()
                self.model = model
                self.y = y
                self.omega = omega

            def forward(self, t, x, *args, **kwargs):
                timesteps = t
                if timesteps.dim() == 0:
                    timesteps = timesteps.repeat(x.shape[0])
                elif timesteps.dim() == 1 and timesteps.shape[0] == 1:
                    timesteps = timesteps.repeat(x.shape[0])
                timesteps = timesteps * 1000.0

                if self.omega == 0:
                    return self.model(x=x, timesteps=timesteps, y=self.y)

                eps = self.model(x=x, timesteps=timesteps, y=self.y)
                unc_eps = self.model(x=x, timesteps=timesteps, y=torch.zeros_like(self.y))
                return eps + self.omega * (eps - unc_eps)

        node_ = NeuralODE(CondModelWrapper(model_, y, omega), solver="euler", sensitivity="adjoint")
    else:
        node_ = NeuralODE(model_, solver="euler", sensitivity="adjoint")

    with torch.no_grad():
        traj = node_.trajectory(
            torch.randn(sample_num, 3, sample_resolution, sample_resolution, device=device),
            t_span=torch.linspace(0, 1, 100, device=device),
        )
        traj = traj[-1, :].view([-1, 3, sample_resolution, sample_resolution]).clip(-1, 1)
        traj = traj / 2 + 0.5
    save_image(traj, savedir + f"{net_}_generated_FM_images_step_{step}.png", nrow=8)
    model.train()


def generate_samples_cond_cfg_semantic(
    model, parallel, savedir, step,
    net_="normal", conditional=False, num_classes=None, omega=0.8,
    apply_threshold=True,
):
    """Generate 4 grayscale mask samples for checkpoint preview during mask-generator training."""
    model.eval()
    model_ = copy.deepcopy(model)
    if parallel:
        model_ = model_.module.to(device)

    y = None
    if conditional and num_classes:
        y = torch.randint(0, num_classes, (4,), device=device)

    if y is not None:
        class CondModelWrapper(torch.nn.Module):
            def __init__(self, m, labels, w):
                super().__init__()
                self.m = m
                self.labels = labels
                self.w = w

            def forward(self, t, x, *args, **kwargs):
                if self.w == 0:
                    return self.m(t, x, y=self.labels)
                eps = self.m(t, x, y=self.labels)
                unc = self.m(t, x, y=None)
                return eps + self.w * (eps - unc)

        node_ = NeuralODE(CondModelWrapper(model_, y, omega), solver="euler", sensitivity="adjoint")
    else:
        node_ = NeuralODE(model_, solver="euler", sensitivity="adjoint")

    with torch.no_grad():
        traj = node_.trajectory(
            torch.randn(4, 1, 256, 256, device=device),
            t_span=torch.linspace(0, 1, 100, device=device),
        )
        traj = traj[-1].view(-1, 1, 256, 256).clip(-1, 1)

        if apply_threshold:
            traj_binary = torch.where(traj > 0, torch.ones_like(traj), -torch.ones_like(traj))
            save_image(traj / 2 + 0.5, savedir + f"{net_}_mask_continuous_step_{step}.png", nrow=2)
            save_image((traj_binary + 1) / 2, savedir + f"{net_}_mask_binary_step_{step}.png", nrow=2)
            torch.save(traj_binary, savedir + f"{net_}_mask_binary_raw_step_{step}.pt")
        else:
            save_image(traj / 2 + 0.5, savedir + f"{net_}_mask_step_{step}.png", nrow=2)

    model.train()
