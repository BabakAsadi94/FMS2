import copy
import os

import pandas as pd
import torch
import torch.nn.functional as F
from absl import app, flags
from tqdm import trange

from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
    TargetConditionalFlowMatcher,
    VariancePreservingConditionalFlowMatcher,
)
from torchcfm.models.unet.unet import UNetModelWrapper

from FMS.synflow.mask_dataset import load_data
from FMS.synflow.utils import ema, generate_samples_cond_cfg_semantic

FLAGS = flags.FLAGS

# ── model ────────────────────────────────────────────────────────────────────
flags.DEFINE_string("model", "icfm", "flow matching variant: icfm | otcfm | fm | si")
flags.DEFINE_integer("num_channel", 128, "base channel count of UNet")

# ── data ─────────────────────────────────────────────────────────────────────
flags.DEFINE_string("data_dir", "./data/CRACK500O", "root directory of mask images")
flags.DEFINE_string("dataset_csv", "./data/CRACK500O/CRACK500O.csv", "CSV with image_name and class columns")

# ── training ─────────────────────────────────────────────────────────────────
flags.DEFINE_float("lr", 2e-5, "peak learning rate")
flags.DEFINE_float("grad_clip", 1.0, "gradient norm clip")
flags.DEFINE_integer("total_steps", 80001, "total optimiser steps")
flags.DEFINE_bool("conditional", False, "enable class-conditional generation")
flags.DEFINE_integer("warmup", 5000, "linear warm-up steps")
flags.DEFINE_integer("batch_size", 16, "batch size")
flags.DEFINE_integer("num_workers", 4, "DataLoader worker count")
flags.DEFINE_float("omega", 0.4, "CFG guidance strength")
flags.DEFINE_bool("cfg", False, "train with CFG dropout (10 % probability)")
flags.DEFINE_bool("parallel", False, "DataParallel multi-GPU training")
flags.DEFINE_float("ema_decay", 0.9999, "EMA decay rate")

# ── checkpoint ───────────────────────────────────────────────────────────────
flags.DEFINE_string("output_dir", "./outputs/mask_generator", "checkpoint and sample directory")
flags.DEFINE_integer("ckpt_step", 0, "resume from this step (0 = train from scratch)")
flags.DEFINE_integer("save_step", 10000, "checkpoint / sample frequency")

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def train(argv):
    # torch.backends.cudnn.benchmark = False

    _cls = pd.read_csv(FLAGS.dataset_csv)["class"]
    num_classes = int(_cls.max()) + 1
    print(
        f"lr={FLAGS.lr}  steps={FLAGS.total_steps}  "
        f"ema_decay={FLAGS.ema_decay}  save_step={FLAGS.save_step}  "
        f"num_classes={num_classes} (max_label={int(_cls.max())}, unique={_cls.nunique()}, "
        f"from {FLAGS.dataset_csv})"
    )

    datalooper = load_data(
        dataset_csv=FLAGS.dataset_csv,
        data_dir=FLAGS.data_dir,
        batch_size=FLAGS.batch_size,
        image_size=256,
        deterministic=False,
    )

    if not FLAGS.conditional:
        raise NotImplementedError("Only conditional (class-conditional) training is supported.")

    net_model = UNetModelWrapper(
        dim=(1, 256, 256),
        num_res_blocks=2,
        num_channels=FLAGS.num_channel,
        channel_mult=(1, 2, 2, 4),
        num_heads=4,
        num_head_channels=32,
        attention_resolutions="32,16,8",
        dropout=0.1,
        class_cond=True,
        num_classes=num_classes,
    ).to(device)

    ema_model = copy.deepcopy(net_model)

    if FLAGS.parallel:
        net_model = torch.nn.DataParallel(net_model)
        ema_model = torch.nn.DataParallel(ema_model)

    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
    optim = torch.optim.Adam(net_model.parameters(), lr=FLAGS.lr)
    warmup_sched = LinearLR(optim, start_factor=1e-8, total_iters=FLAGS.warmup)
    cosine_sched = CosineAnnealingLR(
        optim, T_max=FLAGS.total_steps - FLAGS.warmup, eta_min=FLAGS.lr * 0.1
    )
    sched = SequentialLR(optim, schedulers=[warmup_sched, cosine_sched],
                         milestones=[FLAGS.warmup])

    start_step = -1
    if FLAGS.ckpt_step > 0:
        ckpt_path = os.path.join(FLAGS.output_dir, f"{FLAGS.model}_{FLAGS.ckpt_step}.pt")
        ckpt = torch.load(ckpt_path, map_location=device)
        net_model.load_state_dict(ckpt["net_model"], strict=True)
        ema_model.load_state_dict(ckpt["ema_model"], strict=True)
        optim.load_state_dict(ckpt["optim"])
        for st in optim.state.values():
            for k, v in st.items():
                if torch.is_tensor(v):
                    st[k] = v.to(device)
        if "sched" in ckpt:
            try:
                sched.load_state_dict(ckpt["sched"])
            except Exception as e:
                print(f"Warning: could not restore scheduler state: {e}")
        start_step = int(ckpt.get("step", 0))
        print(f"Resumed from step {start_step} ({ckpt_path})")

    n_params = sum(p.numel() for p in net_model.parameters())
    print(f"Model parameters: {n_params / 1e6:.2f} M")

    sigma = 0.0
    if FLAGS.model == "otcfm":
        FM = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
    elif FLAGS.model == "icfm":
        FM = ConditionalFlowMatcher(sigma=sigma)
    elif FLAGS.model == "fm":
        FM = TargetConditionalFlowMatcher(sigma=sigma)
    elif FLAGS.model == "si":
        FM = VariancePreservingConditionalFlowMatcher(sigma=sigma)
    else:
        raise ValueError(f"Unknown model: {FLAGS.model}")

    savedir = FLAGS.output_dir + "/"
    os.makedirs(savedir, exist_ok=True)

    with trange(start_step + 1, FLAGS.total_steps + 1, dynamic_ncols=True) as pbar:
        for step in pbar:
            optim.zero_grad()

            x1, y1 = next(datalooper)
            x1, y1 = x1.to(device), y1.to(device)
            x0 = torch.randn_like(x1)

            t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)

            if FLAGS.cfg and torch.rand(1).item() < 0.1:
                y1 = None

            vt = net_model(t=t, x=xt, y=y1)
            loss = F.mse_loss(vt, ut)

            if step % 50 == 0:
                print(f"  step {step}  loss={loss.item():.6f}"
                      f"  lr={optim.param_groups[0]['lr']:.2e}")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(net_model.parameters(), FLAGS.grad_clip)
            optim.step()
            sched.step()
            ema(net_model, ema_model, FLAGS.ema_decay)

            if step % FLAGS.save_step == 0:
                generate_samples_cond_cfg_semantic(
                    net_model, FLAGS.parallel, savedir, step,
                    net_="normal", conditional=FLAGS.conditional,
                    num_classes=num_classes, omega=FLAGS.omega,
                )
                generate_samples_cond_cfg_semantic(
                    ema_model, FLAGS.parallel, savedir, step,
                    net_="ema", conditional=FLAGS.conditional,
                    num_classes=num_classes, omega=FLAGS.omega,
                )
                torch.save(
                    {
                        "net_model": net_model.state_dict(),
                        "ema_model": ema_model.state_dict(),
                        "sched": sched.state_dict(),
                        "optim": optim.state_dict(),
                        "step": step,
                    },
                    os.path.join(savedir, f"{FLAGS.model}_{step}.pt"),
                )


if __name__ == "__main__":
    app.run(train)
