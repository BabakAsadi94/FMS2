import copy
import os

import torch
import torch.nn.functional as F
from absl import app, flags
from torchdyn.core import NeuralODE
from torchvision.utils import save_image
from tqdm import trange

from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
    TargetConditionalFlowMatcher,
    VariancePreservingConditionalFlowMatcher,
)
from torchcfm.models.unet.unet_sdm_CrackSDM_v3_encoder import UNetModel

from FMS.synflow.image_datasets import load_data
from FMS.synflow.utils import ema, generate_samples_cond_sfm

FLAGS = flags.FLAGS

# ── model ────────────────────────────────────────────────────────────────────
flags.DEFINE_string("model", "icfm", "flow matching variant: icfm | otcfm | fm | si")
flags.DEFINE_integer("num_channel", 128, "base channel count of UNet")

# ── data ─────────────────────────────────────────────────────────────────────
flags.DEFINE_string("data_type", "crack500", "dataset mode (crack500 | angiography | …)")
flags.DEFINE_string("data_dir", "./data/crack500", "path to dataset root")
flags.DEFINE_integer("img_size", 256, "spatial resolution for training")
flags.DEFINE_integer("num_classes", 2, "number of semantic classes (background + crack)")
flags.DEFINE_bool("class_cond", True, "pass semantic mask as conditioning")

# ── training ─────────────────────────────────────────────────────────────────
flags.DEFINE_float("lr", 5e-5, "peak learning rate")
flags.DEFINE_float("grad_clip", 1.0, "gradient norm clip")
flags.DEFINE_integer("total_steps", 200001, "total optimiser steps")
flags.DEFINE_bool("conditional", False, "enable mask-conditional generation")
flags.DEFINE_integer("warmup", 10000, "linear warm-up steps")
flags.DEFINE_integer("batch_size", 4, "per-GPU batch size")
flags.DEFINE_integer("num_workers", 4, "DataLoader worker count")
flags.DEFINE_float("omega", 0.0, "CFG guidance strength (0 = no guidance)")
flags.DEFINE_float("drop_rate", 0.1, "probability of zeroing out the mask (CFG training)")
flags.DEFINE_bool("cfg", False, "train with classifier-free guidance dropout")
flags.DEFINE_bool("parallel", False, "DataParallel multi-GPU training")
flags.DEFINE_float("ema_decay", 0.9999, "EMA decay rate")

# ── checkpoint ───────────────────────────────────────────────────────────────
flags.DEFINE_string("output_dir", "./logs/synflow", "directory for checkpoints and samples")
flags.DEFINE_integer("ckpt_step", 0, "resume from this step (0 = train from scratch)")
flags.DEFINE_integer("save_step", 10000, "checkpoint / sample frequency")

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def preprocess_input(data):
    """Convert raw label dict to one-hot float conditioning tensor."""
    data["label"] = data["label"].to(device, dtype=torch.long, non_blocking=True)
    label_map = data["label"]                    # (N, 1, H, W)
    bs, _, h, w = label_map.size()
    input_semantics = (
        torch.zeros(bs, FLAGS.num_classes, h, w, device=device, dtype=torch.float32)
        .scatter_(1, label_map, 1.0)
    )
    if FLAGS.drop_rate > 0.0:
        mask = (torch.rand((bs, 1, 1, 1), device=device) > FLAGS.drop_rate).float()
        input_semantics = input_semantics * mask
    cond = {k: v for k, v in data.items() if k not in ("label", "instance", "path", "label_ori")}
    cond["y"] = input_semantics.contiguous()
    return cond


def train(argv):
    print(
        f"lr={FLAGS.lr}  steps={FLAGS.total_steps}  "
        f"ema_decay={FLAGS.ema_decay}  save_step={FLAGS.save_step}"
    )

    datalooper = load_data(
        dataset_mode=FLAGS.data_type,
        data_dir=FLAGS.data_dir,
        batch_size=FLAGS.batch_size,
        image_size=FLAGS.img_size,
        class_cond=FLAGS.class_cond,
        is_train=True,
    )

    if not FLAGS.conditional:
        raise NotImplementedError("Only conditional (mask-guided) training is supported.")

    net_model = UNetModel(
        image_size=FLAGS.img_size,
        in_channels=3,
        model_channels=256,
        out_channels=3,
        num_res_blocks=2,
        attention_resolutions=(32, 16, 8),
        dropout=0,
        channel_mult=(1, 1, 2, 2, 4, 4),
        num_classes=FLAGS.num_classes,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=4,
        num_head_channels=64,
        num_heads_upsample=-1,
        use_scale_shift_norm=True,
        resblock_updown=True,
        use_new_attention_order=False,
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

    # ── optionally resume ────────────────────────────────────────────────────
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

    # ── flow matcher ─────────────────────────────────────────────────────────
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

    # ── training loop ────────────────────────────────────────────────────────
    with trange(start_step + 1, FLAGS.total_steps + 1, dynamic_ncols=True) as pbar:
        for step in pbar:
            optim.zero_grad()

            x1, y1 = next(datalooper)
            y1 = preprocess_input(y1)
            x1 = x1.to(device)
            x0 = torch.randn_like(x1)

            t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
            t, xt, ut = t.to(device), xt.to(device), ut.to(device)

            vt = net_model(x=xt, timesteps=t * 1000, y=y1["y"])
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
                save_image(
                    y1["y"],
                    os.path.join(savedir, f"mask_{step}.png"),
                    nrow=14, normalize=True, value_range=(-1, 1),
                )
                generate_samples_cond_sfm(
                    net_model, FLAGS.parallel, savedir, step,
                    net_="normal", conditional=FLAGS.conditional,
                    omega=FLAGS.omega, mask=y1["y"],
                    sample_num=FLAGS.batch_size, sample_resolution=FLAGS.img_size,
                )
                generate_samples_cond_sfm(
                    ema_model, FLAGS.parallel, savedir, step,
                    net_="ema", conditional=FLAGS.conditional,
                    omega=FLAGS.omega, mask=y1["y"],
                    sample_num=FLAGS.batch_size, sample_resolution=FLAGS.img_size,
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
