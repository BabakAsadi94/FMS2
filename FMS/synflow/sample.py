import os

import torch
from absl import app, flags
from torchdiffeq import odeint
from torchdyn.core import NeuralODE
from torchvision.utils import save_image
from tqdm import trange

from torchcfm.models.unet.unet_sdm_CrackSDM_v3_encoder import UNetModel
from FMS.synflow.image_datasets import load_data_sample

FLAGS = flags.FLAGS

# ── model ────────────────────────────────────────────────────────────────────
flags.DEFINE_integer("num_channel", 128, "base channel of UNet")
flags.DEFINE_string("model_dir", "./outputs/synflow_crack500_256", "checkpoint directory")
flags.DEFINE_string("model_name", "icfm", "flow matching variant used during training")
flags.DEFINE_string("model_type", "ema_model", "which weight to load: ema_model | net_model")
flags.DEFINE_integer("ckpt_step", 90000, "checkpoint step to load")
flags.DEFINE_bool("parallel", False, "wrap model in DataParallel")

# ── data ─────────────────────────────────────────────────────────────────────
flags.DEFINE_string("data_dir", "./data/CRACK500O/annotations/testing", "folder of mask PNGs to condition on")
flags.DEFINE_integer("num_classes", 2, "number of semantic classes (must match training)")
flags.DEFINE_bool("class_cond", True, "enable mask-conditional sampling")
flags.DEFINE_integer("sample_resolution", 256, "spatial resolution of generated images")

# ── sampling ─────────────────────────────────────────────────────────────────
flags.DEFINE_string("save_dir", "./outputs/samples", "where to write generated PNGs")
flags.DEFINE_integer("num_images", 400, "total number of images to generate")
flags.DEFINE_integer("batch_size", 8, "images per forward pass")
flags.DEFINE_float("omega", 0.4, "CFG guidance strength (0 = no guidance)")
flags.DEFINE_integer("num_steps", 200, "number of ODE integration steps")
flags.DEFINE_string(
    "integration_method", "euler",
    "ODE solver: euler | dopri5 | rk4 | midpoint | heun"
)
flags.DEFINE_float("rtol", 1e-5, "relative tolerance (adaptive solvers only)")
flags.DEFINE_float("atol", 1e-5, "absolute tolerance (adaptive solvers only)")

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def preprocess_input(data):
    """Convert raw label dict → one-hot float conditioning tensor on device."""
    data["label"] = data["label"].to(device, dtype=torch.long, non_blocking=True)
    label_map = data["label"]
    bs, _, h, w = label_map.size()
    semantics = (
        torch.zeros(bs, FLAGS.num_classes, h, w, device=device, dtype=torch.float32)
        .scatter_(1, label_map, 1.0)
    )
    return {"y": semantics.contiguous()}


def _make_cond_wrapper(model, y, omega):
    """Return a torchdyn/torchdiffeq-compatible wrapper with fixed conditioning."""
    class _CondModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model
            self.y = y
            self.omega = omega

        def forward(self, t, x, *args, **kwargs):
            ts = t
            if ts.dim() == 0:
                ts = ts.repeat(x.shape[0])
            elif ts.dim() == 1 and ts.shape[0] == 1:
                ts = ts.repeat(x.shape[0])
            ts = ts * 1000.0

            out = self.model(x=x, timesteps=ts, y=self.y)
            if self.omega != 0:
                unc = self.model(x=x, timesteps=ts, y=torch.zeros_like(self.y))
                out = out + self.omega * (out - unc)
            return out

    return _CondModel()


def generate_samples(model, mask, sample_resolution, num_steps, integration_method, rtol, atol):
    """Run the ODE from noise to image, conditioned on *mask*. Returns (B,3,H,W) in [-1,1]."""
    model.eval()
    cond_model = _make_cond_wrapper(model, mask, FLAGS.omega)

    B = mask.shape[0]
    noise = torch.randn(B, 3, sample_resolution, sample_resolution, device=device)
    t_span = torch.linspace(0, 1, num_steps, device=device)

    method = integration_method.lower()
    with torch.no_grad():
        if method == "euler":
            node = NeuralODE(cond_model, solver="euler", sensitivity="adjoint")
            traj = node.trajectory(noise, t_span=t_span)
        elif method == "midpoint":
            node = NeuralODE(cond_model, solver="midpoint", sensitivity="adjoint")
            traj = node.trajectory(noise, t_span=t_span)
        elif method == "heun":
            node = NeuralODE(cond_model, solver="heun", sensitivity="adjoint")
            traj = node.trajectory(noise, t_span=t_span)
        elif method in ("dopri5", "rk4"):
            traj = odeint(cond_model, noise, t_span, rtol=rtol, atol=atol, method=method)
        else:
            raise ValueError(f"Unknown integration_method: {integration_method!r}")

    model.train()
    return traj[-1].view(B, 3, sample_resolution, sample_resolution).clip(-1, 1)


def sample(_argv):
    datalooper = load_data_sample(
        data_dir=FLAGS.data_dir,
        batch_size=FLAGS.batch_size,
        image_size=FLAGS.sample_resolution,
        deterministic=True,
        random_crop=False,
        random_flip=False,
    )

    net_model = UNetModel(
        image_size=FLAGS.sample_resolution,
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

    if FLAGS.parallel:
        net_model = torch.nn.DataParallel(net_model)

    ckpt_path = os.path.join(FLAGS.model_dir, f"{FLAGS.model_name}_{FLAGS.ckpt_step}.pt")
    print(f"Loading {FLAGS.model_type} from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    net_model.load_state_dict(ckpt[FLAGS.model_type], strict=True)

    os.makedirs(FLAGS.save_dir, exist_ok=True)

    for _ in trange(0, FLAGS.num_images, FLAGS.batch_size, desc="sampling"):
        _, cond = next(datalooper)
        labels = preprocess_input(cond)

        imgs = generate_samples(
            model=net_model,
            mask=labels["y"],
            sample_resolution=FLAGS.sample_resolution,
            num_steps=FLAGS.num_steps,
            integration_method=FLAGS.integration_method,
            rtol=FLAGS.rtol,
            atol=FLAGS.atol,
        )

        for j in range(imgs.shape[0]):
            stem = os.path.splitext(os.path.basename(cond["path"][j]))[0]
            save_image(
                imgs[j],
                os.path.join(FLAGS.save_dir, stem + ".png"),
                normalize=True,
                value_range=(-1, 1),
            )


if __name__ == "__main__":
    app.run(sample)
