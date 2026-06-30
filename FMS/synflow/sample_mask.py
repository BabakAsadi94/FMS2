import os

import pandas as pd
import torch
from absl import app, flags
from torchdiffeq import odeint
from torchdyn.core import NeuralODE
from torchvision.utils import save_image
from tqdm import trange

from torchcfm.models.unet.unet import UNetModelWrapper
from FMS.synflow.mask_dataset import load_data

FLAGS = flags.FLAGS

# ── model ────────────────────────────────────────────────────────────────────
flags.DEFINE_integer("num_channel", 128, "base channel of UNet")
flags.DEFINE_string("model_dir", "./outputs/mask_generator", "checkpoint directory")
flags.DEFINE_string("model_name", "icfm", "flow matching variant used during training")
flags.DEFINE_string("model_type", "ema_model", "which weight to load: ema_model | net_model")
flags.DEFINE_integer("ckpt_step", 30000, "checkpoint step to load")
flags.DEFINE_bool("parallel", False, "wrap model in DataParallel")

# ── data ─────────────────────────────────────────────────────────────────────
flags.DEFINE_string("data_dir", "./data/CRACK500O", "root directory of mask images")
flags.DEFINE_string("dataset_csv", "./data/CRACK500O/CRACK500O.csv", "CSV with image_name and class columns")
flags.DEFINE_bool("class_cond", True, "enable class-conditional sampling")

# ── sampling ─────────────────────────────────────────────────────────────────
flags.DEFINE_string("save_dir", "./outputs/mask_samples", "where to write generated PNGs")
flags.DEFINE_integer("num_images", 400, "total number of masks to generate")
flags.DEFINE_integer("batch_size", 8, "masks per forward pass")
flags.DEFINE_float("omega", 0.4, "CFG guidance strength (0 = no guidance)")
flags.DEFINE_integer("num_steps", 200, "number of ODE integration steps")
flags.DEFINE_string(
    "integration_method", "dopri5",
    "ODE solver: euler | dopri5 | rk4 | midpoint"
)
flags.DEFINE_float("rtol", 1e-5, "relative tolerance (adaptive solvers only)")
flags.DEFINE_float("atol", 1e-5, "absolute tolerance (adaptive solvers only)")
flags.DEFINE_string("surfix", "", "optional filename suffix appended before .png")

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def _make_cond_wrapper(model, y, omega):
    class _CondModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model
            self.y = y
            self.omega = omega

        def forward(self, t, x, *args, **kwargs):
            if t.dim() == 0 or t.shape[0] != x.shape[0]:
                t = t.expand(x.shape[0])
            if self.omega == 0:
                return self.model(t, x, y=self.y)
            eps = self.model(t, x, y=self.y)
            unc = self.model(t, x, y=None)
            return eps + self.omega * (eps - unc)

    return _CondModel()


def generate_masks(model, classes, num_steps, integration_method, rtol, atol, apply_threshold=True):
    """Run ODE from noise to binary mask conditioned on integer class labels.
    Returns (B, 1, 256, 256) tensor with values in {-1, 1} if apply_threshold,
    otherwise in [-1, 1].
    """
    model.eval()
    cond_model = _make_cond_wrapper(model, classes, FLAGS.omega)

    B = classes.shape[0]
    noise = torch.randn(B, 1, 256, 256, device=device)
    t_span = torch.linspace(0, 1, num_steps, device=device)

    method = integration_method.lower()
    with torch.no_grad():
        if method == "euler":
            node = NeuralODE(cond_model, solver="euler", sensitivity="adjoint")
            traj = node.trajectory(noise, t_span=t_span)
        elif method == "midpoint":
            node = NeuralODE(cond_model, solver="midpoint", sensitivity="adjoint")
            traj = node.trajectory(noise, t_span=t_span)
        elif method in ("dopri5", "rk4"):
            traj = odeint(cond_model, noise, t_span, rtol=rtol, atol=atol, method=method)
        else:
            raise ValueError(f"Unknown integration_method: {integration_method!r}")

    model.train()
    result = traj[-1].view(B, 1, 256, 256).clip(-1, 1)
    if apply_threshold:
        result = torch.where(result > 0, torch.ones_like(result), -torch.ones_like(result))
    return result


def sample(_argv):
    _cls = pd.read_csv(FLAGS.dataset_csv)["class"]
    num_classes = int(_cls.max()) + 1
    print(f"num_classes={num_classes} (max_label={int(_cls.max())}, unique={_cls.nunique()}, from {FLAGS.dataset_csv})")

    datalooper = load_data(
        dataset_csv=FLAGS.dataset_csv,
        data_dir=FLAGS.data_dir,
        batch_size=FLAGS.batch_size,
        image_size=256,
        deterministic=False,
        only_class=True,
    )

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

    if FLAGS.parallel:
        net_model = torch.nn.DataParallel(net_model)

    ckpt_path = os.path.join(FLAGS.model_dir, f"{FLAGS.model_name}_{FLAGS.ckpt_step}.pt")
    print(f"Loading {FLAGS.model_type} from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    net_model.load_state_dict(ckpt[FLAGS.model_type], strict=True)

    os.makedirs(FLAGS.save_dir, exist_ok=True)

    for _ in trange(0, FLAGS.num_images, FLAGS.batch_size, desc="sampling masks"):
        paths, classes = next(datalooper)
        classes = classes.to(device)

        masks = generate_masks(
            model=net_model,
            classes=classes,
            num_steps=FLAGS.num_steps,
            integration_method=FLAGS.integration_method,
            rtol=FLAGS.rtol,
            atol=FLAGS.atol,
            apply_threshold=True,
        )

        for j in range(masks.shape[0]):
            stem = os.path.splitext(os.path.basename(paths[j]))[0]
            save_image(
                masks[j],
                os.path.join(FLAGS.save_dir, stem + FLAGS.surfix + ".png"),
                normalize=True,
                value_range=(-1, 1),
            )


if __name__ == "__main__":
    app.run(sample)
