import os

import numpy as np
import torch
from absl import app, flags
from PIL import Image
from torchdiffeq import odeint
from torchdyn.core import NeuralODE
from torchvision.utils import save_image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from torchcfm.models.unet.unet_residal import UNetModel

FLAGS = flags.FLAGS

# ── checkpoint ────────────────────────────────────────────────────────────────
flags.DEFINE_string("ckpt_path", "", "direct path to the .pt checkpoint file")
flags.DEFINE_string("model_type", "ema_model", "weight key to load: ema_model | net_model")
flags.DEFINE_bool("parallel", False, "wrap model in DataParallel")
flags.DEFINE_bool("self_cond", False,
                  "checkpoint was trained with self-conditioning (UNet in_channels=6); "
                  "must match the training setting")

# ── data ──────────────────────────────────────────────────────────────────────
flags.DEFINE_string("image_dir", "", "directory of input RGB images")
flags.DEFINE_string("save_dir", "./outputs/segflow_predictions", "directory for output masks")
flags.DEFINE_integer("img_size", 256, "spatial resolution (must match training)")
flags.DEFINE_integer("batch_size", 4, "images per forward pass")

# ── ODE ───────────────────────────────────────────────────────────────────────
flags.DEFINE_integer("num_inference_steps", 10, "ODE steps (euler) or ignored (dopri5)")
flags.DEFINE_string("integration_method", "euler", "ODE solver: euler | dopri5")
flags.DEFINE_float("rtol", 1e-5, "relative tolerance (dopri5 only)")
flags.DEFINE_float("atol", 1e-5, "absolute tolerance (dopri5 only)")

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

_IMG_EXTS = {"jpg", "jpeg", "png", "bmp", "gif"}


def _list_images(directory):
    return [
        os.path.join(directory, f)
        for f in sorted(os.listdir(directory))
        if f.rsplit(".", 1)[-1].lower() in _IMG_EXTS
    ]


class ImageOnlyDataset(Dataset):
    """Loads RGB images from a directory, center-crops to img_size, normalises to [-1,1]."""

    def __init__(self, image_paths, img_size):
        self.image_paths = image_paths
        self.img_size = img_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        pil = Image.open(path).convert("RGB")

        size = self.img_size
        while min(*pil.size) >= 2 * size:
            pil = pil.resize(tuple(x // 2 for x in pil.size), resample=Image.BOX)
        scale = size / min(*pil.size)
        pil = pil.resize(tuple(round(x * scale) for x in pil.size), resample=Image.BICUBIC)
        arr = np.array(pil)
        cy = (arr.shape[0] - size) // 2
        cx = (arr.shape[1] - size) // 2
        arr = arr[cy: cy + size, cx: cx + size]

        tensor = arr.astype(np.float32) / 127.5 - 1.0
        return np.transpose(tensor, [2, 0, 1]), os.path.basename(path)


@torch.no_grad()
def _run_ode(model, x_img):
    """Run ODE from x_img (RGB image) to segmentation mask."""
    B = x_img.shape[0]
    method = FLAGS.integration_method

    # ---- Euler with self-conditioning: manual loop carrying the predicted x1 ----
    if FLAGS.self_cond and method == "euler":
        steps = FLAGS.num_inference_steps
        t_span = torch.linspace(0, 1, steps + 1, device=x_img.device)
        dt = 1.0 / steps
        x_curr = x_img.clone()
        self_cond = torch.zeros_like(x_curr)  # [B, 3, H, W] zero at first step
        for i in range(steps):
            t_b = torch.full((B,), t_span[i].item(), device=x_img.device, dtype=x_curr.dtype)
            v = model(x=torch.cat([x_curr, self_cond], dim=1), timesteps=t_b * 1000.0)
            t_expand = t_b.view(-1, 1, 1, 1)
            self_cond = (x_curr + (1 - t_expand) * v).detach()
            x_curr = x_curr + dt * v
        return x_curr

    class _Wrapper(torch.nn.Module):
        def forward(self, t, x, **kwargs):
            ts = (t if t.dim() > 0 else t.expand(B)) * 1000.0
            # adaptive / stateless solvers cannot carry the self-condition → use zeros
            if FLAGS.self_cond:
                x = torch.cat([x, torch.zeros_like(x)], dim=1)
            return model(x=x, timesteps=ts)

    wrapped = _Wrapper()

    if method == "euler":
        node = NeuralODE(wrapped, solver="euler")
        t_span = torch.linspace(0, 1, FLAGS.num_inference_steps + 1, device=x_img.device)
        traj = node.trajectory(x_img, t_span=t_span)
        return traj[-1]
    elif method == "dopri5":
        t_span = torch.linspace(0, 1, 2, device=x_img.device)
        traj = odeint(wrapped, x_img, t_span, rtol=FLAGS.rtol, atol=FLAGS.atol, method="dopri5")
        return traj[-1]
    else:
        raise ValueError(f"Unknown integration_method {method!r}. Use 'euler' or 'dopri5'.")


def infer(argv):
    if not FLAGS.ckpt_path:
        raise ValueError("--ckpt_path is required")
    if not FLAGS.image_dir:
        raise ValueError("--image_dir is required")

    image_paths = _list_images(FLAGS.image_dir)
    if not image_paths:
        raise ValueError(f"No images found in {FLAGS.image_dir}")
    print(f"Found {len(image_paths)} images in {FLAGS.image_dir}")

    loader = DataLoader(
        ImageOnlyDataset(image_paths, FLAGS.img_size),
        batch_size=FLAGS.batch_size,
        shuffle=False,
        num_workers=1,
        drop_last=False,
    )

    # self-conditioning checkpoints were trained with 6 input channels (xt + self-cond)
    in_channels = 6 if FLAGS.self_cond else 3
    net_model = UNetModel(
        image_size=FLAGS.img_size,
        in_channels=in_channels,
        model_channels=32,
        out_channels=3,
        num_res_blocks=1,
        attention_resolutions=(32, 16, 8),
        dropout=0.0,
        channel_mult=(1, 2, 4),
        use_checkpoint=False,
        use_fp16=False,
        num_heads=4,
        num_head_channels=32,
        num_heads_upsample=-1,
        use_scale_shift_norm=True,
        resblock_updown=True,
        use_new_attention_order=False,
    ).to(device)

    print(f"Loading '{FLAGS.model_type}' from {FLAGS.ckpt_path}  "
          f"(self_cond={FLAGS.self_cond}, in_channels={in_channels})")
    ckpt = torch.load(FLAGS.ckpt_path, map_location=device)
    state_dict = ckpt[FLAGS.model_type] if FLAGS.model_type in ckpt else ckpt
    # strip DataParallel 'module.' prefix if the checkpoint was saved from multi-GPU training
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
    net_model.load_state_dict(state_dict, strict=True)

    if FLAGS.parallel:
        net_model = torch.nn.DataParallel(net_model)

    net_model.eval()
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    print(f"Saving predictions to: {FLAGS.save_dir}")

    total_saved = 0
    for x_img, basenames in tqdm(loader, desc="segflow inference"):
        x_img = x_img.to(device)

        pred = _run_ode(net_model, x_img)

        # pixels < 0 are foreground (crack); invert so crack=black, background=white
        pred_gray   = pred.mean(dim=1, keepdim=True)
        pred_binary = (pred_gray < 0.0).float()
        output      = 1.0 - pred_binary          # background=1 (white), crack=0 (black)

        for i, name in enumerate(basenames):
            stem = os.path.splitext(name)[0]
            save_image(output[i: i + 1], os.path.join(FLAGS.save_dir, f"{stem}.png"))
            total_saved += 1

    print(f"Done. {total_saved} masks saved to {FLAGS.save_dir}")


if __name__ == "__main__":
    app.run(infer)
