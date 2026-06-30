import copy
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from absl import app, flags
from torchdiffeq import odeint
from torchdyn.core import NeuralODE
from torchvision.utils import save_image
from tqdm import trange

from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
    RectifiedConditionalFlowMatcher,
    RectifiedConditionalFlowMatcher_v2,
    TargetConditionalFlowMatcher,
    VariancePreservingConditionalFlowMatcher,
)
from torchcfm.models.unet.unet_residal import UNetModel

from FMS.segflow.image_datasets import load_data
from FMS.synflow.utils import ema

FLAGS = flags.FLAGS

# ── model ─────────────────────────────────────────────────────────────────────
flags.DEFINE_string("model", "icfm", "flow matching variant: icfm | otcfm | fm | si | rfm | rfm_v2")
flags.DEFINE_bool("self_cond", False,
                  "enable self-conditioning: the model also sees its own previous x1 estimate "
                  "(doubles UNet input channels from 3 to 6)")
flags.DEFINE_float("self_cond_prob", 0.5,
                   "probability of computing a self-condition during training (self_cond only)")

# ── data ──────────────────────────────────────────────────────────────────────
flags.DEFINE_list("image_dirs_training", None, "comma-separated training image directories")
flags.DEFINE_list("class_dirs_training", None, "comma-separated training annotation directories")
flags.DEFINE_list("image_dirs_validation", None, "comma-separated validation image directories")
flags.DEFINE_list("class_dirs_validation", None, "comma-separated validation annotation directories")
flags.DEFINE_list("image_dirs_testing", None, "comma-separated testing image directories")
flags.DEFINE_list("class_dirs_testing", None, "comma-separated testing annotation directories")
flags.DEFINE_integer("img_size", 256, "spatial resolution")
flags.DEFINE_integer("batch_size", 4, "batch size")

# ── training ──────────────────────────────────────────────────────────────────
flags.DEFINE_float("lr", 1e-5, "learning rate")
flags.DEFINE_float("grad_clip", 1.0, "gradient norm clip")
flags.DEFINE_integer("total_steps", 48002, "total training steps")
flags.DEFINE_integer("warmup", 10000, "linear warmup steps")
flags.DEFINE_float("ema_decay", 0.9999, "EMA decay rate")
flags.DEFINE_bool("parallel", False, "DataParallel multi-GPU training")
flags.DEFINE_integer("ckpt_step", 0, "resume from this step (0 = train from scratch)")

# ── eval / sampling ───────────────────────────────────────────────────────────
flags.DEFINE_integer("num_inference_steps", 10, "ODE steps during evaluation")
flags.DEFINE_string("integration_method", "euler", "ODE solver: euler | dopri5")
flags.DEFINE_bool("save_trajactory", False, "save full ODE trajectories during evaluation")
flags.DEFINE_integer("visit_step", 2000, "visualization frequency (reserved for future use)")

# ── output ────────────────────────────────────────────────────────────────────
flags.DEFINE_string("output_dir", "./outputs/segflow", "checkpoint directory")
flags.DEFINE_integer("save_step", 2000, "checkpoint and evaluation frequency")
flags.DEFINE_string("sample_saved_dir", "./outputs/segflow/predictions",
                    "directory for saving val/test predictions")
flags.DEFINE_string("save_trajactory_dir", "./outputs/segflow/trajectories",
                    "directory to save ODE trajectories when save_trajactory=True")

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


# ── ODE helpers ───────────────────────────────────────────────────────────────

@torch.no_grad()
def pipeline_rf(timesteps, model, z0):
    device = z0.device
    dtype = z0.dtype
    ttlsteps = len(timesteps)
    ts = timesteps
    dt = 1.0 / ttlsteps
    x = z0.clone()
    all_x = []
    B = x.shape[0]
    for t in ts:
        t_b = torch.full((B,), float(t.item()), device=device, dtype=dtype)
        v = model(x=x, timesteps=t_b)
        x = x + dt * v
        all_x.append(x)
    x = x.clamp(-1.0, 1.0)
    return x, all_x


def to_binary_mask(t, threshold=0.0):
    """Convert [-1,1] tensor to binary {0,1}. Averages to grayscale if 3-channel."""
    if t.dim() != 4:
        raise ValueError(f"Expected 4D tensor [B,C,H,W], got {t.shape}")
    g = t.mean(dim=1, keepdim=True) if t.shape[1] == 3 else t
    return (g < threshold).float()


@torch.no_grad()
def generate_samples_with_neural_ode(model, x, num_steps=100, integration_method="dopri5",
                                      rtol=1e-5, atol=1e-5, return_trajectories=False):
    model.eval()
    batch_size = x.shape[0]
    self_conditioning = FLAGS.self_cond

    # ---- Euler with self-conditioning: manual loop carrying the predicted x1 ----
    if integration_method == "euler" and self_conditioning:
        t_span = torch.linspace(0, 1, num_steps + 1, device=x.device)
        dt = 1.0 / num_steps
        x_curr = x.clone()
        self_cond = torch.zeros_like(x_curr)  # [B, 3, H, W] zero at first step

        traj_list = [x_curr.detach().cpu()] if return_trajectories else None
        vel_list = [] if return_trajectories else None

        for i in range(num_steps):
            t_b = torch.full((batch_size,), t_span[i].item(), device=x.device, dtype=x_curr.dtype)
            model_input = torch.cat([x_curr, self_cond], dim=1)  # [B, 6, H, W]
            v = model(x=model_input, timesteps=t_b * 1000.0)
            # update self-condition: estimate target x1 = x_curr + (1 - t) * v
            t_expand = t_b.view(-1, 1, 1, 1)
            self_cond = (x_curr + (1 - t_expand) * v).detach()
            x_curr = x_curr + dt * v
            if return_trajectories:
                traj_list.append(x_curr.detach().cpu())
                vel_list.append(v.detach().cpu())

        out = x_curr
        if return_trajectories:
            return {
                "output": out,
                "trajectories": torch.stack(traj_list),
                "velocities": torch.stack(vel_list) if vel_list else None,
                "timesteps": t_span.detach().cpu(),
            }
        return out

    class _Wrapper(torch.nn.Module):
        def __init__(self, model, batch_size, self_cond):
            super().__init__()
            self.model = model
            self.batch_size = batch_size
            self.self_cond = self_cond

        def forward(self, t, x, **kwargs):
            if isinstance(t, (int, float)):
                timesteps = torch.full((self.batch_size,), float(t), device=x.device, dtype=x.dtype)
            elif t.dim() == 0:
                timesteps = torch.full((self.batch_size,), t.item(), device=x.device, dtype=x.dtype)
            else:
                timesteps = t
            # adaptive / stateless solvers cannot carry the self-condition → use zeros
            if self.self_cond:
                x = torch.cat([x, torch.zeros_like(x)], dim=1)
            return self.model(x=x, timesteps=timesteps * 1000.0)

    cond_model = _Wrapper(model, batch_size, self_conditioning)

    if integration_method == "euler":
        node = NeuralODE(cond_model, solver="euler")
        t_span = torch.linspace(0, 1, num_steps + 1, device=x.device)
        traj = node.trajectory(x, t_span=t_span)
        out = traj[-1]
        if return_trajectories:
            velocities = torch.stack([
                cond_model(t_span[i], traj[i]).detach().cpu()
                for i in range(traj.shape[0])
            ])
            return {
                "output": out,
                "trajectories": traj.detach().cpu(),
                "velocities": velocities,
                "timesteps": t_span.detach().cpu(),
            }
    elif integration_method == "dopri5":
        t_span = torch.linspace(0, 1, 2, device=x.device)
        traj = odeint(cond_model, x, t_span, rtol=rtol, atol=atol, method="dopri5")
        out = traj[-1]
        if return_trajectories:
            return {
                "output": out,
                "trajectories": traj.detach().cpu(),
                "velocities": None,
                "timesteps": t_span.detach().cpu(),
            }
    else:
        raise NotImplementedError(
            f"Unknown integration method '{integration_method}'. Supported: 'euler', 'dopri5'"
        )

    if return_trajectories:
        return {"output": out, "trajectories": None, "velocities": None, "timesteps": None}
    return out


@torch.no_grad()
def generate_samples_with_pipelines(model, x1, num_inference_steps=120,
                                     integration_method="euler",
                                     binarize=False, threshold=0.0,
                                     return_single_channel=True,
                                     return_trajectories=False):
    x_seg = generate_samples_with_neural_ode(
        model=model,
        x=x1,
        num_steps=num_inference_steps,
        integration_method=integration_method,
        return_trajectories=return_trajectories,
    )

    if return_trajectories:
        x_out = x_seg["output"]
    else:
        x_out = x_seg

    if binarize:
        x_bin = to_binary_mask(x_out, threshold=threshold)
        if return_single_channel:
            if return_trajectories:
                x_seg["output"] = x_bin
                return x_seg
            return x_bin
        x_bin = x_bin.repeat(1, 3, 1, 1)
        if return_trajectories:
            x_seg["output"] = x_bin
            return x_seg
        return x_bin

    return x_seg


# ── metric computation ────────────────────────────────────────────────────────

@torch.no_grad()
def compute_matrix(dataloader, dataset_len, model, device, batch_size=8,
                   num_inference_steps=120, model_name="Model",
                   save_predictions=True, step=None, split="val"):
    print(f"Computing metrics on {dataset_len} {split} samples...")

    save_dir = None
    if save_predictions and FLAGS.sample_saved_dir:
        folder_name = f"{model_name}_{split}_{step}"
        save_dir = os.path.join(FLAGS.sample_saved_dir, folder_name)
        os.makedirs(save_dir, exist_ok=True)
        print(f"  Saving predictions to: {save_dir}")

    traj_dir = None
    if FLAGS.save_trajactory and FLAGS.save_trajactory_dir:
        traj_folder = f"{model_name}_{split}_{step}"
        traj_dir = os.path.join(FLAGS.save_trajactory_dir, traj_folder)
        os.makedirs(traj_dir, exist_ok=True)
        print(f"  Saving trajectories to: {traj_dir}")

    total_intersection = 0.0
    total_union = 0.0
    total_tp = 0.0
    total_fp = 0.0
    total_fn = 0.0

    sample_ious = []

    num_batches = dataset_len // batch_size + (1 if dataset_len % batch_size else 0)
    samples_processed = 0

    print(f"Processing {num_batches} batches (batch_size={batch_size})")

    for batch_idx in range(num_batches):
        x_img, y_mask, image_name = next(dataloader)
        x_img = x_img.to(device)
        y_mask = y_mask.to(device)

        if samples_processed >= dataset_len:
            break

        samples_to_process = min(x_img.shape[0], dataset_len - samples_processed)
        if samples_to_process < x_img.shape[0]:
            x_img = x_img[:samples_to_process]
            y_mask = y_mask[:samples_to_process]

        pred_result = generate_samples_with_pipelines(
            model,
            x_img,
            num_inference_steps=num_inference_steps,
            integration_method=FLAGS.integration_method,
            return_trajectories=FLAGS.save_trajactory,
        )
        if FLAGS.save_trajactory:
            pred_mask = pred_result["output"]
            traj_batch = pred_result.get("trajectories")
            vel_batch = pred_result.get("velocities")
            t_batch = pred_result.get("timesteps")
        else:
            pred_mask = pred_result
            traj_batch = vel_batch = t_batch = None

        y_gray = y_mask.mean(dim=1, keepdim=True)
        y_binary = (y_gray < 0.0).float()

        if pred_mask.shape[1] == 3:
            pred_gray = pred_mask.mean(dim=1, keepdim=True)
            pred_binary = (pred_gray < 0.0).float()
        else:
            pred_binary = (pred_mask < 0.0).float()

        for i in range(samples_to_process):
            img_name = os.path.splitext(os.path.basename(image_name[i]))[0]

            if save_dir is not None:
                binary_inverted = 1.0 - pred_binary[i:i + 1]
                save_image(binary_inverted, os.path.join(save_dir, f"{img_name}.png"))

            if traj_dir is not None and traj_batch is not None and vel_batch is not None:
                np.savez_compressed(
                    os.path.join(traj_dir, f"{img_name}.npz"),
                    trajectories=traj_batch[:, i].clamp(-1, 1).numpy(),
                    velocities=vel_batch[:, i].clamp(-1, 1).numpy(),
                    timesteps=t_batch.numpy(),
                )

        for i in range(samples_to_process):
            s_pred = pred_binary[i:i + 1]
            s_gt = y_binary[i:i + 1]
            s_inter = (s_pred * s_gt).sum().item()
            s_union = ((s_pred + s_gt) > 0).float().sum().item()
            sample_ious.append((
                s_inter / (s_union + 1e-8),
                image_name[i],
            ))

        intersection = (pred_binary * y_binary).sum()
        union = ((pred_binary + y_binary) > 0).float().sum()
        tp = (pred_binary * y_binary).sum()
        fp = (pred_binary * (1 - y_binary)).sum()
        fn = ((1 - pred_binary) * y_binary).sum()

        total_intersection += intersection.item()
        total_union += union.item()
        total_tp += tp.item()
        total_fp += fp.item()
        total_fn += fn.item()

        samples_processed += samples_to_process
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == num_batches:
            print(f"  {samples_processed}/{dataset_len} — IoU so far: "
                  f"{total_intersection / (total_union + 1e-8):.4f}")

    mIoU = total_intersection / (total_union + 1e-8)
    F1 = (2 * total_tp) / (2 * total_tp + total_fp + total_fn + 1e-8)

    summary = (
        f"\n[Metrics | model={model_name} | split={split} | step={step}]\n"
        f"  samples={samples_processed}  mIoU={mIoU:.4f}  F1={F1:.4f}\n"
        f"  TP={total_tp:.0f}  FP={total_fp:.0f}  FN={total_fn:.0f}"
    )
    print(summary)
    log_path = os.path.join(FLAGS.output_dir, "train.log")
    with open(log_path, "a") as f:
        f.write(summary + "\n")

    if len(sample_ious) >= 6:
        sample_ious.sort(key=lambda x: x[0])
        visualize_best_worst_samples(
            sample_ious[:3], sample_ious[-3:],
            model_name=model_name, step=step, split=split,
        )

    return float(mIoU), float(F1)


@torch.no_grad()
def display_images(normal_seg, ema_seg, x_img, y_mask, step=0):
    log_path = os.path.join(FLAGS.output_dir, "train.log")
    lines = [
        f"\n[Validation @ step {step}]",
        f"  batch_size={x_img.shape[0]}  image_shape={list(x_img.shape[1:])}",
    ]
    msg = "\n".join(lines)
    print(msg)
    with open(log_path, "a") as f:
        f.write(msg + "\n")


def visualize_best_worst_samples(worst_samples, best_samples,
                                  model_name="Model", step=None, split="val"):
    log_path = os.path.join(FLAGS.output_dir, "train.log")
    lines = [
        f"\n[Best/Worst Samples | model={model_name} | split={split} | step={step}]",
        "  Worst 3 (lowest IoU):",
    ]
    for rank, (iou, fname) in enumerate(worst_samples, 1):
        lines.append(f"    #{rank}  IoU={iou:.4f}  file={fname}")
    lines.append("  Best 3 (highest IoU):")
    for rank, (iou, fname) in enumerate(best_samples, 1):
        lines.append(f"    #{rank}  IoU={iou:.4f}  file={fname}")

    msg = "\n".join(lines)
    print(msg)
    with open(log_path, "a") as f:
        f.write(msg + "\n")


# ── training loop ─────────────────────────────────────────────────────────────

def train(argv):
    print(
        f"lr={FLAGS.lr}  total_steps={FLAGS.total_steps}  "
        f"ema_decay={FLAGS.ema_decay}  save_step={FLAGS.save_step}"
    )

    datalooper, len_dataset = load_data(
        data_dir=FLAGS.image_dirs_training,
        class_dir=FLAGS.class_dirs_training,
        batch_size=FLAGS.batch_size,
        image_size=FLAGS.img_size,
        deterministic=False,
        random_crop=False,
        random_flip=False,
    )

    datalooper_val, len_dataset_val = load_data(
        data_dir=FLAGS.image_dirs_validation,
        class_dir=FLAGS.class_dirs_validation,
        batch_size=FLAGS.batch_size,
        image_size=FLAGS.img_size,
        deterministic=True,
        random_crop=False,
        random_flip=False,
        drop_last=False,
    )

    datalooper_test, len_dataset_test = load_data(
        data_dir=FLAGS.image_dirs_testing,
        class_dir=FLAGS.class_dirs_testing,
        batch_size=FLAGS.batch_size,
        image_size=FLAGS.img_size,
        deterministic=True,
        random_crop=False,
        random_flip=False,
        drop_last=False,
    )

    # self-conditioning feeds the model's own previous x1 estimate as extra input
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

    total_params = sum(p.numel() for p in net_model.parameters())
    print(f"UNet parameters: {total_params / 1e6:.2f}M  "
          f"(self_cond={FLAGS.self_cond}, in_channels={in_channels})")

    ema_model = copy.deepcopy(net_model)

    if FLAGS.parallel and use_cuda and torch.cuda.device_count() > 1:
        print("Warning: parallel training is performing slightly worse than single GPU training "
              "due to statistics computation in dataparallel.")
        net_model = torch.nn.DataParallel(net_model)
        ema_model = torch.nn.DataParallel(ema_model)

    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
    optim = torch.optim.Adam(net_model.parameters(), lr=FLAGS.lr)
    warmup = LinearLR(optim, start_factor=1e-8, total_iters=FLAGS.warmup)
    cosine = CosineAnnealingLR(optim, T_max=FLAGS.total_steps - FLAGS.warmup, eta_min=FLAGS.lr * 0.1)
    sched = SequentialLR(optim, schedulers=[warmup, cosine], milestones=[FLAGS.warmup])

    start_step = -1
    if FLAGS.ckpt_step > 0:
        ckpt_path = os.path.join(FLAGS.output_dir, f"{FLAGS.model}_{FLAGS.ckpt_step}.pt")
        checkpoint = torch.load(ckpt_path, map_location=device)
        net_model.load_state_dict(checkpoint["net_model"], strict=True)
        ema_model.load_state_dict(checkpoint["ema_model"], strict=True)
        optim.load_state_dict(checkpoint["optim"])
        for st in optim.state.values():
            for k, v in st.items():
                if torch.is_tensor(v):
                    st[k] = v.to(device)
        if "sched" in checkpoint:
            try:
                sched.load_state_dict(checkpoint["sched"])
            except Exception as e:
                print(f"Warning: could not restore scheduler state: {e}")
        start_step = int(checkpoint.get("step", 0))
        print(f"Resumed from step {start_step} @ {ckpt_path}")

    model_size = sum(p.data.nelement() for p in net_model.parameters())
    print(f"Model params: {model_size / 1024 / 1024:.2f} M")

    sigma = 0.0
    print(f"Using model: {FLAGS.model}")
    if FLAGS.model == "otcfm":
        FM = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
    elif FLAGS.model == "icfm":
        FM = ConditionalFlowMatcher(sigma=sigma)
    elif FLAGS.model == "rfm":
        FM = RectifiedConditionalFlowMatcher(schedule="t2")
    elif FLAGS.model == "rfm_v2":
        FM = RectifiedConditionalFlowMatcher_v2(schedule="t2")
    elif FLAGS.model == "fm":
        FM = TargetConditionalFlowMatcher(sigma=sigma)
    elif FLAGS.model == "si":
        FM = VariancePreservingConditionalFlowMatcher(sigma=sigma)
    else:
        raise NotImplementedError(f"Unknown model '{FLAGS.model}'")

    savedir = FLAGS.output_dir + "/"
    os.makedirs(savedir, exist_ok=True)

    with trange(start_step + 1, FLAGS.total_steps, dynamic_ncols=True) as pbar:
        for step in pbar:
            optim.zero_grad()
            x_img, y_mask, image_name = next(datalooper)
            x_img, y_mask = x_img.to(device), y_mask.to(device)

            if FLAGS.integration_method == "euler":
                steps = FLAGS.num_inference_steps
                t_span = torch.linspace(0, 1, steps + 1, device=device, dtype=x_img.dtype)
                t_idx = step % (steps + 1)
                t = t_span[t_idx].expand(x_img.shape[0])
            else:
                t = torch.rand(x_img.shape[0], device=device, dtype=x_img.dtype)

            t, xt, ut = FM.sample_location_and_conditional_flow(x_img, y_mask, t=t)

            if FLAGS.self_cond:
                # With probability self_cond_prob, run a first no-grad pass with a zero
                # placeholder, estimate the target x1_hat = xt + (1-t)*v, and feed it back
                # as the self-condition for the real (gradient-tracked) forward pass.
                self_cond = torch.zeros_like(xt)  # [B, 3, H, W]
                if random.random() < FLAGS.self_cond_prob:
                    with torch.no_grad():
                        vt_uncond = net_model(
                            x=torch.cat([xt, self_cond], dim=1), timesteps=t * 1000.0
                        )
                        t_expand = t.view(-1, 1, 1, 1)
                        self_cond = (xt + (1 - t_expand) * vt_uncond).detach()
                vt = net_model(x=torch.cat([xt, self_cond], dim=1), timesteps=t * 1000.0)
            else:
                vt = net_model(x=xt, timesteps=t * 1000.0)

            loss = F.mse_loss(vt, ut, reduction="none").mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net_model.parameters(), FLAGS.grad_clip)
            optim.step()
            sched.step()
            ema(net_model, ema_model, FLAGS.ema_decay)

            if step % 100 == 0:
                print(f"  step {step}  loss={loss.item():.6f}"
                      f"  lr={optim.param_groups[0]['lr']:.2e}")

            if step % FLAGS.visit_step == 0:
                pass  # visualisation hook (reserved)

            if (step >= 10000 and step % FLAGS.save_step == 0) or step == 10:
                net_model.eval()
                ema_model.eval()

                print("\n" + "=" * 50)
                print(f"Evaluating EMA Model at step {step}")
                print("=" * 50)
                ema_mIoU, ema_F1 = compute_matrix(
                    datalooper_val, len_dataset_val, ema_model, device,
                    batch_size=FLAGS.batch_size,
                    num_inference_steps=FLAGS.num_inference_steps,
                    model_name="ema", step=step, split="val",
                )

                print("\n" + "=" * 50)
                print(f"Testing EMA Model at step {step}")
                print("=" * 50)
                test_ema_mIoU, test_ema_F1 = compute_matrix(
                    datalooper_test, len_dataset_test, ema_model, device,
                    batch_size=FLAGS.batch_size,
                    num_inference_steps=FLAGS.num_inference_steps,
                    model_name="ema", step=step, split="test",
                )

                net_model.train()
                ema_model.train()

                torch.save(
                    {
                        "net_model": net_model.state_dict(),
                        "ema_model": ema_model.state_dict(),
                        "sched": sched.state_dict(),
                        "optim": optim.state_dict(),
                        "step": step,
                        "ema_mIoU": ema_mIoU,
                        "ema_F1": ema_F1,
                    },
                    os.path.join(
                        savedir,
                        f"{FLAGS.model}_{step}.pt",
                        # f"{FLAGS.model}_{step}_emamIoU_{ema_mIoU:.4f}"
                        # f"_testemamIoU_{test_ema_mIoU:.4f}.pt",
                    ),
                )


if __name__ == "__main__":
    app.run(train)
