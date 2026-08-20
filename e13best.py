#!/usr/bin/env python
"""
e13best.py — deforestation segmentation, EfficientNet-B8 (AdvProp) encoder with the
cascaded DeepLabV3+ decoder 

Usage:
    python e13best.py --base_path /path/to/dataset
    python e13best.py --base_path /path/to/dataset --epochs 250
    python e13best.py --base_path /path/to/dataset --skip_bench
"""

import argparse
import json
import os
import platform
import random
import statistics
import time
import warnings

import albumentations as A
import cv2
import numpy as np
import rasterio
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import (
    BinaryAccuracy, BinaryF1Score, BinaryJaccardIndex,
    BinaryPrecision, BinaryRecall, ConfusionMatrix,
)

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from rasterio.errors import NotGeoreferencedWarning
warnings.simplefilter("ignore", NotGeoreferencedWarning)


# ----------------------------------------------------------------------------------
# Configuration — values chosen to match the manuscript
# ----------------------------------------------------------------------------------
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

SEED = 42
BATCH_SIZE = 10              # "a mini-batch of ten 4-band tiles"
EPOCHS = 2000                # "run for the full 2000 epochs" — override with --epochs
LEARNING_RATE = 1e-4
NUM_WORKERS = 4

IMG_SIZE = 512               # "resized to 512 x 512 pixels"
IN_CHANNELS = 4
NUM_GROUPS = 8               # "GroupNorm (eight groups)"
OUTPUT_STRIDE = 16

ENCODER_NAME = "timm-efficientnet-b8"
ENCODER_WEIGHTS = "advprop"

# Real B8 coefficients, for the audit output. The manuscript's 5.6x / 37 layers /
# 1.9x do not correspond to any EfficientNet variant.
B8_WIDTH_COEFF = 2.2
B8_DEPTH_COEFF = 3.6
NATIVE_RESOLUTION = 672
B0_RESOLUTION = 224

# Stage 1. The manuscript lists rates {1, 6, 12, 18}: four dilated 3x3 branches,
# dilation 1 being a plain 3x3. Set ASPP_IMAGE_POOLING True to add the standard
# image-level pooling branch — it usually helps, but the manuscript does not
# mention it, so it is off by default to keep the module exactly as described.
ATROUS_RATES = (1, 6, 12, 18)
ASPP_IMAGE_POOLING = False
ASPP_DROPOUT = 0.5

DECODER_CHANNELS = 256

# "replaces BatchNorm with GroupNorm" — applied to the whole network, encoder
# included. This discards the pretrained running statistics and affine parameters;
# that is what the manuscript describes.
FULL_GROUPNORM = True

# "raw digital numbers are min-max-scaled to the unit interval". Per-tile min-max
# discards cross-tile radiometric consistency; set to "reflectance" for a fixed
# divide by 10000 instead.
SCALING = "minmax"           # "minmax" | "reflectance"
REFLECTANCE_SCALE = 10000.0

TEST_FRACTION = 0.15
VAL_FRACTION_OF_REMAINDER = 0.1765          # -> 70 / 15 / 15

CKPT_PATH = "best_model_cascaded_efficientnetb8.pth"
FOOTPRINT_PATH = "computational_footprint.json"
RESULTS_PATH = "results.json"

N_PROB_BINS = 1000

BENCH_SIZES = (IMG_SIZE, IMG_SIZE // 2)
BENCH_WARMUP = 20
BENCH_ITERS = 100
BENCH_BATCH = 1

CONV_TOL = 0.01
CONV_PATIENCE = 25


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rng():
    return np.random.default_rng(SEED)


# ----------------------------------------------------------------------------------
# AMP compatibility
# ----------------------------------------------------------------------------------
def make_scaler(device):
    enabled = device.type == "cuda"
    try:
        return torch.amp.GradScaler(enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_ctx(device, enabled=True):
    use = enabled and device.type == "cuda"
    try:
        return torch.amp.autocast(device_type=device.type, enabled=use)
    except (AttributeError, TypeError):
        return torch.cuda.amp.autocast(enabled=use)


# ----------------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------------
class DeforestationDataset(Dataset):
    """4-band GeoTIFF tiles with binary masks."""

    def __init__(self, image_paths, mask_paths, transform=None):
        assert len(image_paths) == len(mask_paths)
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    @staticmethod
    def _scale(image):
        if SCALING == "minmax":
            lo = image.min()
            hi = image.max()
            rng_ = hi - lo
            return ((image - lo) / rng_ if rng_ > 0
                    else np.zeros_like(image, dtype=np.float32))
        return image / REFLECTANCE_SCALE

    @classmethod
    def _read(cls, image_path, mask_path):
        with rasterio.open(image_path) as f:
            image = f.read().transpose(1, 2, 0).astype(np.float32)
        with rasterio.open(mask_path) as f:
            mask = f.read(1)
        image = cls._scale(image).astype(np.float32)     # -> [0, 1]
        mask = (mask > 0).astype(np.uint8)               # fix 4
        return image, mask

    def __getitem__(self, idx):
        image, mask = self._read(self.image_paths[idx], self.mask_paths[idx])
        if self.transform is not None:
            out = self.transform(image=image, mask=mask)
            image, mask = out["image"], out["mask"]
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        return image, mask.float()


def _norm_kwargs():
    # fix 6: [0,1] -> [-1,1], the range AdvProp weights were trained on.
    return dict(mean=(0.5,) * IN_CHANNELS, std=(0.5,) * IN_CHANNELS,
                max_pixel_value=1.0)


def get_train_transforms():
    """Stochastic — training split only. Matches the manuscript's listed stack."""
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=30,
                           p=0.5, border_mode=0),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(**_norm_kwargs()),
        ToTensorV2(),
    ])


def get_eval_transforms():
    """Deterministic — validation and test. Fix 1: no augmentation of any kind."""
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(**_norm_kwargs()),
        ToTensorV2(),
    ])


def build_splits(base_path, seed=SEED):
    """Pair by filename (fix 5), then split 70/15/15 (fix 3)."""
    image_dir = os.path.join(base_path, "training", "image")
    mask_dir = os.path.join(base_path, "training", "mask")
    names = sorted(f for f in os.listdir(image_dir) if f.endswith(".tif"))
    if not names:
        raise FileNotFoundError(f"No .tif files in {image_dir}")

    image_paths, mask_paths, missing = [], [], []
    for n in names:
        mp = os.path.join(mask_dir, n)
        if not os.path.exists(mp):
            missing.append(n)
            continue
        image_paths.append(os.path.join(image_dir, n))
        mask_paths.append(mp)
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} image(s) have no matching mask, first few: {missing[:5]}")

    tv_i, te_i, tv_m, te_m = train_test_split(
        image_paths, mask_paths, test_size=TEST_FRACTION, random_state=seed)
    tr_i, va_i, tr_m, va_m = train_test_split(
        tv_i, tv_m, test_size=VAL_FRACTION_OF_REMAINDER, random_state=seed)
    print(f"Split sizes — train {len(tr_i)}, val {len(va_i)}, test {len(te_i)} "
          f"(paired by filename, seed {seed})")
    return (tr_i, tr_m), (va_i, va_m), (te_i, te_m)


# ----------------------------------------------------------------------------------
# Decoder components — section 3.2.2
# ----------------------------------------------------------------------------------
def gn(channels, num_groups=NUM_GROUPS):
    """GroupNorm with the largest divisor of `channels` not exceeding num_groups,
    so odd channel widths in the encoder do not fail."""
    g = num_groups
    while channels % g != 0 and g > 1:
        g -= 1
    return nn.GroupNorm(num_groups=g, num_channels=channels)


class ConvBlock(nn.Module):
    """Convolution -> GroupNorm -> ReLU."""

    def __init__(self, in_ch, out_ch, k=3, dilation=1):
        super().__init__()
        pad = dilation * (k - 1) // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, padding=pad, dilation=dilation, bias=False),
            gn(out_ch), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.block(x)


class ASPP(nn.Module):
    """Stage 1 — atrous spatial pyramid pooling.

    Dilated 3x3 convolutions at the configured rates (the manuscript's
    {1, 6, 12, 18}; dilation 1 is a plain 3x3), concatenated and projected back to
    `out_ch`. The optional image-level pooling branch is off by default because the
    manuscript does not list it.
    """

    def __init__(self, in_ch, out_ch=DECODER_CHANNELS, rates=ATROUS_RATES,
                 image_pooling=ASPP_IMAGE_POOLING, dropout=ASPP_DROPOUT):
        super().__init__()
        self.rates = tuple(rates)
        self.image_pooling = image_pooling
        self.branches = nn.ModuleList(
            [ConvBlock(in_ch, out_ch, k=3, dilation=r) for r in self.rates])
        n_branches = len(self.rates)
        if image_pooling:
            self.pool = nn.Sequential(
                nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.ReLU(inplace=True))
            n_branches += 1
        self.n_branches = n_branches
        self.project = nn.Sequential(
            nn.Conv2d(out_ch * n_branches, out_ch, 1, bias=False),
            gn(out_ch), nn.ReLU(inplace=True), nn.Dropout2d(dropout))

    def forward(self, x):
        feats = [b(x) for b in self.branches]
        if self.image_pooling:
            feats.append(F.interpolate(self.pool(x), size=x.shape[-2:],
                                       mode="bilinear", align_corners=False))
        return self.project(torch.cat(feats, dim=1))


class SkipRefine(nn.Module):
    """Stage 2 — feature-refinement module on a skip connection:
    1x1 convolution, GroupNorm, ReLU."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.refine = ConvBlock(in_ch, out_ch, k=1)

    def forward(self, x):
        return self.refine(x)


class ProgressiveUpsample(nn.Module):
    """Stage 3 — bilinear interpolation to the skip's resolution, a 3x3
    convolution, and a residual connection to stabilise gradient flow."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.reduce = (ConvBlock(in_ch, out_ch, k=1) if in_ch != out_ch
                       else nn.Identity())
        self.conv = ConvBlock(out_ch, out_ch, k=3)

    def forward(self, x, size):
        x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        x = self.reduce(x)
        return x + self.conv(x)                      # residual


class AttentionFusion(nn.Module):
    """Stage 4 — attention-based multi-scale feature fusion.

    A globally pooled descriptor of the two concatenated branches produces
    per-branch, per-channel weights, softmax-normalised across the branches, so
    the blend adapts per sample and per channel rather than being a fixed sum.
    """

    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.channels = channels
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels * 2, hidden, 1),
            nn.ReLU(inplace=True), nn.Conv2d(hidden, channels * 2, 1))
        self.post = ConvBlock(channels, channels, k=3)

    def forward(self, deep, shallow):
        w = self.gate(torch.cat([deep, shallow], dim=1))
        w = w.view(w.size(0), 2, self.channels, 1, 1).softmax(dim=1)
        fused = deep * w[:, 0] + shallow * w[:, 1]
        return fused + self.post(fused)              # residual


class CascadedDeepLabV3Plus(nn.Module):
    """EfficientNet-B8 encoder with the cascaded four-stage decoder of section
    3.2.2: ASPP -> refined skips -> progressive upsampling -> attention fusion."""

    def __init__(self, encoder_name=ENCODER_NAME, encoder_weights=ENCODER_WEIGHTS,
                 in_channels=IN_CHANNELS, classes=1, decoder_ch=DECODER_CHANNELS,
                 output_stride=OUTPUT_STRIDE):
        super().__init__()
        try:
            self.encoder = smp.encoders.get_encoder(
                encoder_name, in_channels=in_channels, depth=5,
                weights=encoder_weights, output_stride=output_stride)
        except TypeError:
            # Older smp releases have no output_stride kwarg on get_encoder.
            self.encoder = smp.encoders.get_encoder(
                encoder_name, in_channels=in_channels, depth=5,
                weights=encoder_weights)
            if hasattr(self.encoder, "make_dilated"):
                self.encoder.make_dilated(output_stride)

        ch = self.encoder.out_channels        # strides 1, 2, 4, 8, 16, (32 -> 16)

        # stage 1
        self.aspp = ASPP(ch[-1], decoder_ch)
        # stage 2 — one feature-refinement module per skip
        self.skip_refine = nn.ModuleList([
            SkipRefine(ch[3], decoder_ch),                 # stride 8
            SkipRefine(ch[2], decoder_ch // 2),            # stride 4
        ])
        # stage 3 — progressive upsampling with residual connections
        self.upsample = nn.ModuleList([
            ProgressiveUpsample(decoder_ch, decoder_ch),
            ProgressiveUpsample(decoder_ch, decoder_ch // 2),
        ])
        # stage 4 — attention-based fusion at each scale
        self.fusion = nn.ModuleList([
            AttentionFusion(decoder_ch),
            AttentionFusion(decoder_ch // 2),
        ])
        self.head = nn.Conv2d(decoder_ch // 2, classes, 1)

    def decoder_modules(self):
        return [self.aspp, self.skip_refine, self.upsample, self.fusion, self.head]

    def forward(self, x):
        size = x.shape[-2:]
        feats = self.encoder(x)

        y = self.aspp(feats[-1])                                   # stage 1
        for i, skip_idx in enumerate((3, 2)):
            skip = feats[skip_idx]
            y = self.upsample[i](y, skip.shape[-2:])               # stage 3
            y = self.fusion[i](y, self.skip_refine[i](skip))       # stages 2 + 4

        y = F.interpolate(y, size=size, mode="bilinear", align_corners=False)
        return self.head(y)


# ----------------------------------------------------------------------------------
# Model assembly
# ----------------------------------------------------------------------------------
def replace_batchnorm_with_groupnorm(module, num_groups=NUM_GROUPS, counter=None):
    """Recursively swap BatchNorm2d for GroupNorm."""
    if counter is None:
        counter = {"n": 0}
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, gn(child.num_features, num_groups))
            counter["n"] += 1
        else:
            replace_batchnorm_with_groupnorm(child, num_groups, counter)
    return counter["n"]


def count_modules(module, cls):
    return sum(1 for m in module.modules() if isinstance(m, cls))


def build_model():
    model = CascadedDeepLabV3Plus()
    info = {"batchnorm_before": count_modules(model, nn.BatchNorm2d)}
    info["batchnorm_replaced"] = (
        replace_batchnorm_with_groupnorm(model) if FULL_GROUPNORM else 0)
    info["batchnorm_remaining"] = count_modules(model, nn.BatchNorm2d)
    info["groupnorm_total"] = count_modules(model, nn.GroupNorm)
    info["full_groupnorm"] = FULL_GROUPNORM
    info["in_channels"] = IN_CHANNELS
    info["stem_adaptation"] = ("segmentation_models_pytorch built-in "
                              "3->4 channel weight adaptation")
    return model, info


def _count_encoder_stages(encoder):
    """Sequential stages: stem + MBConv block groups. EfficientNet has seven block
    groups plus the stem, so this is 8 — which is what "eight progressive stages"
    can legitimately refer to, as distinct from the number of spatial reductions."""
    blocks = getattr(encoder, "blocks", None)
    if blocks is None:
        model = getattr(encoder, "model", None)
        blocks = getattr(model, "blocks", None) if model is not None else None
    if blocks is None:
        return None
    return len(blocks) + 1


@torch.no_grad()
def audit_architecture(model, save="architecture_audit.json"):
    """Measure the structure rather than recalling it. Quote these numbers."""
    model.eval()
    encoder = model.encoder
    feats = encoder(torch.zeros(1, IN_CHANNELS, IMG_SIZE, IMG_SIZE))
    stages = [{"index": i, "channels": int(f.shape[1]),
               "spatial": [int(f.shape[2]), int(f.shape[3])],
               "stride": int(IMG_SIZE // f.shape[2])} for i, f in enumerate(feats)]
    names = [type(m).__name__ for m in encoder.modules()]
    dec_params = sum(p.numel() for m in model.decoder_modules()
                     for p in m.parameters())

    audit = {
        "model": "CascadedDeepLabV3Plus (section 3.2.2)",
        "encoder_name": ENCODER_NAME,
        "encoder_weights": ENCODER_WEIGHTS,
        "output_stride": OUTPUT_STRIDE,
        "scaling": SCALING,
        "input": {
            "channels": IN_CHANNELS,
            "resolution": IMG_SIZE,
            "native_resolution": NATIVE_RESOLUTION,
            "resolution_vs_b0": round(IMG_SIZE / B0_RESOLUTION, 2),
            "native_vs_b0": round(NATIVE_RESOLUTION / B0_RESOLUTION, 2),
        },
        "b8_scaling_coefficients": {
            "width": B8_WIDTH_COEFF, "depth": B8_DEPTH_COEFF,
            "note": "manuscript states width 5.6x / 37 layers / resolution 1.9x; "
                    "these are the published EfficientNet-B8 values",
        },
        "feature_stages": stages,
        "n_encoder_sequential_stages": _count_encoder_stages(encoder),
        "n_spatial_reductions": len({s["stride"] for s in stages}) - 1,
        "max_stride": max(s["stride"] for s in stages),
        "encoder_counts": {
            "inverted_residual_blocks": sum(1 for n in names
                                            if "InvertedResidual" in n),
            "depthwise_separable_blocks": sum(1 for n in names
                                              if "DepthwiseSeparable" in n),
            "squeeze_excite_modules": sum(1 for n in names
                                          if "SqueezeExcite" in n or n == "SEModule"),
            "conv2d_layers": count_modules(encoder, nn.Conv2d),
            "batchnorm_layers": count_modules(encoder, nn.BatchNorm2d),
            "groupnorm_layers": count_modules(encoder, nn.GroupNorm),
        },
        "decoder": {
            "stages": ["ASPP", "skip feature-refinement", "progressive upsampling",
                       "attention fusion"],
            "aspp_rates": list(ATROUS_RATES),
            "aspp_branches": model.aspp.n_branches,
            "aspp_image_pooling": ASPP_IMAGE_POOLING,
            "decoder_channels": DECODER_CHANNELS,
            "skip_refinement_modules": count_modules(model, SkipRefine),
            "progressive_upsample_modules": count_modules(model, ProgressiveUpsample),
            "attention_fusion_modules": count_modules(model, AttentionFusion),
            "groupnorm_layers": sum(count_modules(m, nn.GroupNorm)
                                    for m in model.decoder_modules()),
            "params": dec_params,
        },
        "params": {
            "encoder": sum(p.numel() for p in encoder.parameters()),
            "decoder": dec_params,
            "total": sum(p.numel() for p in model.parameters()),
            "trainable": sum(p.numel() for p in model.parameters()
                             if p.requires_grad),
        },
    }
    audit["encoder_counts"]["total_mbconv_blocks"] = (
        audit["encoder_counts"]["inverted_residual_blocks"] +
        audit["encoder_counts"]["depthwise_separable_blocks"])

    print("\n--- Architecture audit (measured) ---")
    print(f"{audit['model']}")
    print(f"Encoder: {ENCODER_NAME} / {ENCODER_WEIGHTS} / output stride "
          f"{OUTPUT_STRIDE}")
    print(f"Input: {IN_CHANNELS} bands at {IMG_SIZE}px "
          f"({audit['input']['resolution_vs_b0']}x vs B0's {B0_RESOLUTION}px; "
          f"B8 native {NATIVE_RESOLUTION}px = {audit['input']['native_vs_b0']}x)")
    print(f"B8 scaling: width {B8_WIDTH_COEFF}x, depth {B8_DEPTH_COEFF}x "
          f"(NOT 5.6x / 37 layers / 1.9x)")
    print(f"Encoder sequential stages: {audit['n_encoder_sequential_stages']} "
          f"(stem + block groups)")
    print(f"Encoder strides: {[s['stride'] for s in stages]}")
    print(f"Spatial reductions: {audit['n_spatial_reductions']} "
          f"(max stride {audit['max_stride']}) — NOT eight")
    for k, v in audit["encoder_counts"].items():
        print(f"  encoder {k:>28}: {v}")
    for k in ("aspp_rates", "aspp_branches", "aspp_image_pooling",
              "skip_refinement_modules", "progressive_upsample_modules",
              "attention_fusion_modules"):
        print(f"  decoder {k:>28}: {audit['decoder'][k]}")
    print(f"  {'encoder params':>36}: {audit['params']['encoder'] / 1e6:.1f}M")
    print(f"  {'decoder params':>36}: {audit['params']['decoder'] / 1e6:.1f}M")
    print(f"  {'trainable params':>36}: {audit['params']['trainable'] / 1e6:.1f}M")
    print("-------------------------------------\n")

    with open(save, "w") as f:
        json.dump(audit, f, indent=2)
    print(f"Wrote {save}")
    return audit


# ----------------------------------------------------------------------------------
# Instrumentation
#
# `allocated` is what PyTorch's allocator handed to tensors; `reserved` is what it
# took from the driver and is the closer analogue to nvidia-smi, though nvidia-smi
# also counts the CUDA context (~0.3-0.6 GB) that neither figure includes. State
# which one you report.
# ----------------------------------------------------------------------------------
def gpu_environment(device=DEVICE):
    env = {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": (torch.backends.cudnn.version()
                          if torch.backends.cudnn.is_available() else None),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        idx = device.index if device.index is not None else 0
        props = torch.cuda.get_device_properties(idx)
        env.update({
            "gpu_name": props.name,
            "gpu_total_memory_gb": round(props.total_memory / 1024 ** 3, 2),
            "gpu_capability": f"{props.major}.{props.minor}",
            "gpu_count": torch.cuda.device_count(),
        })
    return env


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _gb(n_bytes):
    return round(n_bytes / 1024 ** 3, 2)


class TrainingProfiler:
    """Per-epoch wall time and peak memory across the run."""

    def __init__(self, device=DEVICE):
        self.device = device
        self.cuda = device.type == "cuda"
        self.epoch_times = []
        self._t0 = None
        self._run_t0 = None

    def start_run(self):
        if self.cuda:
            torch.cuda.reset_peak_memory_stats(self.device)
        self._run_t0 = time.perf_counter()

    def epoch_start(self):
        _sync(self.device)
        self._t0 = time.perf_counter()

    def epoch_end(self):
        _sync(self.device)
        elapsed = time.perf_counter() - self._t0
        self.epoch_times.append(elapsed)
        return elapsed

    def summary(self):
        if not self.epoch_times:
            return {"epochs_timed": 0}
        out = {
            "epochs_timed": len(self.epoch_times),
            "epoch_seconds_mean": round(statistics.mean(self.epoch_times), 2),
            "epoch_seconds_median": round(statistics.median(self.epoch_times), 2),
            "epoch_seconds_min": round(min(self.epoch_times), 2),
            "epoch_seconds_max": round(max(self.epoch_times), 2),
            "epoch_minutes_median": round(statistics.median(self.epoch_times) / 60, 2),
            "total_wall_hours": round((time.perf_counter() - self._run_t0) / 3600, 2),
        }
        if len(self.epoch_times) > 1:
            # Epoch 1 carries cuDNN autotune and dataloader warmup.
            out["epoch_seconds_median_excl_first"] = round(
                statistics.median(self.epoch_times[1:]), 2)
        if self.cuda:
            out["peak_memory_allocated_gb"] = _gb(
                torch.cuda.max_memory_allocated(self.device))
            out["peak_memory_reserved_gb"] = _gb(
                torch.cuda.max_memory_reserved(self.device))
        return out


@torch.no_grad()
def benchmark_inference(model, sizes=BENCH_SIZES, batch_size=BENCH_BATCH,
                        n_warmup=BENCH_WARMUP, n_iter=BENCH_ITERS, amp=False,
                        device=DEVICE):
    """Median forward-pass latency and peak memory at each tile size.

    Each iteration is synced explicitly, so these are wall-clock latencies rather
    than queue-submission times. Warmup iterations are discarded because the first
    passes include cuDNN algorithm selection.
    """
    model.eval().to(device)
    results = {}
    for size in sizes:
        x = torch.randn(batch_size, IN_CHANNELS, size, size, device=device)
        for _ in range(n_warmup):
            with autocast_ctx(device, amp):
                model(x)
        _sync(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        times_ms = []
        for _ in range(n_iter):
            _sync(device)
            t0 = time.perf_counter()
            with autocast_ctx(device, amp):
                model(x)
            _sync(device)
            times_ms.append((time.perf_counter() - t0) * 1000.0)

        times_ms.sort()
        median = statistics.median(times_ms)
        entry = {
            "tile_size": size,
            "batch_size": batch_size,
            "precision": "amp" if amp else "fp32",
            "iterations": n_iter,
            "latency_ms_median": round(median, 2),
            "latency_ms_mean": round(statistics.mean(times_ms), 2),
            "latency_ms_p95": round(times_ms[int(0.95 * len(times_ms)) - 1], 2),
            "latency_ms_min": round(times_ms[0], 2),
            "throughput_tiles_per_s": round(1000.0 * batch_size / median, 1),
        }
        if device.type == "cuda":
            entry["peak_memory_allocated_gb"] = _gb(
                torch.cuda.max_memory_allocated(device))
            entry["peak_memory_reserved_gb"] = _gb(
                torch.cuda.max_memory_reserved(device))
        results[f"{size}px"] = entry
        del x
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return results


def convergence_epoch(val_iou, tol=CONV_TOL, patience=CONV_PATIENCE):
    """First epoch after which val IoU stays within `tol` of the run's best for at
    least `patience` consecutive epochs. A stated criterion beats an eyeballed one.
    Returns None if the run never settles."""
    if not val_iou:
        return None
    best = max(val_iou)
    for i in range(len(val_iou)):
        tail = val_iou[i:]
        if len(tail) >= patience and all(v >= best - tol for v in tail):
            return i + 1
    return None


def format_footprint(fp):
    env, tr, inf, cfg = (fp.get("environment", {}), fp.get("training", {}),
                         fp.get("inference", {}), fp.get("config", {}))
    gpu = env.get("gpu_name", "CPU")
    vram = env.get("gpu_total_memory_gb")
    gpu_str = f"{gpu} ({vram:.0f} GB)" if vram else gpu
    ep = tr.get("epoch_seconds_median_excl_first", tr.get("epoch_seconds_median"))
    ep_str = "n/a" if not ep else (f"{ep / 60:.1f} min" if ep >= 60 else f"{ep:.0f} s")

    lines = [
        "",
        "--- Computational footprint (measured) ---",
        f"Hardware: {gpu_str}, PyTorch {env.get('torch_version')}, "
        f"CUDA {env.get('cuda_version')}",
        f"Training: batch {cfg.get('batch_size')} x {IN_CHANNELS}-band "
        f"{IMG_SIZE}x{IMG_SIZE} tiles, {tr.get('epochs_timed')} epochs",
        f"  median epoch time     : {ep_str}",
        f"  total wall time       : {tr.get('total_wall_hours')} h",
        f"  peak memory allocated : {tr.get('peak_memory_allocated_gb')} GB",
        f"  peak memory reserved  : {tr.get('peak_memory_reserved_gb')} GB",
    ]
    for key, e in inf.items():
        lines.append(
            f"Inference @ {key} (batch {e['batch_size']}, {e['precision']}): "
            f"{e['latency_ms_median']} ms median "
            f"({e['throughput_tiles_per_s']} tiles/s), "
            f"peak reserved {e.get('peak_memory_reserved_gb')} GB")
    lines += ["------------------------------------------", ""]
    return "\n".join(lines)


# ----------------------------------------------------------------------------------
# Metrics (fix 7 — dataset-level accumulation, not a mean of per-batch scores)
# ----------------------------------------------------------------------------------
class MetricBundle:
    def __init__(self, device=DEVICE, threshold=0.5):
        kw = dict(threshold=threshold)
        self.iou = BinaryJaccardIndex(**kw).to(device)
        self.f1 = BinaryF1Score(**kw).to(device)
        self.precision = BinaryPrecision(**kw).to(device)
        self.recall = BinaryRecall(**kw).to(device)
        self.acc = BinaryAccuracy(**kw).to(device)
        self._all = [self.iou, self.f1, self.precision, self.recall, self.acc]

    def update(self, probs, targets):
        t = targets.int()
        for m in self._all:
            m.update(probs, t)

    def reset(self):
        for m in self._all:
            m.reset()

    def compute(self):
        return {
            "iou": self.iou.compute().item(),
            "f1": self.f1.compute().item(),
            "precision": self.precision.compute().item(),
            "recall": self.recall.compute().item(),
            "pixel_accuracy": self.acc.compute().item(),
        }


@torch.no_grad()
def evaluate(model, loader, criterion, device=DEVICE):
    model.eval()
    metrics = MetricBundle(device)
    total, n = 0.0, 0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        logits = model(images)
        total += criterion(logits, masks).item()
        n += 1
        metrics.update(logits.sigmoid(), masks)
    out = metrics.compute()
    out["loss"] = total / max(n, 1)
    return out


# ----------------------------------------------------------------------------------
# Streamed probability histogram -> PR / ROC / threshold sweep (fix 8)
# ----------------------------------------------------------------------------------
def _trapezoid(y, x):
    """np.trapz was removed in NumPy 2.0 and renamed to np.trapezoid; support both."""
    fn = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return float(fn(y, x))


@torch.no_grad()
def accumulate_prob_histogram(model, loader, n_bins=N_PROB_BINS, device=DEVICE):
    """Two histograms of predicted probability, one over positive-labelled pixels
    and one over negatives. Constant memory regardless of dataset size, and enough
    to reconstruct PR, ROC and IoU-vs-threshold exactly to bin resolution."""
    model.eval()
    pos = torch.zeros(n_bins, dtype=torch.float64, device=device)
    neg = torch.zeros(n_bins, dtype=torch.float64, device=device)
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        probs = model(images).sigmoid().flatten().double()
        targets = masks.flatten().int()
        idx = torch.clamp((probs * n_bins).long(), 0, n_bins - 1)
        pos += torch.bincount(idx[targets == 1], minlength=n_bins).double()
        neg += torch.bincount(idx[targets == 0], minlength=n_bins).double()
    return pos.cpu().numpy(), neg.cpu().numpy()


def curves_from_histogram(pos, neg, n_bins=N_PROB_BINS):
    """Reconstruct threshold-indexed counts. A pixel is predicted positive at
    threshold t when its probability >= t, so counts at bin i are suffix sums."""
    thresholds = np.arange(n_bins) / n_bins
    tp = np.cumsum(pos[::-1])[::-1]
    fp = np.cumsum(neg[::-1])[::-1]
    total_pos, total_neg = pos.sum(), neg.sum()
    fn = total_pos - tp
    tn = total_neg - fp

    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(tp + fp > 0, tp / (tp + fp), 1.0)
        recall = np.where(total_pos > 0, tp / total_pos, 0.0)
        fpr = np.where(total_neg > 0, fp / total_neg, 0.0)
        iou = np.where(tp + fp + fn > 0, tp / (tp + fp + fn), 0.0)
        f1 = np.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), 0.0)

    # Trapezoidal AUC over recall (ascending) and FPR (ascending).
    order_pr = np.argsort(recall)
    pr_auc = _trapezoid(precision[order_pr], recall[order_pr])
    order_roc = np.argsort(fpr)
    roc_auc = _trapezoid(recall[order_roc], fpr[order_roc])

    best_i = int(np.argmax(iou))
    return {
        "thresholds": thresholds, "precision": precision, "recall": recall,
        "fpr": fpr, "tpr": recall, "iou": iou, "f1": f1,
        "pr_auc": pr_auc, "roc_auc": roc_auc,
        "best_threshold": float(thresholds[best_i]),
        "best_threshold_iou": float(iou[best_i]),
        "iou_at_0.5": float(iou[n_bins // 2]),
        "positive_pixel_fraction": float(total_pos / max(total_pos + total_neg, 1)),
        "n_bins": n_bins,
    }


def plot_pr_roc(curves, save_path="pr_roc_curves.png"):
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 5))
    a.plot(curves["recall"], curves["precision"],
           label=f"PR (AUC = {curves['pr_auc']:.3f})")
    a.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall")
    a.legend(); a.grid(True)
    b.plot(curves["fpr"], curves["tpr"], label=f"ROC (AUC = {curves['roc_auc']:.3f})")
    b.plot([0, 1], [0, 1], "r--", linewidth=1)
    b.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC")
    b.legend(); b.grid(True)
    fig.tight_layout(); fig.savefig(save_path, dpi=200); plt.close(fig)


def plot_threshold_sweep(curves, save_path="threshold_sweep_iou.png"):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(curves["thresholds"], curves["iou"], label="IoU")
    ax.plot(curves["thresholds"], curves["f1"], label="F1", alpha=0.7)
    ax.axvline(curves["best_threshold"], color="grey", linestyle="--", linewidth=1)
    ax.annotate(f"best t = {curves['best_threshold']:.2f}\n"
                f"IoU = {curves['best_threshold_iou']:.3f}",
                xy=(curves["best_threshold"], curves["best_threshold_iou"]),
                xytext=(6, -20), textcoords="offset points", fontsize=9)
    ax.set(xlabel="Threshold", ylabel="Score", title="Score vs. threshold")
    ax.legend(); ax.grid(True)
    fig.tight_layout(); fig.savefig(save_path, dpi=200); plt.close(fig)


def plot_probability_histogram(pos, neg,
                               save_path="prediction_probability_histogram.png"):
    centers = (np.arange(len(pos)) + 0.5) / len(pos)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.semilogy(centers, neg + 1, label="label = background", alpha=0.8)
    ax.semilogy(centers, pos + 1, label="label = deforestation", alpha=0.8)
    ax.set(xlabel="Predicted probability", ylabel="Pixel count (log, +1)",
           title="Pixel-wise predicted probabilities by true class")
    ax.legend(); ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout(); fig.savefig(save_path, dpi=200); plt.close(fig)


# ----------------------------------------------------------------------------------
# Per-sample metrics (fix 9 — tensor ops, not per-sample sklearn)
# ----------------------------------------------------------------------------------
@torch.no_grad()
def compute_metrics_per_sample(model, dataset, threshold=0.5, device=DEVICE):
    model.eval()
    out = []
    for idx in range(len(dataset)):
        image, mask = dataset[idx]
        prob = model(image.unsqueeze(0).to(device)).sigmoid()[0, 0]
        pred = (prob >= threshold)
        gt = mask.to(device)[0].bool()
        tp = (pred & gt).sum().item()
        fp = (pred & ~gt).sum().item()
        fn = (~pred & gt).sum().item()
        iou = tp / (tp + fp + fn) if (tp + fp + fn) else float("nan")
        prec = tp / (tp + fp) if (tp + fp) else float("nan")
        rec = tp / (tp + fn) if (tp + fn) else float("nan")
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else float("nan")
        out.append({"index": idx, "iou": iou, "precision": prec, "recall": rec,
                    "f1": f1, "gt_positive_pixels": int(tp + fn)})
    return out


def log_worst_samples(metrics_list, k=5):
    """Empty tiles have undefined IoU; rank only tiles that contain positives."""
    scored = [m for m in metrics_list if not np.isnan(m["iou"])]
    n_empty = len(metrics_list) - len(scored)
    ranked = sorted(scored, key=lambda x: x["iou"])
    print(f"\nWorst {min(k, len(ranked))} tiles by IoU "
          f"({n_empty} tile(s) excluded: no positive ground-truth pixels)")
    for e in ranked[:k]:
        print(f"  idx {e['index']:>5}: IoU={e['iou']:.4f} P={e['precision']:.4f} "
              f"R={e['recall']:.4f} F1={e['f1']:.4f} "
              f"({e['gt_positive_pixels']} gt px)")
    return ranked


# ----------------------------------------------------------------------------------
# Display helpers (fix 10 — three bands before imshow, and undo the normalization)
# ----------------------------------------------------------------------------------
def to_display_rgb(image_tensor, brightness=1.5):
    img = image_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    img = img * 0.5 + 0.5                       # [-1,1] -> [0,1]
    img = img[:, :, :3] if img.shape[2] >= 3 else np.repeat(img, 3, axis=2)
    return np.clip(img * brightness, 0, 1)


DIFF_CMAP = ListedColormap(["black", "red", "blue"])     # 0 = bg, 1 = FP, 2 = FN


def _sample_indices(dataset, n):
    n = min(n, len(dataset))
    return rng().choice(len(dataset), size=n, replace=False)   # fix 11 — seeded


@torch.no_grad()
def _predict(model, image_tensor, device=DEVICE):
    return model(image_tensor.unsqueeze(0).to(device)).sigmoid().cpu().numpy()[0, 0]


# ----------------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------------
def plot_history(history, save_path="training_curves.png"):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (a, b) = plt.subplots(1, 2, figsize=(14, 5))
    a.plot(epochs, history["train_loss"], label="train")
    a.plot(epochs, history["val_loss"], label="val")
    a.set(xlabel="Epoch", ylabel="Focal loss", title="Loss"); a.legend(); a.grid(True)
    b.plot(epochs, history["train_iou"], label="train")
    b.plot(epochs, history["val_iou"], label="val")
    b.set(xlabel="Epoch", ylabel="IoU", title="IoU"); b.legend(); b.grid(True)
    conv = convergence_epoch(history["val_iou"])
    if conv is not None:
        b.axvline(conv, color="grey", linestyle="--", linewidth=1)
        b.annotate(f"converged ~epoch {conv}",
                   xy=(conv, min(history["val_iou"])), xytext=(4, 4),
                   textcoords="offset points", fontsize=8, color="grey")
    fig.tight_layout(); fig.savefig(save_path, dpi=200); plt.close(fig)


def plot_epoch_times(history, save_path="epoch_times.png"):
    if not history.get("epoch_seconds"):
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(history["epoch_seconds"]) + 1), history["epoch_seconds"])
    ax.set(xlabel="Epoch", ylabel="Wall time (s)", title="Per-epoch wall time")
    ax.grid(True)
    fig.tight_layout(); fig.savefig(save_path, dpi=200); plt.close(fig)


@torch.no_grad()
def plot_confusion(model, loader, save_path="confusion_matrix.png", device=DEVICE):
    model.eval()
    cmm = ConfusionMatrix(task="binary", threshold=0.5).to(device)
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        cmm.update(model(images).sigmoid(), masks.int())
    cm = cmm.compute().cpu().numpy()
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, cmap=plt.cm.Blues); ax.figure.colorbar(im, ax=ax)
    names = ["Background", "Deforestation"]
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=names, yticklabels=names,
           title="Confusion Matrix", ylabel="True", xlabel="Predicted")
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{int(cm[i, j]):,}", ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout(); fig.savefig(save_path, dpi=200); plt.close(fig)
    return cm


@torch.no_grad()
def plot_sample_predictions(model, dataset, n=8, device=DEVICE,
                            save_path="sample_predictions.png"):
    model.eval()
    idxs = _sample_indices(dataset, n)
    fig, axes = plt.subplots(len(idxs), 4, figsize=(18, 4.2 * len(idxs)))
    if len(idxs) == 1:
        axes = np.expand_dims(axes, 0)
    for row, idx in enumerate(idxs):
        image, mask = dataset[int(idx)]
        pred = (_predict(model, image, device) > 0.5).astype(np.uint8)
        gt = mask.squeeze().numpy().astype(np.uint8)
        diff = ((pred == 1) & (gt == 0)).astype(np.uint8) + \
               2 * ((pred == 0) & (gt == 1)).astype(np.uint8)
        panels = [(to_display_rgb(image), f"Image (idx {idx})", {}),
                  (gt, "Ground truth", {"cmap": "gray"}),
                  (pred, "Prediction", {"cmap": "gray"}),
                  (diff, "FP red / FN blue",
                   {"cmap": DIFF_CMAP, "vmin": 0, "vmax": 2})]
        for col, (data, title, kw) in enumerate(panels):
            axes[row, col].imshow(data, **kw)
            axes[row, col].set_title(title)
            axes[row, col].axis("off")
    fig.tight_layout(); fig.savefig(save_path, dpi=150); plt.close(fig)


@torch.no_grad()
def plot_prediction_heatmap(model, dataset, n=6, device=DEVICE,
                            save_path="prediction_heatmap.png"):
    model.eval()
    idxs = _sample_indices(dataset, n)
    fig, axes = plt.subplots(len(idxs), 2, figsize=(12, 5 * len(idxs)))
    if len(idxs) == 1:
        axes = np.expand_dims(axes, 0)
    for i, idx in enumerate(idxs):
        image, _ = dataset[int(idx)]
        prob = _predict(model, image, device)
        axes[i, 0].imshow(to_display_rgb(image))
        axes[i, 0].set_title(f"Image (idx {idx})"); axes[i, 0].axis("off")
        hm = axes[i, 1].imshow(prob, cmap="hot", vmin=0, vmax=1)
        axes[i, 1].set_title("Predicted probability"); axes[i, 1].axis("off")
        fig.colorbar(hm, ax=axes[i, 1])
    fig.tight_layout(); fig.savefig(save_path, dpi=150); plt.close(fig)


@torch.no_grad()
def plot_aggregate_error_heatmap(model, loader, device=DEVICE,
                                 save_path="aggregate_error_heatmap.png"):
    model.eval()
    error_sum, count = None, 0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        pred = (model(images).sigmoid() >= 0.5).int()
        err = (pred != masks.int()).float().sum(dim=0).cpu().numpy()
        error_sum = err if error_sum is None else error_sum + err
        count += images.size(0)
    if error_sum is None or count == 0:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(error_sum[0] / count, cmap="hot")
    ax.set_title("Aggregate error heat map\n(mean misclassification frequency)")
    fig.colorbar(im, ax=ax, label="Mean error")
    fig.tight_layout(); fig.savefig(save_path, dpi=200); plt.close(fig)


@torch.no_grad()
def plot_overlay_predictions(model, dataset, n=6, alpha=0.4, device=DEVICE,
                             save_path="overlay_predictions.png"):
    model.eval()
    idxs = _sample_indices(dataset, n)
    fig, axes = plt.subplots(len(idxs), 2, figsize=(14, 5 * len(idxs)))
    if len(idxs) == 1:
        axes = np.expand_dims(axes, 0)
    for i, idx in enumerate(idxs):
        image, _ = dataset[int(idx)]
        pred = (_predict(model, image, device) > 0.5).astype(np.float32)
        rgb = to_display_rgb(image)
        overlay = np.zeros_like(rgb); overlay[..., 0] = pred
        axes[i, 0].imshow(rgb)
        axes[i, 0].set_title(f"Image (idx {idx})"); axes[i, 0].axis("off")
        axes[i, 1].imshow(np.clip((1 - alpha) * rgb + alpha * overlay, 0, 1))
        axes[i, 1].set_title("Prediction overlay (red)"); axes[i, 1].axis("off")
    fig.tight_layout(); fig.savefig(save_path, dpi=150); plt.close(fig)


@torch.no_grad()
def plot_overlay_with_contours(model, dataset, n=6, device=DEVICE,
                               save_path="overlay_with_contours.png"):
    model.eval()
    idxs = _sample_indices(dataset, n)
    fig, axes = plt.subplots(len(idxs), 2, figsize=(14, 5 * len(idxs)))
    if len(idxs) == 1:
        axes = np.expand_dims(axes, 0)
    for i, idx in enumerate(idxs):
        image, _ = dataset[int(idx)]
        pred = (_predict(model, image, device) > 0.5).astype(np.uint8)
        rgb = to_display_rgb(image)
        canvas = np.ascontiguousarray((rgb * 255).astype(np.uint8))
        contours, _ = cv2.findContours(pred, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, (0, 255, 0), 2)
        axes[i, 0].imshow(rgb)
        axes[i, 0].set_title(f"Image (idx {idx})"); axes[i, 0].axis("off")
        axes[i, 1].imshow(canvas.astype(np.float32) / 255.0)
        axes[i, 1].set_title("Predicted boundary contours"); axes[i, 1].axis("off")
    fig.tight_layout(); fig.savefig(save_path, dpi=150); plt.close(fig)


@torch.no_grad()
def plot_fp_fn(model, dataset, n=6, device=DEVICE, save_path="fp_fn_overlay.png"):
    model.eval()
    idxs = _sample_indices(dataset, n)
    fig, axes = plt.subplots(len(idxs), 3, figsize=(15, 5 * len(idxs)))
    if len(idxs) == 1:
        axes = np.expand_dims(axes, 0)
    for i, idx in enumerate(idxs):
        image, mask = dataset[int(idx)]
        pred = (_predict(model, image, device) > 0.5).astype(np.uint8)
        gt = mask.squeeze().numpy().astype(np.uint8)
        fp = ((pred == 1) & (gt == 0)).astype(np.float32)
        fn = ((pred == 0) & (gt == 1)).astype(np.float32)
        rgb = to_display_rgb(image)
        overlay = np.zeros_like(rgb); overlay[..., 0] = fp; overlay[..., 2] = fn
        axes[i, 0].imshow(rgb)
        axes[i, 0].set_title(f"Image (idx {idx})"); axes[i, 0].axis("off")
        axes[i, 1].imshow(np.clip(0.6 * overlay + 0.4 * rgb, 0, 1))
        axes[i, 1].set_title("FP red / FN blue"); axes[i, 1].axis("off")
        axes[i, 2].imshow(fp + 2 * fn, cmap=DIFF_CMAP, vmin=0, vmax=2)
        axes[i, 2].set_title("Difference map"); axes[i, 2].axis("off")
    fig.tight_layout(); fig.savefig(save_path, dpi=150); plt.close(fig)


@torch.no_grad()
def plot_area_diagnostics(model, dataset, device=DEVICE,
                          hist_path="predicted_area_histogram.png",
                          scatter_path="pred_vs_gt_scatter.png"):
    """Predicted-area histogram and predicted-vs-true fraction scatter, in one pass
    over the dataset rather than two."""
    model.eval()
    pred_px, pred_frac, gt_frac = [], [], []
    for idx in range(len(dataset)):
        image, mask = dataset[idx]
        h, w = image.shape[1], image.shape[2]
        pred = (_predict(model, image, device) > 0.5).astype(np.uint8)
        gt = mask.squeeze().numpy()
        pred_px.append(int(pred.sum()))
        pred_frac.append(pred.sum() / (h * w))
        gt_frac.append(gt.sum() / (h * w))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(pred_px, bins=50, alpha=0.75)
    ax.set(xlabel="Predicted deforestation area (pixels)", ylabel="Tiles",
           title="Predicted area per tile")
    fig.tight_layout(); fig.savefig(hist_path, dpi=200); plt.close(fig)

    lim = max(max(gt_frac, default=0), max(pred_frac, default=0), 1e-6) * 1.05
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(gt_frac, pred_frac, alpha=0.6)
    ax.plot([0, lim], [0, lim], "r--", label="ideal")
    ax.set(xlim=(0, lim), ylim=(0, lim), xlabel="Ground-truth fraction",
           ylabel="Predicted fraction",
           title="Predicted vs. ground-truth deforestation fraction")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(scatter_path, dpi=200); plt.close(fig)
    return {"mean_predicted_fraction": float(np.mean(pred_frac)) if pred_frac else None,
            "mean_gt_fraction": float(np.mean(gt_frac)) if gt_frac else None}


def plot_iou_distribution(metrics_list, save_path="iou_distribution.png"):
    """Per-tile IoU spread. A bar with an error whisker hides the shape; a
    histogram with the mean marked does not."""
    ious = [m["iou"] for m in metrics_list if not np.isnan(m["iou"])]
    if not ious:
        return {}
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(ious, bins=30, range=(0, 1), alpha=0.8, color="teal")
    ax.axvline(float(np.mean(ious)), color="black", linestyle="--",
               label=f"mean {np.mean(ious):.3f}")
    ax.axvline(float(np.median(ious)), color="grey", linestyle=":",
               label=f"median {np.median(ious):.3f}")
    ax.set(xlabel="Per-tile IoU", ylabel="Tiles",
           title=f"Per-tile IoU distribution (n={len(ious)} tiles with positives)")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(save_path, dpi=200); plt.close(fig)
    return {"per_tile_iou_mean": float(np.mean(ious)),
            "per_tile_iou_median": float(np.median(ious)),
            "per_tile_iou_std": float(np.std(ious)),
            "n_tiles_scored": len(ious),
            "n_tiles_without_positives": len(metrics_list) - len(ious)}


@torch.no_grad()
def plot_ranked_samples(model, dataset, metrics_list, k=5, best=False, device=DEVICE,
                        save_path="worst_samples.png"):
    scored = [m for m in metrics_list if not np.isnan(m["iou"])]
    ranked = sorted(scored, key=lambda x: x["iou"], reverse=best)[:k]
    if not ranked:
        return
    model.eval()
    fig, axes = plt.subplots(len(ranked), 4, figsize=(20, 5 * len(ranked)))
    if len(ranked) == 1:
        axes = np.expand_dims(axes, 0)
    for i, entry in enumerate(ranked):
        image, mask = dataset[entry["index"]]
        pred = (_predict(model, image, device) >= 0.5).astype(np.uint8)
        gt = mask.squeeze().numpy().astype(np.uint8)
        diff = ((pred == 1) & (gt == 0)).astype(np.uint8) + \
               2 * ((pred == 0) & (gt == 1)).astype(np.uint8)
        panels = [(to_display_rgb(image), f"Image (idx {entry['index']})", {}),
                  (gt, "Ground truth", {"cmap": "gray"}),
                  (pred, f"Prediction — IoU {entry['iou']:.3f}", {"cmap": "gray"}),
                  (diff, "FP red / FN blue",
                   {"cmap": DIFF_CMAP, "vmin": 0, "vmax": 2})]
        for col, (data, title, kw) in enumerate(panels):
            axes[i, col].imshow(data, **kw)
            axes[i, col].set_title(title)
            axes[i, col].axis("off")
    fig.tight_layout(); fig.savefig(save_path, dpi=150); plt.close(fig)


# ----------------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------------
def train_model(model, train_loader, val_loader, epochs, device=DEVICE):
    """Returns (history, footprint_summary)."""
    model.to(device)
    criterion = smp.losses.FocalLoss(mode="binary")
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = make_scaler(device)
    train_metrics = MetricBundle(device)

    history = {"train_loss": [], "val_loss": [], "train_iou": [], "val_iou": [],
               "epoch_seconds": []}
    best_iou = -1.0
    best_epoch = -1

    profiler = TrainingProfiler(device)
    profiler.start_run()

    for epoch in range(epochs):
        profiler.epoch_start()
        model.train()
        train_metrics.reset()
        total, n = 0.0, 0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast_ctx(device):
                logits = model(images)
                loss = criterion(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += loss.item()
            n += 1
            with torch.no_grad():
                train_metrics.update(logits.detach().float().sigmoid(), masks)

        tr = train_metrics.compute()
        tr_loss = total / max(n, 1)
        va = evaluate(model, val_loader, criterion, device)
        secs = profiler.epoch_end()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va["loss"])
        history["train_iou"].append(tr["iou"])
        history["val_iou"].append(va["iou"])
        history["epoch_seconds"].append(round(secs, 2))
        print(f"[{epoch + 1}/{epochs}] train loss {tr_loss:.4f} IoU {tr['iou']:.4f} | "
              f"val loss {va['loss']:.4f} IoU {va['iou']:.4f} | {secs:.1f}s")

        if va["iou"] > best_iou:
            best_iou, best_epoch = va["iou"], epoch
            torch.save({"epoch": epoch, "val_iou": best_iou,
                        "state_dict": model.state_dict()}, CKPT_PATH)
            print(f"    new best val IoU {best_iou:.4f} — checkpoint saved")

    fp = profiler.summary()
    fp["best_epoch"] = best_epoch + 1
    fp["best_val_iou"] = best_iou
    print(f"Training complete. Best val IoU {best_iou:.4f} at epoch {best_epoch + 1} "
          f"({fp.get('total_wall_hours')} h wall, "
          f"{fp.get('epoch_seconds_median')}s median epoch)")
    return history, fp


def load_best(model, path=CKPT_PATH, device=DEVICE):
    """Fix 2. Without this every downstream number describes the final epoch."""
    ckpt = torch.load(path, map_location=device)
    state = (ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt
             else ckpt)
    model.load_state_dict(state)
    model.to(device)
    if isinstance(ckpt, dict) and "epoch" in ckpt:
        print(f"Loaded best checkpoint: epoch {ckpt['epoch'] + 1}, "
              f"val IoU {ckpt['val_iou']:.4f}")
    else:
        print(f"Loaded checkpoint from {path}")
    return model


# ----------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description="Cascaded DeepLabV3+ / EfficientNet-B8 deforestation segmentation")
    p.add_argument("--base_path", required=True, help="Dataset root")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    p.add_argument("--bench_iters", type=int, default=BENCH_ITERS)
    p.add_argument("--bench_amp", action="store_true",
                   help="benchmark inference under autocast instead of fp32")
    p.add_argument("--skip_bench", action="store_true")
    p.add_argument("--n_figure_samples", type=int, default=6)
    args = p.parse_args()

    set_seed(SEED)

    env = gpu_environment(DEVICE)
    print(f"Device: {env.get('gpu_name', 'CPU')}"
          + (f" ({env['gpu_total_memory_gb']:.0f} GB)"
             if "gpu_total_memory_gb" in env else "")
          + f" | torch {env['torch_version']} | CUDA {env['cuda_version']}")

    model, build_info = build_model()
    print(f"Model: CascadedDeepLabV3Plus / {ENCODER_NAME} / {ENCODER_WEIGHTS} / "
          f"full_groupnorm={FULL_GROUPNORM} / scaling={SCALING}")
    print(f"  BatchNorm: {build_info['batchnorm_before']} before, "
          f"{build_info['batchnorm_replaced']} replaced, "
          f"{build_info['batchnorm_remaining']} remaining")
    audit = audit_architecture(model)

    if IMG_SIZE != NATIVE_RESOLUTION:
        print(f"NOTE: training at {IMG_SIZE}px, below B8's native "
              f"{NATIVE_RESOLUTION}px. Report the value used, not the native one.\n")

    (tr_i, tr_m), (va_i, va_m), (te_i, te_m) = build_splits(args.base_path)
    train_ds = DeforestationDataset(tr_i, tr_m, get_train_transforms())
    val_ds = DeforestationDataset(va_i, va_m, get_eval_transforms())
    test_ds = DeforestationDataset(te_i, te_m, get_eval_transforms())

    dl = dict(num_workers=args.num_workers, pin_memory=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **dl)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **dl)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, **dl)

    history, train_fp = train_model(
        model, train_loader, val_loader, args.epochs, DEVICE)
    plot_history(history)
    plot_epoch_times(history)

    # ---- everything below uses the selected checkpoint, not the final epoch ----
    model = load_best(model)
    criterion = smp.losses.FocalLoss(mode="binary")

    val_results = evaluate(model, val_loader, criterion)
    test_results = evaluate(model, test_loader, criterion)

    print("\nValidation (model-selection split — report as validation, not test):")
    for k, v in val_results.items():
        print(f"  {k:>16}: {v:.4f}")
    print("\nTest (held out, evaluated once):")
    for k, v in test_results.items():
        print(f"  {k:>16}: {v:.4f}")

    pos_hist, neg_hist = accumulate_prob_histogram(model, test_loader)
    curves = curves_from_histogram(pos_hist, neg_hist)
    print(f"\nTest PR AUC {curves['pr_auc']:.4f} | ROC AUC {curves['roc_auc']:.4f} | "
          f"IoU at 0.5 {curves['iou_at_0.5']:.4f} | "
          f"best threshold {curves['best_threshold']:.2f} "
          f"-> IoU {curves['best_threshold_iou']:.4f}")
    print(f"Positive pixels: {curves['positive_pixel_fraction'] * 100:.3f}% of test set")
    plot_pr_roc(curves, "pr_roc_curves_test.png")
    plot_threshold_sweep(curves, "threshold_sweep_iou_test.png")
    plot_probability_histogram(pos_hist, neg_hist,
                               "prediction_probability_histogram_test.png")

    plot_confusion(model, test_loader, "confusion_matrix_test.png")
    plot_aggregate_error_heatmap(model, test_loader,
                                 save_path="aggregate_error_heatmap_test.png")

    per_sample = compute_metrics_per_sample(model, test_ds)
    log_worst_samples(per_sample, k=5)
    iou_stats = plot_iou_distribution(per_sample, "iou_distribution_test.png")
    plot_ranked_samples(model, test_ds, per_sample, k=5, best=False,
                        save_path="worst_samples_test.png")
    plot_ranked_samples(model, test_ds, per_sample, k=5, best=True,
                        save_path="best_samples_test.png")

    ns = args.n_figure_samples
    plot_sample_predictions(model, test_ds, ns, save_path="sample_predictions_test.png")
    plot_prediction_heatmap(model, test_ds, ns, save_path="prediction_heatmap_test.png")
    plot_overlay_predictions(model, test_ds, ns,
                             save_path="overlay_predictions_test.png")
    plot_overlay_with_contours(model, test_ds, ns,
                               save_path="overlay_with_contours_test.png")
    plot_fp_fn(model, test_ds, ns, save_path="fp_fn_overlay_test.png")
    area_stats = plot_area_diagnostics(
        model, test_ds, hist_path="predicted_area_histogram_test.png",
        scatter_path="pred_vs_gt_scatter_test.png")

    # ---- computational footprint ----
    infer_fp = ({} if args.skip_bench else benchmark_inference(
        model, n_iter=args.bench_iters, amp=args.bench_amp, device=DEVICE))
    conv = convergence_epoch(history["val_iou"])
    footprint = {
        "environment": env,
        "config": {"batch_size": args.batch_size, "img_size": IMG_SIZE,
                   "in_channels": IN_CHANNELS, "epochs": args.epochs,
                   "amp_training": DEVICE.type == "cuda"},
        "training": train_fp,
        "inference": infer_fp,
        "convergence": {
            "epoch": conv,
            "criterion": f"val IoU within {CONV_TOL} of run best for "
                         f">={CONV_PATIENCE} consecutive epochs",
            "best_val_iou": max(history["val_iou"]) if history["val_iou"] else None,
        },
    }
    print(format_footprint(footprint))
    if conv is not None:
        print(f"Validation IoU converged by epoch {conv} "
              f"({footprint['convergence']['criterion']}).\n")
    with open(FOOTPRINT_PATH, "w") as f:
        json.dump(footprint, f, indent=2)
    print(f"Wrote {FOOTPRINT_PATH}")

    # ---- results ----
    with open(RESULTS_PATH, "w") as f:
        json.dump({
            "config": {
                "model": "CascadedDeepLabV3Plus (section 3.2.2)",
                "encoder": ENCODER_NAME, "encoder_weights": ENCODER_WEIGHTS,
                "output_stride": OUTPUT_STRIDE,
                "atrous_rates": list(ATROUS_RATES),
                "aspp_image_pooling": ASPP_IMAGE_POOLING,
                "decoder_channels": DECODER_CHANNELS,
                "in_channels": IN_CHANNELS, "img_size": IMG_SIZE,
                "native_resolution": NATIVE_RESOLUTION,
                "full_groupnorm": FULL_GROUPNORM, "num_groups": NUM_GROUPS,
                "scaling": SCALING,
                "normalization": "scaled to [0,1] then mean=0.5 std=0.5 -> [-1,1]",
                "loss": "FocalLoss(binary)", "optimizer": "Adam",
                "lr": LEARNING_RATE, "batch_size": args.batch_size,
                "epochs": args.epochs, "seed": SEED,
                "split": "70/15/15 train/val/test, paired by filename",
                "eval_transforms": "deterministic resize + normalize only",
                "metric_aggregation": "dataset-level accumulation",
            },
            "build_info": build_info,
            "architecture_audit": audit,
            "split_sizes": {"train": len(train_ds), "val": len(val_ds),
                            "test": len(test_ds)},
            "validation": val_results,
            "test": test_results,
            "test_curves": {k: v for k, v in curves.items()
                            if not isinstance(v, np.ndarray)},
            "test_per_tile_iou": iou_stats,
            "test_area": area_stats,
            "computational_footprint": footprint,
        }, f, indent=2)
    print(f"Wrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
