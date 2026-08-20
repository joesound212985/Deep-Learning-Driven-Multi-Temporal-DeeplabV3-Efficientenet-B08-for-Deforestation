#!/usr/bin/env python
"""
Deforestation segmentation — EfficientNet-B8 encoder with a cascaded DeepLabV3+-style
decoder.

Encoder: EfficientNet-B8 (AdvProp weights), 4-channel input, output stride 16.
Decoder: ASPP (rates 6/12/18 + 1x1 + image pooling)
         -> skip feature-refinement modules (1x1 conv, GroupNorm, ReLU)
         -> progressive bilinear upsampling with 3x3 convs and residual connections
         -> attention-based multi-scale feature fusion.
Loss:    Focal.


Usage:
    python train_deforestation.py --base_path /path/to/dataset
"""

import argparse
import json
import os
import random

import albumentations as A
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

import warnings
from rasterio.errors import NotGeoreferencedWarning
warnings.simplefilter("ignore", NotGeoreferencedWarning)


# ----------------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------------
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

SEED = 42
BATCH_SIZE = 5
EPOCHS = 250
LEARNING_RATE = 1e-4
NUM_WORKERS = 4
PRELOAD = False              # cache dataset in RAM; forces NUM_WORKERS = 0

IMG_SIZE = 512               # B8's native training resolution is 672
IN_CHANNELS = 4
OUTPUT_STRIDE = 16
ATROUS_RATES = (6, 12, 18)
DECODER_CHANNELS = 256
NUM_GROUPS = 8

ENCODER_NAME = "timm-efficientnet-b8"
ENCODER_WEIGHTS = "advprop"

# EfficientNet-B8 scaling coefficients: width 2.2x, depth 3.6x, native input 672px
# (B0 baseline is 224px, so native is ~3.0x). Training at IMG_SIZE below 672 is a
# memory trade-off — report the resolution actually used, not the native one.
NATIVE_RESOLUTION = 672

# Stem adaptation: how the 4th band's filters are initialized from 3-channel weights.
STEM_INIT = "mean"           # "mean" | "repeat" | "zero"
STEM_RESCALE = True          # scale filters by 3/4 to preserve activation magnitude

# The decoder uses GroupNorm throughout. Set True to also replace the encoder's
# BatchNorm — this discards the pretrained running statistics and affine parameters,
# which is what the original script did.
ENCODER_GROUPNORM = False

# Reflectance scaling, then Normalize maps [0,1] -> [-1,1] for the advprop weights.
REFLECTANCE_SCALE = 10000.0

CKPT_PATH = "best_model.pth"


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------------
class DeforestationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None, preload=False):
        assert len(image_paths) == len(mask_paths)
        self.image_paths, self.mask_paths = image_paths, mask_paths
        self.transform, self.preload = transform, preload
        if preload:
            print(f"Preloading {len(image_paths)} samples...")
            self.cache = [self._read(i, m) for i, m in zip(image_paths, mask_paths)]
            print("Preload complete.")

    @staticmethod
    def _read(image_path, mask_path):
        with rasterio.open(image_path) as f:
            image = f.read().transpose(1, 2, 0).astype(np.float32)
        with rasterio.open(mask_path) as f:
            mask = f.read(1)
        image = image / REFLECTANCE_SCALE          # -> [0, 1]
        mask = (mask > 0).astype(np.uint8)         # force strictly binary
        return image, mask

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image, mask = (self.cache[idx] if self.preload
                       else self._read(self.image_paths[idx], self.mask_paths[idx]))
        if self.transform is not None:
            out = self.transform(image=image, mask=mask)
            image, mask = out["image"], out["mask"]
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        return image, mask.float()


def _norm():
    return dict(mean=(0.5,) * IN_CHANNELS, std=(0.5,) * IN_CHANNELS, max_pixel_value=1.0)


def get_train_transforms():
    """Stochastic augmentation — training only."""
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=30,
                           p=0.5, border_mode=0),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(**_norm()),
        ToTensorV2(),
    ])


def get_eval_transforms():
    """Deterministic — validation and test. No augmentation of any kind."""
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(**_norm()),
        ToTensorV2(),
    ])


def build_splits(base_path, seed=SEED):
    """Pair images to masks by filename, then split 70/15/15."""
    image_dir = os.path.join(base_path, "training", "image")
    mask_dir = os.path.join(base_path, "training", "mask")
    names = sorted(f for f in os.listdir(image_dir) if f.endswith(".tif"))
    image_paths, mask_paths = [], []
    for n in names:
        mp = os.path.join(mask_dir, n)
        if not os.path.exists(mp):
            raise FileNotFoundError(f"No mask for image {n}")
        image_paths.append(os.path.join(image_dir, n))
        mask_paths.append(mp)

    tv_i, te_i, tv_m, te_m = train_test_split(
        image_paths, mask_paths, test_size=0.15, random_state=seed)
    tr_i, va_i, tr_m, va_m = train_test_split(
        tv_i, tv_m, test_size=0.1765, random_state=seed)
    print(f"Split sizes — train {len(tr_i)}, val {len(va_i)}, test {len(te_i)}")
    return (tr_i, tr_m), (va_i, va_m), (te_i, te_m)


# ----------------------------------------------------------------------------------
# Decoder components
# ----------------------------------------------------------------------------------
def gn(channels, num_groups=NUM_GROUPS):
    """GroupNorm with the largest divisor of `channels` not exceeding num_groups."""
    g = num_groups
    while channels % g != 0 and g > 1:
        g -= 1
    return nn.GroupNorm(g, channels)


class ConvBlock(nn.Module):
    """Conv -> GroupNorm -> ReLU."""

    def __init__(self, in_ch, out_ch, k=3, dilation=1):
        super().__init__()
        pad = dilation * (k - 1) // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, padding=pad, dilation=dilation, bias=False),
            gn(out_ch), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.block(x)


class ASPP(nn.Module):
    """Atrous spatial pyramid pooling: a 1x1 branch, three dilated 3x3 branches at
    the configured rates, and an image-level pooling branch, concatenated and
    projected back to `out_ch`."""

    def __init__(self, in_ch, out_ch=DECODER_CHANNELS, rates=ATROUS_RATES):
        super().__init__()
        self.branches = nn.ModuleList(
            [ConvBlock(in_ch, out_ch, k=1)] +
            [ConvBlock(in_ch, out_ch, k=3, dilation=r) for r in rates])
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.ReLU(inplace=True))
        self.project = nn.Sequential(
            nn.Conv2d(out_ch * (len(rates) + 2), out_ch, 1, bias=False),
            gn(out_ch), nn.ReLU(inplace=True), nn.Dropout2d(0.5))

    def forward(self, x):
        size = x.shape[-2:]
        feats = [b(x) for b in self.branches]
        feats.append(F.interpolate(self.pool(x), size=size,
                                   mode="bilinear", align_corners=False))
        return self.project(torch.cat(feats, dim=1))


class AttentionFusion(nn.Module):
    """Gated fusion of two same-resolution feature maps. A globally pooled descriptor
    of the concatenation produces per-branch, per-channel weights, softmax-normalized
    across the two branches, so the blend adapts per sample and per channel."""

    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.channels = channels
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels * 2, hidden, 1),
            nn.ReLU(inplace=True), nn.Conv2d(hidden, channels * 2, 1))

    def forward(self, deep, shallow):
        w = self.gate(torch.cat([deep, shallow], dim=1))
        w = w.view(w.size(0), 2, self.channels, 1, 1).softmax(dim=1)
        return deep * w[:, 0] + shallow * w[:, 1]


class UpStage(nn.Module):
    """One progressive-upsampling stage: bilinear x2, 3x3 conv with a residual
    connection, a refined skip feature, attention fusion, and a second residual."""

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.reduce = ConvBlock(in_ch, out_ch, k=1) if in_ch != out_ch else nn.Identity()
        self.conv = ConvBlock(out_ch, out_ch, k=3)
        self.skip = ConvBlock(skip_ch, out_ch, k=1)      # feature-refinement module
        self.fuse = AttentionFusion(out_ch)
        self.post = ConvBlock(out_ch, out_ch, k=3)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.reduce(x)
        x = x + self.conv(x)                             # residual
        x = self.fuse(x, self.skip(skip))                # attention fusion
        return x + self.post(x)                          # residual


class CascadedDeepLab(nn.Module):
    """EfficientNet encoder with a cascaded ASPP / refined-skip / progressive-upsample
    / attention-fusion decoder."""

    def __init__(self, encoder_name=ENCODER_NAME, encoder_weights=ENCODER_WEIGHTS,
                 classes=1, decoder_ch=DECODER_CHANNELS, output_stride=OUTPUT_STRIDE):
        super().__init__()
        try:
            self.encoder = smp.encoders.get_encoder(
                encoder_name, in_channels=3, depth=5,
                weights=encoder_weights, output_stride=output_stride)
        except TypeError:
            # Older smp releases have no output_stride kwarg on get_encoder.
            self.encoder = smp.encoders.get_encoder(
                encoder_name, in_channels=3, depth=5, weights=encoder_weights)
            if hasattr(self.encoder, "make_dilated"):
                self.encoder.make_dilated(output_stride)

        ch = self.encoder.out_channels        # strides 1, 2, 4, 8, 16, (32 -> 16)
        self.aspp = ASPP(ch[-1], decoder_ch)
        self.up1 = UpStage(decoder_ch, ch[3], decoder_ch)          # -> stride 8
        self.up2 = UpStage(decoder_ch, ch[2], decoder_ch // 2)     # -> stride 4
        self.head = nn.Conv2d(decoder_ch // 2, classes, 1)

    def decoder_modules(self):
        return [self.aspp, self.up1, self.up2, self.head]

    def forward(self, x):
        size = x.shape[-2:]
        feats = self.encoder(x)
        y = self.aspp(feats[-1])
        y = self.up1(y, feats[3])
        y = self.up2(y, feats[2])
        y = F.interpolate(y, size=size, mode="bilinear", align_corners=False)
        return self.head(y)


# ----------------------------------------------------------------------------------
# Model assembly
# ----------------------------------------------------------------------------------
def replace_bn_with_gn(module, num_groups=NUM_GROUPS, counter=None):
    if counter is None:
        counter = {"n": 0}
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, gn(child.num_features, num_groups))
            counter["n"] += 1
        else:
            replace_bn_with_gn(child, num_groups, counter)
    return counter["n"]


def count_bn(module):
    return sum(1 for m in module.modules() if isinstance(m, nn.BatchNorm2d))


def _first_conv(module):
    """Return (parent, attr_name, conv) for the first Conv2d in the module tree."""
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            return module, name, child
        found = _first_conv(child)
        if found is not None:
            return found
    return None


def expand_stem(model, in_channels=IN_CHANNELS, strategy=STEM_INIT, rescale=STEM_RESCALE):
    """
    Replace the stem convolution so it accepts `in_channels` bands, reusing the
    pretrained 3-channel filters. Done explicitly rather than left to the library
    default so the initialization can be stated in the write-up.

      'mean'   — extra bands initialized to the mean of the pretrained RGB filters
      'repeat' — extra bands copy the pretrained first-channel filters
      'zero'   — extra bands initialized to zero
    """
    parent, name, conv = _first_conv(model)
    if conv.in_channels == in_channels:
        return {"changed": False}

    old_w = conv.weight.data.clone()
    new_conv = nn.Conv2d(in_channels, conv.out_channels, conv.kernel_size,
                         stride=conv.stride, padding=conv.padding,
                         dilation=conv.dilation, groups=conv.groups,
                         bias=conv.bias is not None)
    w = torch.zeros(conv.out_channels, in_channels, *conv.kernel_size)
    n_copy = min(old_w.shape[1], in_channels)
    w[:, :n_copy] = old_w[:, :n_copy]
    if in_channels > old_w.shape[1]:
        extra = in_channels - old_w.shape[1]
        if strategy == "mean":
            fill = old_w.mean(dim=1, keepdim=True).repeat(1, extra, 1, 1)
        elif strategy == "repeat":
            fill = old_w[:, :1].repeat(1, extra, 1, 1)
        elif strategy == "zero":
            fill = torch.zeros(conv.out_channels, extra, *conv.kernel_size)
        else:
            raise ValueError(f"Unknown stem strategy: {strategy}")
        w[:, old_w.shape[1]:] = fill
    if rescale:
        w *= old_w.shape[1] / in_channels

    new_conv.weight.data = w
    if conv.bias is not None:
        new_conv.bias.data = conv.bias.data.clone()
    setattr(parent, name, new_conv)
    return {"changed": True, "strategy": strategy, "rescale": rescale,
            "from_channels": int(old_w.shape[1]), "to_channels": in_channels}


def build_model():
    model = CascadedDeepLab()
    info = {"bn_before": count_bn(model)}
    info["stem"] = expand_stem(model)
    info["bn_replaced_in_encoder"] = (
        replace_bn_with_gn(model.encoder) if ENCODER_GROUPNORM else 0)
    info["encoder_bn_preserved"] = count_bn(model.encoder)
    info["encoder_groupnorm"] = ENCODER_GROUPNORM
    return model, info


@torch.no_grad()
def audit_architecture(model, save="architecture_audit.json"):
    """Measure the model's actual structure. Quote these numbers, not estimates."""
    model.eval()
    encoder = model.encoder
    feats = encoder(torch.zeros(1, IN_CHANNELS, IMG_SIZE, IMG_SIZE))
    stages = [{"index": i, "channels": int(f.shape[1]),
               "spatial": [int(f.shape[2]), int(f.shape[3])],
               "stride": int(IMG_SIZE // f.shape[2])} for i, f in enumerate(feats)]
    names = [type(m).__name__ for m in encoder.modules()]
    dec_params = sum(p.numel() for m in model.decoder_modules() for p in m.parameters())

    audit = {
        "encoder_name": ENCODER_NAME,
        "encoder_weights": ENCODER_WEIGHTS,
        "input": {"channels": IN_CHANNELS, "resolution": IMG_SIZE,
                  "native_resolution": NATIVE_RESOLUTION,
                  "resolution_vs_b0_224": round(IMG_SIZE / 224.0, 2),
                  "native_vs_b0_224": round(NATIVE_RESOLUTION / 224.0, 2)},
        "feature_stages": stages,
        "n_feature_maps_returned": len(feats),
        "n_spatial_reductions": len({s["stride"] for s in stages}) - 1,
        "max_stride": max(s["stride"] for s in stages),
        "encoder_counts": {
            "inverted_residual_blocks": sum(1 for n in names if "InvertedResidual" in n),
            "depthwise_separable_blocks": sum(1 for n in names if "DepthwiseSeparable" in n),
            "squeeze_excite_modules": sum(1 for n in names
                                          if "SqueezeExcite" in n or n == "SEModule"),
            "conv2d_layers": sum(1 for m in encoder.modules() if isinstance(m, nn.Conv2d)),
            "batchnorm_layers": count_bn(encoder),
            "groupnorm_layers": sum(1 for m in encoder.modules()
                                    if isinstance(m, nn.GroupNorm)),
        },
        "decoder": {
            "aspp_rates": list(ATROUS_RATES),
            "aspp_branches": len(ATROUS_RATES) + 2,      # + 1x1 + image pooling
            "decoder_channels": DECODER_CHANNELS,
            "upsample_stages": 2,
            "attention_fusion_modules": sum(1 for m in model.modules()
                                            if isinstance(m, AttentionFusion)),
            "skip_refinement_modules": 2,
            "groupnorm_layers": sum(1 for m in model.decoder_modules()
                                    for mm in m.modules()
                                    if isinstance(mm, nn.GroupNorm)),
            "params": dec_params,
        },
        "params": {
            "encoder_total": sum(p.numel() for p in encoder.parameters()),
            "decoder_total": dec_params,
            "model_total": sum(p.numel() for p in model.parameters()),
            "model_trainable": sum(p.numel() for p in model.parameters()
                                   if p.requires_grad),
        },
    }
    audit["encoder_counts"]["total_mbconv_blocks"] = (
        audit["encoder_counts"]["inverted_residual_blocks"] +
        audit["encoder_counts"]["depthwise_separable_blocks"])

    print("\n--- Architecture audit (measured, not estimated) ---")
    print(f"Encoder: {ENCODER_NAME}, weights={ENCODER_WEIGHTS}")
    print(f"Input: {IN_CHANNELS} channels at {IMG_SIZE}px "
          f"({audit['input']['resolution_vs_b0_224']}x vs B0's 224px; "
          f"B8 native is {NATIVE_RESOLUTION}px = {audit['input']['native_vs_b0_224']}x)")
    print(f"Feature maps: {len(feats)}, strides {[s['stride'] for s in stages]}, "
          f"channels {[s['channels'] for s in stages]}")
    print(f"Spatial reductions: {audit['n_spatial_reductions']} "
          f"(max stride {audit['max_stride']})")
    for k, v in audit["encoder_counts"].items():
        print(f"  encoder {k:>28}: {v}")
    for k in ("aspp_rates", "aspp_branches", "upsample_stages",
              "attention_fusion_modules", "skip_refinement_modules"):
        print(f"  decoder {k:>28}: {audit['decoder'][k]}")
    print(f"  {'encoder params':>36}: {audit['params']['encoder_total'] / 1e6:.1f}M")
    print(f"  {'decoder params':>36}: {audit['params']['decoder_total'] / 1e6:.1f}M")
    print(f"  {'trainable params':>36}: {audit['params']['model_trainable'] / 1e6:.1f}M")
    print("---------------------------------------------------\n")

    with open(save, "w") as f:
        json.dump(audit, f, indent=2)
    print(f"Wrote {save}")
    return audit


# ----------------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------------
class MetricBundle:
    """Accumulates over the whole dataset, computes once — a dataset-level score,
    not a mean of per-batch scores."""

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
        return {"iou": self.iou.compute().item(), "f1": self.f1.compute().item(),
                "precision": self.precision.compute().item(),
                "recall": self.recall.compute().item(),
                "pixel_accuracy": self.acc.compute().item()}


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
# Training
# ----------------------------------------------------------------------------------
def train_model(model, train_loader, val_loader, epochs, device=DEVICE):
    model.to(device)
    criterion = smp.losses.FocalLoss(mode="binary")
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))
    train_metrics = MetricBundle(device)

    history = {"train_loss": [], "val_loss": [], "train_iou": [], "val_iou": []}
    best_iou = -1.0

    for epoch in range(epochs):
        model.train()
        train_metrics.reset()
        total, n = 0.0, 0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type,
                                    enabled=(device.type == "cuda")):
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

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va["loss"])
        history["train_iou"].append(tr["iou"])
        history["val_iou"].append(va["iou"])
        print(f"[{epoch + 1}/{epochs}] train loss {tr_loss:.4f} IoU {tr['iou']:.4f} | "
              f"val loss {va['loss']:.4f} IoU {va['iou']:.4f}")

        if va["iou"] > best_iou:
            best_iou = va["iou"]
            torch.save({"epoch": epoch, "val_iou": best_iou,
                        "state_dict": model.state_dict()}, CKPT_PATH)
            print(f"    new best val IoU {best_iou:.4f} — checkpoint saved")

    print(f"Training complete. Best val IoU: {best_iou:.4f}")
    return history


def load_best(model, path=CKPT_PATH, device=DEVICE):
    """Without this, every downstream number describes the final epoch."""
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    print(f"Loaded checkpoint from epoch {ckpt['epoch'] + 1} "
          f"(val IoU {ckpt['val_iou']:.4f})")
    return model


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


def denorm_rgb(image_tensor):
    """Undo normalization for display; return the first three bands."""
    img = image_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    img = img * 0.5 + 0.5
    img = img[:, :, :3] if img.shape[2] >= 3 else np.repeat(img, 3, axis=2)
    return np.clip(img * 1.5, 0, 1)


@torch.no_grad()
def plot_samples(model, dataset, n=8, save_path="sample_predictions.png", device=DEVICE):
    model.eval()
    idxs = np.random.default_rng(SEED).choice(
        len(dataset), size=min(n, len(dataset)), replace=False)
    fig, axes = plt.subplots(len(idxs), 4, figsize=(18, 4.2 * len(idxs)))
    if len(idxs) == 1:
        axes = np.expand_dims(axes, 0)
    cmap = matplotlib.colors.ListedColormap(["black", "red", "blue"])
    for row, idx in enumerate(idxs):
        image, mask = dataset[int(idx)]
        prob = model(image.unsqueeze(0).to(device)).sigmoid().cpu().numpy()[0, 0]
        pred = (prob > 0.5).astype(np.uint8)
        gt = mask.squeeze().numpy().astype(np.uint8)
        diff = ((pred == 1) & (gt == 0)).astype(np.uint8) + \
               2 * ((pred == 0) & (gt == 1)).astype(np.uint8)
        panels = [(denorm_rgb(image), f"Image (idx {idx})", {}),
                  (gt, "Ground truth", {"cmap": "gray"}),
                  (pred, "Prediction", {"cmap": "gray"}),
                  (diff, "FP red / FN blue", {"cmap": cmap, "vmin": 0, "vmax": 2})]
        for col, (data, title, kw) in enumerate(panels):
            axes[row, col].imshow(data, **kw)
            axes[row, col].set_title(title)
            axes[row, col].axis("off")
    fig.tight_layout(); fig.savefig(save_path, dpi=150); plt.close(fig)


# ----------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description="Deforestation segmentation — EfficientNet-B8 + cascaded decoder")
    p.add_argument("--base_path", required=True, help="Dataset root")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    args = p.parse_args()

    set_seed(SEED)

    model, build_info = build_model()
    print(f"Model: cascaded decoder / {ENCODER_NAME} / "
          f"encoder_groupnorm={ENCODER_GROUPNORM} / stem_init={STEM_INIT}")
    print(f"  BatchNorm: {build_info['bn_before']} before, "
          f"{build_info['bn_replaced_in_encoder']} replaced in encoder, "
          f"{build_info['encoder_bn_preserved']} preserved")
    audit = audit_architecture(model)

    if IMG_SIZE != NATIVE_RESOLUTION:
        print(f"NOTE: training at {IMG_SIZE}px, below B8's native "
              f"{NATIVE_RESOLUTION}px. Report the value used, not the native one.\n")

    (tr_i, tr_m), (va_i, va_m), (te_i, te_m) = build_splits(args.base_path)
    train_ds = DeforestationDataset(tr_i, tr_m, get_train_transforms(), PRELOAD)
    val_ds = DeforestationDataset(va_i, va_m, get_eval_transforms(), PRELOAD)
    test_ds = DeforestationDataset(te_i, te_m, get_eval_transforms(), PRELOAD)

    nw = 0 if PRELOAD else NUM_WORKERS
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=nw, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=nw, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=nw, pin_memory=True)

    history = train_model(model, train_loader, val_loader, args.epochs, DEVICE)
    plot_history(history)

    # Everything below uses the selected checkpoint, not the final epoch.
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

    plot_confusion(model, test_loader, "confusion_matrix_test.png")
    plot_samples(model, test_ds, 8, "sample_predictions_test.png")

    with open("results.json", "w") as f:
        json.dump({
            "config": {
                "encoder": ENCODER_NAME, "encoder_weights": ENCODER_WEIGHTS,
                "decoder": "CascadedDeepLab", "output_stride": OUTPUT_STRIDE,
                "atrous_rates": list(ATROUS_RATES),
                "decoder_channels": DECODER_CHANNELS, "num_groups": NUM_GROUPS,
                "encoder_groupnorm": ENCODER_GROUPNORM,
                "stem_init": STEM_INIT, "stem_rescale": STEM_RESCALE,
                "in_channels": IN_CHANNELS, "img_size": IMG_SIZE,
                "native_resolution": NATIVE_RESOLUTION,
                "loss": "FocalLoss(binary)", "optimizer": "Adam",
                "lr": LEARNING_RATE, "batch_size": args.batch_size,
                "epochs": args.epochs, "seed": SEED,
            },
            "build_info": build_info,
            "architecture_audit": audit,
            "split_sizes": {"train": len(train_ds), "val": len(val_ds),
                            "test": len(test_ds)},
            "validation": val_results,
            "test": test_results,
        }, f, indent=2)
    print("\nWrote results.json")


if __name__ == "__main__":
    main()
