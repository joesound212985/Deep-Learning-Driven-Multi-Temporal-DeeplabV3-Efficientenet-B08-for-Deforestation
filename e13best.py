#!/usr/bin/env python
"""
DeepLabV3+ deforestation segmentation — corrected pipeline.

Fixes relative to the original script:
  1. Held-out test split (train/val/test), with test used only once at the end.
  2. Deterministic evaluation transforms (no flips/rotations/jitter at eval time).
  3. Best checkpoint is reloaded before any evaluation or figure generation.
  4. Dataset-level IoU via metric accumulation, not a mean of per-batch IoUs.
  5. Explicit ASPP dilation rates and output stride.
  6. Normalization matched to the pretrained encoder's expected input range.
  7. GroupNorm confined to the decoder by default, preserving encoder BN statistics.
  8. Seeded splits and reproducible evaluation.
  9. Optional custom decoder (--decoder custom) implementing ASPP + skip refinement +
     progressive upsampling with residuals + attention-based multi-scale fusion.

Usage:
    python train_deforestation.py --base_path /path/to/dataset
    python train_deforestation.py --base_path /path/to/dataset --decoder custom
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
    BinaryAccuracy,
    BinaryF1Score,
    BinaryJaccardIndex,
    BinaryPrecision,
    BinaryRecall,
    ConfusionMatrix,
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
IMG_SIZE = 512
IN_CHANNELS = 4
OUTPUT_STRIDE = 16
ATROUS_RATES = (6, 12, 18)
NUM_GROUPS = 8

ENCODER_NAME = "timm-efficientnet-b8"
ENCODER_WEIGHTS = "advprop"

# advprop weights were trained on inputs scaled to approximately [-1, 1].
# Reflectance is divided by SCALE first, giving [0, 1]; these then map to [-1, 1].
REFLECTANCE_SCALE = 10000.0
NORM_MEAN = (0.5,) * IN_CHANNELS
NORM_STD = (0.5,) * IN_CHANNELS

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
    """4-channel GeoTIFF imagery with binary masks."""

    def __init__(self, image_paths, mask_paths, transform=None, preload=False):
        assert len(image_paths) == len(mask_paths)
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.preload = preload

        if preload:
            self.cache = []
            print(f"Preloading {len(image_paths)} samples...")
            for ip, mp in zip(image_paths, mask_paths):
                self.cache.append(self._read(ip, mp))
            print("Preload complete.")

    @staticmethod
    def _read(image_path, mask_path):
        with rasterio.open(image_path) as f:
            image = f.read().transpose(1, 2, 0).astype(np.float32)
        with rasterio.open(mask_path) as f:
            mask = f.read(1)
        # Scale reflectance to [0, 1] before any geometric or photometric op.
        image = image / REFLECTANCE_SCALE
        # Force a strictly binary mask; guards against 0/255 or stray label values.
        mask = (mask > 0).astype(np.uint8)
        return image, mask

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if self.preload:
            image, mask = self.cache[idx]
        else:
            image, mask = self._read(self.image_paths[idx], self.mask_paths[idx])

        if self.transform is not None:
            out = self.transform(image=image, mask=mask)
            image, mask = out["image"], out["mask"]

        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        return image, mask.float()


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
        A.Normalize(mean=NORM_MEAN, std=NORM_STD, max_pixel_value=1.0),
        ToTensorV2(),
    ])


def get_eval_transforms():
    """Deterministic — validation and test. No augmentation of any kind."""
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=NORM_MEAN, std=NORM_STD, max_pixel_value=1.0),
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

    tv_img, te_img, tv_msk, te_msk = train_test_split(
        image_paths, mask_paths, test_size=0.15, random_state=seed)
    tr_img, va_img, tr_msk, va_msk = train_test_split(
        tv_img, tv_msk, test_size=0.1765, random_state=seed)

    print(f"Split sizes — train {len(tr_img)}, val {len(va_img)}, test {len(te_img)}")
    return (tr_img, tr_msk), (va_img, va_msk), (te_img, te_msk)


# ----------------------------------------------------------------------------------
# Custom decoder (opt-in via --decoder custom)
# ----------------------------------------------------------------------------------
def gn(channels, num_groups=NUM_GROUPS):
    g = num_groups
    while channels % g != 0 and g > 1:
        g -= 1
    return nn.GroupNorm(g, channels)


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, dilation=1):
        super().__init__()
        pad = dilation * (k - 1) // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, padding=pad, dilation=dilation, bias=False),
            gn(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ASPP(nn.Module):
    """Atrous spatial pyramid pooling: 1x1 branch, three dilated 3x3 branches,
    and an image-level pooling branch."""

    def __init__(self, in_ch, out_ch=256, rates=ATROUS_RATES):
        super().__init__()
        self.branches = nn.ModuleList(
            [ConvBlock(in_ch, out_ch, k=1)] +
            [ConvBlock(in_ch, out_ch, k=3, dilation=r) for r in rates]
        )
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.project = nn.Sequential(
            nn.Conv2d(out_ch * (len(rates) + 2), out_ch, 1, bias=False),
            gn(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5),
        )

    def forward(self, x):
        size = x.shape[-2:]
        feats = [b(x) for b in self.branches]
        pooled = F.interpolate(self.pool(x), size=size, mode="bilinear", align_corners=False)
        feats.append(pooled)
        return self.project(torch.cat(feats, dim=1))


class SkipRefine(nn.Module):
    """1x1 projection + GroupNorm + ReLU on an encoder skip connection."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.refine = ConvBlock(in_ch, out_ch, k=1)

    def forward(self, x):
        return self.refine(x)


class AttentionFusion(nn.Module):
    """Gated fusion of two same-resolution feature maps. A global-average-pooled
    descriptor of the concatenation produces per-branch, per-channel weights that
    are softmax-normalized across the two branches."""

    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels * 2, 1),
        )
        self.channels = channels

    def forward(self, deep, shallow):
        w = self.gate(torch.cat([deep, shallow], dim=1))
        w = w.view(w.size(0), 2, self.channels, 1, 1).softmax(dim=1)
        return deep * w[:, 0] + shallow * w[:, 1]


class UpStage(nn.Module):
    """Bilinear x2 upsample, 3x3 conv, residual connection, then attention fusion
    with the refined skip feature."""

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.reduce = ConvBlock(in_ch, out_ch, k=1) if in_ch != out_ch else nn.Identity()
        self.conv = ConvBlock(out_ch, out_ch, k=3)
        self.skip = SkipRefine(skip_ch, out_ch)
        self.fuse = AttentionFusion(out_ch)
        self.post = ConvBlock(out_ch, out_ch, k=3)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.reduce(x)
        x = x + self.conv(x)              # residual
        s = self.skip(skip)
        x = self.fuse(x, s)
        return x + self.post(x)           # residual


class CustomDeepLab(nn.Module):
    """DeepLabV3+ encoder with a cascaded decoder: ASPP -> refined skips ->
    progressive upsampling with residuals -> attention-based fusion."""

    def __init__(self, encoder_name=ENCODER_NAME, encoder_weights=ENCODER_WEIGHTS,
                 in_channels=IN_CHANNELS, classes=1, decoder_ch=256,
                 output_stride=OUTPUT_STRIDE):
        super().__init__()
        self.encoder = smp.encoders.get_encoder(
            encoder_name, in_channels=in_channels, depth=5, weights=encoder_weights,
            output_stride=output_stride,
        )
        ch = self.encoder.out_channels          # (stride 1, 2, 4, 8, 16, 32->16)
        self.aspp = ASPP(ch[-1], decoder_ch)
        self.up1 = UpStage(decoder_ch, ch[3], decoder_ch)        # -> stride 8
        self.up2 = UpStage(decoder_ch, ch[2], decoder_ch // 2)   # -> stride 4
        self.head = nn.Conv2d(decoder_ch // 2, classes, 1)

    def forward(self, x):
        size = x.shape[-2:]
        feats = self.encoder(x)
        y = self.aspp(feats[-1])
        y = self.up1(y, feats[3])
        y = self.up2(y, feats[2])
        y = F.interpolate(y, size=size, mode="bilinear", align_corners=False)
        return self.head(y)


# ----------------------------------------------------------------------------------
# Model construction
# ----------------------------------------------------------------------------------
def replace_bn_with_gn(module, num_groups=NUM_GROUPS):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, gn(child.num_features, num_groups))
        else:
            replace_bn_with_gn(child, num_groups)


def build_model(decoder="baseline", gn_scope="decoder"):
    """
    decoder:  'baseline' (stock smp DeepLabV3+) or 'custom' (the cascaded decoder).
    gn_scope: 'decoder' replaces BatchNorm in the decoder only, preserving the
              encoder's pretrained BN statistics. 'all' replaces everywhere, which
              discards those statistics. 'none' leaves BatchNorm intact.
    """
    if decoder == "custom":
        model = CustomDeepLab()
        if gn_scope == "all":
            replace_bn_with_gn(model)
        # The custom decoder already uses GroupNorm throughout.
        return model

    model = smp.DeepLabV3Plus(
        encoder_name=ENCODER_NAME,
        encoder_weights=ENCODER_WEIGHTS,
        encoder_output_stride=OUTPUT_STRIDE,
        decoder_atrous_rates=ATROUS_RATES,
        in_channels=IN_CHANNELS,
        classes=1,
    )
    if gn_scope == "all":
        replace_bn_with_gn(model)
    elif gn_scope == "decoder":
        replace_bn_with_gn(model.decoder)
        replace_bn_with_gn(model.segmentation_head)
    return model


# ----------------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------------
class MetricBundle:
    """Accumulates over the whole dataset, then computes once. This is a
    dataset-level score, not a mean of per-batch scores."""

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
    total_loss, n_batches = 0.0, 0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        logits = model(images)
        total_loss += criterion(logits, masks).item()
        n_batches += 1
        metrics.update(logits.sigmoid(), masks)
    out = metrics.compute()
    out["loss"] = total_loss / max(n_batches, 1)
    return out


# ----------------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------------
def train_model(model, train_loader, val_loader, epochs=EPOCHS, device=DEVICE):
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
        total_loss, n_batches = 0.0, 0

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

            total_loss += loss.item()
            n_batches += 1
            with torch.no_grad():
                train_metrics.update(logits.detach().float().sigmoid(), masks)

        tr = train_metrics.compute()
        tr_loss = total_loss / max(n_batches, 1)
        va = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va["loss"])
        history["train_iou"].append(tr["iou"])
        history["val_iou"].append(va["iou"])

        print(f"[{epoch + 1}/{epochs}] "
              f"train loss {tr_loss:.4f} IoU {tr['iou']:.4f} | "
              f"val loss {va['loss']:.4f} IoU {va['iou']:.4f}")

        if va["iou"] > best_iou:
            best_iou = va["iou"]
            torch.save({"epoch": epoch, "val_iou": best_iou,
                        "state_dict": model.state_dict()}, CKPT_PATH)
            print(f"    new best val IoU {best_iou:.4f} — checkpoint saved")

    print(f"Training complete. Best val IoU: {best_iou:.4f}")
    return history


def load_best(model, path=CKPT_PATH, device=DEVICE):
    """Restore the selected checkpoint. Without this, every downstream number
    describes the final epoch rather than the selected model."""
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
    a.set_xlabel("Epoch"); a.set_ylabel("Focal loss"); a.set_title("Loss")
    a.legend(); a.grid(True)
    b.plot(epochs, history["train_iou"], label="train")
    b.plot(epochs, history["val_iou"], label="val")
    b.set_xlabel("Epoch"); b.set_ylabel("IoU"); b.set_title("IoU")
    b.legend(); b.grid(True)
    fig.tight_layout(); fig.savefig(save_path, dpi=200); plt.close(fig)


@torch.no_grad()
def plot_confusion(model, loader, save_path="confusion_matrix.png", device=DEVICE):
    model.eval()
    cm_metric = ConfusionMatrix(task="binary", threshold=0.5).to(device)
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        cm_metric.update(model(images).sigmoid(), masks.int())
    cm = cm_metric.compute().cpu().numpy()

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
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
    img = img * np.array(NORM_STD) + np.array(NORM_MEAN)
    img = img[:, :, :3] if img.shape[2] >= 3 else np.repeat(img, 3, axis=2)
    return np.clip(img * 1.5, 0, 1)


@torch.no_grad()
def plot_samples(model, dataset, n=8, save_path="sample_predictions.png",
                 device=DEVICE, seed=SEED):
    model.eval()
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(dataset), size=min(n, len(dataset)), replace=False)
    fig, axes = plt.subplots(len(idxs), 4, figsize=(18, 4.2 * len(idxs)))
    if len(idxs) == 1:
        axes = np.expand_dims(axes, 0)

    for row, idx in enumerate(idxs):
        image, mask = dataset[int(idx)]
        prob = model(image.unsqueeze(0).to(device)).sigmoid().cpu().numpy()[0, 0]
        pred = (prob > 0.5).astype(np.uint8)
        gt = mask.squeeze().numpy().astype(np.uint8)
        diff = ((pred == 1) & (gt == 0)).astype(np.uint8) + \
               2 * ((pred == 0) & (gt == 1)).astype(np.uint8)

        for col, (data, title, kw) in enumerate([
            (denorm_rgb(image), f"Image (idx {idx})", {}),
            (gt, "Ground truth", {"cmap": "gray"}),
            (pred, "Prediction", {"cmap": "gray"}),
            (diff, "FP red / FN blue",
             {"cmap": matplotlib.colors.ListedColormap(["black", "red", "blue"]),
              "vmin": 0, "vmax": 2}),
        ]):
            axes[row, col].imshow(data, **kw)
            axes[row, col].set_title(title)
            axes[row, col].axis("off")

    fig.tight_layout(); fig.savefig(save_path, dpi=150); plt.close(fig)


# ----------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="DeepLabV3+ deforestation segmentation")
    parser.add_argument("--base_path", required=True, help="Dataset root")
    parser.add_argument("--decoder", choices=["baseline", "custom"], default="baseline")
    parser.add_argument("--gn_scope", choices=["none", "decoder", "all"], default="decoder")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--preload", action="store_true",
                        help="Cache the dataset in RAM. Use num_workers=0 with this, "
                             "since each worker forks a full copy.")
    args = parser.parse_args()

    set_seed(SEED)
    (tr_i, tr_m), (va_i, va_m), (te_i, te_m) = build_splits(args.base_path)

    train_ds = DeforestationDataset(tr_i, tr_m, get_train_transforms(), args.preload)
    val_ds = DeforestationDataset(va_i, va_m, get_eval_transforms(), args.preload)
    test_ds = DeforestationDataset(te_i, te_m, get_eval_transforms(), args.preload)

    nw = 0 if args.preload else args.num_workers
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=nw, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=nw, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=nw, pin_memory=True)

    model = build_model(decoder=args.decoder, gn_scope=args.gn_scope)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.decoder} decoder, gn_scope={args.gn_scope}, "
          f"{n_params / 1e6:.1f}M trainable parameters")

    history = train_model(model, train_loader, val_loader,
                          epochs=args.epochs, device=DEVICE)
    plot_history(history)

    # Everything below runs on the selected checkpoint, not the final epoch.
    model = load_best(model)
    criterion = smp.losses.FocalLoss(mode="binary")

    val_results = evaluate(model, val_loader, criterion)
    test_results = evaluate(model, test_loader, criterion)

    print("\nValidation (used for model selection — report as validation):")
    for k, v in val_results.items():
        print(f"  {k:>16}: {v:.4f}")
    print("\nTest (held out, evaluated once):")
    for k, v in test_results.items():
        print(f"  {k:>16}: {v:.4f}")

    plot_confusion(model, test_loader, save_path="confusion_matrix_test.png")
    plot_samples(model, test_ds, n=8, save_path="sample_predictions_test.png")

    with open("results.json", "w") as f:
        json.dump({
            "config": {
                "decoder": args.decoder, "gn_scope": args.gn_scope,
                "encoder": ENCODER_NAME, "encoder_weights": ENCODER_WEIGHTS,
                "output_stride": OUTPUT_STRIDE, "atrous_rates": list(ATROUS_RATES),
                "in_channels": IN_CHANNELS, "img_size": IMG_SIZE,
                "batch_size": args.batch_size, "epochs": args.epochs,
                "lr": LEARNING_RATE, "seed": SEED,
            },
            "split_sizes": {"train": len(train_ds), "val": len(val_ds),
                            "test": len(test_ds)},
            "validation": val_results,
            "test": test_results,
        }, f, indent=2)
    print("\nWrote results.json")


if __name__ == "__main__":
    main()
