#!/usr/bin/env python
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
import rasterio
import matplotlib
matplotlib.use('agg')  # Use a non-GUI backend
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score, precision_recall_curve, roc_curve, auc
from matplotlib.colors import ListedColormap
from torchmetrics.classification import JaccardIndex, ConfusionMatrix
from sklearn.model_selection import train_test_split
import warnings
from rasterio.errors import NotGeoreferencedWarning

# ------------------------------------------------------------------------------------
# Global Constants
# ------------------------------------------------------------------------------------
warnings.simplefilter("ignore", NotGeoreferencedWarning)
torch.cuda.empty_cache()

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 8
EPOCHS = 1000  # Adjust as needed.
LEARNING_RATE = 1e-4
ENCODER_NAME = "timm-efficientnet-b8"
ENCODER_WEIGHTS = "advprop"

# ------------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------------
def brighten_image(image_np, factor=1.5):
    """
    Adjust image brightness for visualization.
    """
    return np.clip(image_np * factor, 0, 1)

def replace_batchnorm_with_groupnorm(module, num_groups=8):
    """
    Recursively replaces all nn.BatchNorm2d layers with nn.GroupNorm.
    Useful for small batch sizes.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            num_channels = child.num_features
            setattr(module, name, nn.GroupNorm(num_groups=num_groups, num_channels=num_channels))
        else:
            replace_batchnorm_with_groupnorm(child, num_groups=num_groups)

# ------------------------------------------------------------------------------------
# compute_iou_manual
# ------------------------------------------------------------------------------------
def compute_iou_manual(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Computes the Intersection over Union (IoU) for a single sample.
    If the target mask is empty (i.e., no fire), returns None.
    """
    if target.sum() == 0:
        return None  # Skip IoU calculation for empty ground truth
    
    # Binarize prediction based on the threshold.
    pred_mask = (pred >= threshold).int()
    target = target.int()
    
    intersection = (pred_mask * target).sum().float()
    union = ((pred_mask + target) >= 1).sum().float()
    
    return (intersection / (union + 1e-6)).item()

# ------------------------------------------------------------------------------------
# Custom Dataset for FireDataset_20m
# ------------------------------------------------------------------------------------
class FireDataset(Dataset):
    def __init__(self, base_path, file_list=None, transform=None, load_in_memory=True):
        """
        Args:
            base_path (str): Path to FireDataset_20m.
            file_list (list, optional): List of file names to use.
            transform: Albumentations transform.
            load_in_memory (bool): If True, preload all images and masks into RAM.
        """
        self.base_path = base_path
        self.transform = transform
        self.load_in_memory = load_in_memory
        self.image_dir = os.path.join(base_path, "training", "image")
        self.mask_dir = os.path.join(base_path, "training", "mask")
        if file_list is None:
            self.file_names = sorted([f for f in os.listdir(self.image_dir) if f.endswith(".tif")])
        else:
            self.file_names = file_list

        # Preload all data if required
        if self.load_in_memory:
            self.data = []
            print("Preloading dataset into memory...")
            for file_name in self.file_names:
                image_path = os.path.join(self.image_dir, file_name)
                mask_path = os.path.join(self.mask_dir, file_name)
                
                # Load image and mask using rasterio
                with rasterio.open(image_path) as src:
                    image = src.read().transpose(1, 2, 0)  # Convert to HWC format.
                with rasterio.open(mask_path) as src:
                    mask = src.read(1)
                
                image = image.astype(np.float32) / 255.0
                mask = mask.astype(np.uint8)
                self.data.append((image, mask))
            print(f"Loaded {len(self.data)} samples into memory.")

    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, idx):
        if self.load_in_memory:
            image, mask = self.data[idx]
        else:
            file_name = self.file_names[idx]
            image_path = os.path.join(self.image_dir, file_name)
            mask_path = os.path.join(self.mask_dir, file_name)
            
            with rasterio.open(image_path) as src:
                image = src.read().transpose(1, 2, 0)
            with rasterio.open(mask_path) as src:
                mask = src.read(1)
            
            image = image.astype(np.float32) / 255.0
            mask = mask.astype(np.uint8)
        
        # If a transformation is provided, apply it.
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]
        else:
            image = ToTensorV2()(image=image)["image"]
            mask = torch.tensor(mask).unsqueeze(0).float()
        
        # Ensure mask has the right shape (C, H, W)
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        return image, mask


# ------------------------------------------------------------------------------------
# Data Augmentation
# ------------------------------------------------------------------------------------
def get_transforms():
    return A.Compose([
        A.Resize(512, 512),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Normalize(mean=0.0, std=1.0, max_pixel_value=1.0),
        ToTensorV2(),
    ])

# ------------------------------------------------------------------------------------
# Visualization Functions
# ------------------------------------------------------------------------------------
def plot_confusion_matrix(confusion_matrix, class_names=None, save_path="confusion_matrix.png"):
    if class_names is None:
        class_names = ['No Fire', 'Fire']
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(confusion_matrix.shape[1]),
        yticks=np.arange(confusion_matrix.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title="Confusion Matrix",
        ylabel="True label",
        xlabel="Predicted label"
    )
    thresh = confusion_matrix.max() / 2.0
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            ax.text(
                j, i, int(confusion_matrix[i, j]),
                ha="center", va="center",
                color="white" if confusion_matrix[i, j] > thresh else "black"
            )
    fig.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

def plot_sample_predictions(model, dataset, num_samples=10, device=DEVICE, save_path="sample_predictions.png"):
    model.eval()
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    for i, idx in enumerate(indices):
        image_tensor, mask_tensor = dataset[idx]
        image_tensor = image_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(image_tensor).sigmoid()
        image_np = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        image_np = brighten_image(image_np, factor=1.5)
        mask_np = mask_tensor.squeeze(0).cpu().numpy()
        pred_np = pred.squeeze(0).squeeze(0).cpu().numpy()
        pred_mask = (pred_np > 0.5).astype(np.uint8)
        
        # Original image
        if image_np.shape[2] >= 3:
            display_image = image_np[:, :, :3]
        else:
            display_image = np.repeat(image_np, 3, axis=2)
        
        axes[i, 0].imshow(display_image)
        axes[i, 0].set_title("Image")
        axes[i, 0].axis('off')
        
        # Ground truth
        axes[i, 1].imshow(mask_np, cmap='gray')
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis('off')
        
        # Prediction
        axes[i, 2].imshow(pred_mask, cmap='gray')
        axes[i, 2].set_title("Prediction")
        axes[i, 2].axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

def plot_iou_over_epochs(train_iou, val_iou, save_path="iou_over_epochs.png"):
    plt.figure(figsize=(8, 6))
    epochs = range(1, len(train_iou) + 1)
    plt.plot(epochs, train_iou, marker='o', label='Train IoU')
    plt.plot(epochs, val_iou, marker='s', label='Val IoU')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.title('IoU Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_loss_over_epochs(train_loss, val_loss, save_path="loss_over_epochs.png"):
    plt.figure(figsize=(8, 6))
    epochs = range(1, len(train_loss) + 1)
    plt.plot(epochs, train_loss, marker='o', label='Train Loss')
    plt.plot(epochs, val_loss, marker='s', label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_precision_recall_roc(all_probs, all_targets, save_path="pr_roc_curves.png"):
    precision, recall, _ = precision_recall_curve(all_targets, all_probs)
    pr_auc = auc(recall, precision)
    fpr, tpr, _ = roc_curve(all_targets, all_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(12, 6))
    
    # Precision-Recall Curve
    plt.subplot(1, 2, 1)
    plt.plot(recall, precision, marker='.', color='b',
             label=f'PR Curve (AUC = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.grid(True)
    
    # ROC Curve
    plt.subplot(1, 2, 2)
    plt.plot(fpr, tpr, marker='.', color='r',
             label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', label='No Skill')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_threshold_sweep_iou(all_probs, all_targets, save_path="threshold_sweep_iou.png"):
    thresholds = np.linspace(0, 1, 100)
    ious = []
    for thresh in thresholds:
        preds = (all_probs >= thresh).astype(int)
        iou = jaccard_score(all_targets, preds, average='binary')
        ious.append(iou)
    plt.figure(figsize=(8,6))
    plt.plot(thresholds, ious, label='IoU')
    plt.xlabel('Threshold')
    plt.ylabel('IoU')
    plt.title('IoU vs. Threshold')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_difference_maps(model, dataset, num_samples=10, device=DEVICE, save_path="difference_maps.png"):
    model.eval()
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    fig, axes = plt.subplots(num_samples, 4, figsize=(20, 5 * num_samples))
    for i, idx in enumerate(indices):
        image_tensor, mask_tensor = dataset[idx]
        with torch.no_grad():
            pred = model(image_tensor.unsqueeze(0).to(device)).sigmoid().cpu().numpy()[0, 0]
        pred_mask = (pred > 0.5).astype(np.uint8)
        gt_mask = mask_tensor.squeeze().cpu().numpy().astype(np.uint8)
        diff_map = np.zeros_like(gt_mask)
        diff_map[(pred_mask == 1) & (gt_mask == 0)] = 1  # False Positive
        diff_map[(pred_mask == 0) & (gt_mask == 1)] = 2  # False Negative
        
        axes[i, 0].imshow(image_tensor.permute(1, 2, 0).cpu().numpy())
        axes[i, 0].set_title("Image")
        axes[i, 0].axis('off')
        axes[i, 1].imshow(gt_mask, cmap='gray')
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis('off')
        axes[i, 2].imshow(pred_mask, cmap='gray')
        axes[i, 2].set_title("Prediction")
        axes[i, 2].axis('off')
        axes[i, 3].imshow(diff_map, cmap=ListedColormap(['black', 'red', 'blue']), vmin=0, vmax=2)
        axes[i, 3].set_title("Difference Map (FP=Red, FN=Blue)")
        axes[i, 3].axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

def plot_predicted_area_histogram(model, dataset, num_samples=10, save_path="predicted_area_histogram.png"):
    model.eval()
    predicted_areas = []
    for idx in range(len(dataset)):
        image_tensor, _ = dataset[idx]
        with torch.no_grad():
            pred = model(image_tensor.unsqueeze(0).to(DEVICE)).sigmoid().cpu().numpy()[0, 0]
        pred_mask = (pred > 0.5).astype(np.uint8)
        predicted_areas.append(pred_mask.sum())
    plt.figure(figsize=(8,6))
    plt.hist(predicted_areas, bins=50, color='blue', alpha=0.7)
    plt.xlabel('Predicted Fire Area (pixels)')
    plt.ylabel('Frequency')
    plt.title('Histogram of Predicted Fire Areas')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_prediction_heatmap(model, dataset, num_samples=10, device=DEVICE, save_path="prediction_heatmap.png"):
    model.eval()
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    fig, axes = plt.subplots(num_samples, 2, figsize=(12, 5 * num_samples))
    for i, idx in enumerate(indices):
        image_tensor, _ = dataset[idx]
        with torch.no_grad():
            pred = model(image_tensor.unsqueeze(0).to(device)).sigmoid().cpu().numpy()
        prob_map = pred[0, 0]
        image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
        image_np = brighten_image(image_np, factor=1.5)
        if image_np.shape[2] > 3:
            image_np = image_np[:, :, :3]
        axes[i, 0].imshow(image_np)
        axes[i, 0].set_title("Original Image")
        axes[i, 0].axis('off')
        heatmap = axes[i, 1].imshow(prob_map, cmap='hot')
        axes[i, 1].set_title("Prediction Heat Map")
        axes[i, 1].axis('off')
        fig.colorbar(heatmap, ax=axes[i, 1])
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

def plot_aggregate_error_heatmap(model, dataloader, device=DEVICE, save_path="aggregate_error_heatmap.png"):
    model.eval()
    error_sum = None
    count = 0
    with torch.no_grad():
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)
            preds = model(images).sigmoid()
            pred_masks = (preds >= 0.5).int()
            errors = (pred_masks != masks.int()).float()
            batch_error = errors.sum(dim=0).cpu().numpy()  # shape: (1, H, W)
            if error_sum is None:
                error_sum = batch_error
            else:
                error_sum += batch_error
            count += images.size(0)
    if error_sum is not None:
        error_heatmap = error_sum[0] / count
        plt.figure(figsize=(8,6))
        plt.imshow(error_heatmap, cmap='hot')
        plt.title("Aggregate Error Heat Map\n(Average misclassification frequency)")
        plt.colorbar(label="Average Error")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

def plot_overlay_predictions(model, dataset, num_samples=10, device=DEVICE, alpha=0.4, save_path="overlay_predictions.png"):
    model.eval()
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    fig, axes = plt.subplots(num_samples, 2, figsize=(14, 5 * num_samples))
    for i, idx in enumerate(indices):
        image_tensor, _ = dataset[idx]
        with torch.no_grad():
            pred = model(image_tensor.unsqueeze(0).to(device)).sigmoid().cpu().numpy()
        pred_mask = (pred[0, 0] > 0.5).astype(np.uint8)
        image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
        image_np = brighten_image(image_np, factor=1.5)
        if image_np.shape[2] > 3:
            image_np = image_np[:, :, :3]
        overlay = np.zeros_like(image_np)
        overlay[..., 0] = pred_mask  # Red channel overlay
        axes[i, 0].imshow(image_np)
        axes[i, 0].set_title("Original Image")
        axes[i, 0].axis('off')
        blended = (1 - alpha) * image_np + alpha * overlay
        axes[i, 1].imshow(blended)
        axes[i, 1].set_title("Overlay (Predicted in Red)")
        axes[i, 1].axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

def plot_overlay_predictions_with_contours(model, dataset, num_samples=10, device=DEVICE, save_path="overlay_with_contours.png"):
    model.eval()
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    fig, axes = plt.subplots(num_samples, 2, figsize=(14, 5 * num_samples))
    for i, idx in enumerate(indices):
        image_tensor, _ = dataset[idx]
        image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
        image_np = brighten_image(image_np, factor=1.5)
        if image_np.shape[2] > 3:
            image_np = image_np[:, :, :3]
        with torch.no_grad():
            pred = model(image_tensor.unsqueeze(0).to(device)).sigmoid().cpu().numpy()
        pred_mask = (pred[0, 0] > 0.5).astype(np.uint8)
        image_uint8 = (image_np * 255).astype(np.uint8)
        contours, _ = cv2.findContours(pred_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        image_contour = image_uint8.copy()
        cv2.drawContours(image_contour, contours, -1, (0, 255, 0), 2)
        image_contour = image_contour.astype(np.float32) / 255.0
        axes[i, 0].imshow(image_np)
        axes[i, 0].set_title("Original Image")
        axes[i, 0].axis('off')
        axes[i, 1].imshow(image_contour)
        axes[i, 1].set_title("Overlay with Prediction Contours")
        axes[i, 1].axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

def plot_fp_fn(model, dataset, num_samples=10, device=DEVICE, save_path="fp_fn_overlay.png"):
    diff_cmap = ListedColormap(["black", "red", "blue"])  # 0=BG, 1=FP, 2=FN
    model.eval()
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    for i, idx in enumerate(indices):
        image_tensor, mask_tensor = dataset[idx]
        with torch.no_grad():
            pred = model(image_tensor.unsqueeze(0).to(device)).sigmoid().cpu().numpy()
        pred_mask = (pred[0, 0] > 0.5).astype(np.uint8)
        gt_mask = mask_tensor.squeeze().cpu().numpy().astype(np.uint8)
        fp = np.logical_and(pred_mask == 1, gt_mask == 0).astype(np.uint8)
        fn = np.logical_and(pred_mask == 0, gt_mask == 1).astype(np.uint8)
        image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
        image_np = brighten_image(image_np, factor=1.5)
        if image_np.shape[2] > 3:
            image_np = image_np[:, :, :3]
        
        # Original image
        axes[i, 0].imshow(image_np)
        axes[i, 0].set_title("Original Image")
        axes[i, 0].axis('off')
        
        # FP=red, FN=blue
        overlay = np.zeros_like(image_np)
        overlay[..., 0] = fp
        overlay[..., 2] = fn
        blended = 0.6 * overlay + 0.4 * image_np
        axes[i, 1].imshow(blended)
        axes[i, 1].set_title("FP=Red, FN=Blue")
        axes[i, 1].axis('off')
        
        diff_map = fp + 2 * fn
        axes[i, 2].imshow(diff_map, cmap=diff_cmap, vmin=0, vmax=2)
        axes[i, 2].set_title("Difference Map (FP=1, FN=2)")
        axes[i, 2].axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

def plot_predicted_vs_ground_truth_area_scatter(model, dataset, device=DEVICE, save_path="pred_vs_gt_scatter.png"):
    model.eval()
    pred_fractions = []
    gt_fractions = []
    for idx in range(len(dataset)):
        image_tensor, mask_tensor = dataset[idx]
        h, w = image_tensor.shape[1], image_tensor.shape[2]
        gt = mask_tensor.squeeze().cpu().numpy()
        gt_fraction = gt.sum() / (h * w)
        gt_fractions.append(gt_fraction)
        with torch.no_grad():
            pred = model(image_tensor.unsqueeze(0).to(device)).sigmoid().cpu().numpy()
        pred_mask = (pred[0, 0] > 0.5).astype(np.uint8)
        pred_fraction = pred_mask.sum() / (h * w)
        pred_fractions.append(pred_fraction)
    plt.figure(figsize=(8,6))
    plt.scatter(gt_fractions, pred_fractions, alpha=0.7, color='purple')
    plt.plot([0, 1], [0, 1], 'r--', label="Ideal")
    plt.xlabel("Ground Truth Fraction")
    plt.ylabel("Predicted Fraction")
    plt.title("Predicted vs. Ground Truth Fire Fraction per Sample")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_prediction_probability_histogram(all_probs, bins=50, save_path="prediction_probability_histogram.png"):
    plt.figure(figsize=(8,6))
    plt.hist(all_probs, bins=bins, color='orange', alpha=0.75)
    plt.xlabel("Predicted Probability")
    plt.ylabel("Frequency")
    plt.title("Histogram of Pixel-wise Predicted Probabilities")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_iou_bar_with_error(model, dataset, device=DEVICE, save_path="iou_bar_with_error.png"):
    model.eval()
    iou_metric = JaccardIndex(task="binary", threshold=0.5).to(device)
    ious = []
    for idx in range(len(dataset)):
        image_tensor, mask_tensor = dataset[idx]
        with torch.no_grad():
            pred = model(image_tensor.unsqueeze(0).to(device)).sigmoid()
        sample_iou = iou_metric(pred, mask_tensor.unsqueeze(0).int().to(device)).item()
        ious.append(sample_iou)
    mean_iou = np.mean(ious)
    std_iou = np.std(ious)
    plt.figure(figsize=(6,6))
    plt.bar(["IoU"], [mean_iou], yerr=[std_iou], capsize=10, color='teal')
    plt.ylim(0, 1)
    plt.title("Mean IoU with Standard Deviation")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def compute_metrics_per_sample(model, dataset, threshold=0.5, device=DEVICE):
    """
    Computes per-sample metrics (IoU, precision, recall, F1) for each sample in the dataset.
    Samples with an empty ground truth mask (i.e. no fire) are skipped.
    """
    model.eval()
    metrics_list = []
    iou_metric = JaccardIndex(task="binary", threshold=threshold).to(device)
    
    for idx in range(len(dataset)):
        image_tensor, mask_tensor = dataset[idx]
        
        # Skip samples with no fire in ground truth
        if mask_tensor.sum().item() == 0:
            continue
        
        with torch.no_grad():
            pred = model(image_tensor.unsqueeze(0).to(device)).sigmoid()
        
        # Flatten predictions & GT for metric calculations
        pred_mask = (pred >= threshold).int().cpu().numpy().flatten()
        gt_mask = mask_tensor.cpu().numpy().flatten().astype(np.int32)
        
        sample_iou = iou_metric(pred, mask_tensor.unsqueeze(0).int().to(device)).item()
        sample_precision = precision_score(gt_mask, pred_mask, zero_division=0)
        sample_recall = recall_score(gt_mask, pred_mask, zero_division=0)
        sample_f1 = f1_score(gt_mask, pred_mask, zero_division=0)
        
        metrics_list.append({
            'index': idx,
            'iou': sample_iou,
            'precision': sample_precision,
            'recall': sample_recall,
            'f1': sample_f1
        })
    
    return metrics_list

def log_worst_samples(model, dataset, k=5, threshold=0.5, device=DEVICE):
    metrics_list = compute_metrics_per_sample(model, dataset, threshold, device)
    sorted_metrics = sorted(metrics_list, key=lambda x: x['iou'])
    print("Worst performing samples (by IoU):")
    for entry in sorted_metrics[:k]:
        print(
            f"Sample {entry['index']}: "
            f"IoU={entry['iou']:.4f}, Precision={entry['precision']:.4f}, "
            f"Recall={entry['recall']:.4f}, F1={entry['f1']:.4f}"
        )

def plot_worst_samples(model, dataset, k=5, threshold=0.5, device=DEVICE, save_path="worst_samples.png"):
    diff_cmap = ListedColormap(["black", "red", "blue"])  # 0=BG, 1=FP, 2=FN
    metrics_list = compute_metrics_per_sample(model, dataset, threshold, device)
    sorted_metrics = sorted(metrics_list, key=lambda x: x['iou'])
    worst_samples = sorted_metrics[:k]
    fig, axes = plt.subplots(k, 4, figsize=(20, 5 * k))
    if k == 1:
        axes = np.expand_dims(axes, axis=0)
    model.eval()
    for i, entry in enumerate(worst_samples):
        idx = entry['index']
        image_tensor, mask_tensor = dataset[idx]
        with torch.no_grad():
            pred = model(image_tensor.unsqueeze(0).to(device)).sigmoid().cpu().numpy()[0, 0]
        pred_mask = (pred >= threshold).astype(np.uint8)
        image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
        image_np = brighten_image(image_np, factor=1.5)
        if image_np.shape[2] > 3:
            image_np = image_np[:, :, :3]
        image_np = np.clip(image_np, 0, 1)
        gt_mask = mask_tensor.squeeze().cpu().numpy().astype(np.uint8)
        fp = ((pred_mask == 1) & (gt_mask == 0)).astype(np.uint8)
        fn = ((pred_mask == 0) & (gt_mask == 1)).astype(np.uint8)
        diff_map = fp + 2 * fn
        
        # Column 0: Original image
        axes[i, 0].imshow(image_np)
        axes[i, 0].set_title(f"Image (idx={idx})")
        axes[i, 0].axis('off')
        
        # Column 1: Ground truth
        axes[i, 1].imshow(gt_mask, cmap='gray')
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis('off')
        
        # Column 2: Prediction
        axes[i, 2].imshow(pred_mask, cmap='gray')
        axes[i, 2].set_title(f"Prediction\nIoU={entry['iou']:.3f}")
        axes[i, 2].axis('off')
        
        # Column 3: FP=red, FN=blue
        axes[i, 3].imshow(diff_map, cmap=diff_cmap, vmin=0, vmax=2)
        axes[i, 3].set_title("FP=red, FN=blue")
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

def plot_best_samples(model, dataset, k=5, threshold=0.5, device=DEVICE, save_path="best_samples.png"):
    diff_cmap = ListedColormap(["black", "red", "blue"])
    metrics_list = compute_metrics_per_sample(model, dataset, threshold, device)
    sorted_metrics = sorted(metrics_list, key=lambda x: x['iou'], reverse=True)
    best_samples = sorted_metrics[:k]
    fig, axes = plt.subplots(k, 4, figsize=(20, 5 * k))
    if k == 1:
        axes = np.expand_dims(axes, axis=0)
    model.eval()
    for i, entry in enumerate(best_samples):
        idx = entry['index']
        image_tensor, mask_tensor = dataset[idx]
        with torch.no_grad():
            pred = model(image_tensor.unsqueeze(0).to(device)).sigmoid().cpu().numpy()[0, 0]
        pred_mask = (pred >= threshold).astype(np.uint8)
        image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
        image_np = brighten_image(image_np, factor=1.5)
        if image_np.shape[2] > 3:
            image_np = image_np[:, :, :3]
        image_np = np.clip(image_np, 0, 1)
        gt_mask = mask_tensor.squeeze().cpu().numpy().astype(np.uint8)
        fp = ((pred_mask == 1) & (gt_mask == 0)).astype(np.uint8)
        fn = ((pred_mask == 0) & (gt_mask == 1)).astype(np.uint8)
        diff_map = fp + 2 * fn
        
        # Column 0: Original image
        axes[i, 0].imshow(image_np)
        axes[i, 0].set_title(f"Image (idx={idx})")
        axes[i, 0].axis('off')
        
        # Column 1: Ground Truth
        axes[i, 1].imshow(gt_mask, cmap='gray')
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis('off')
        
        # Column 2: Prediction
        axes[i, 2].imshow(pred_mask, cmap='gray')
        axes[i, 2].set_title(f"Prediction\nIoU={entry['iou']:.3f}")
        axes[i, 2].axis('off')
        
        # Column 3: FP=red, FN=blue
        axes[i, 3].imshow(diff_map, cmap=diff_cmap, vmin=0, vmax=2)
        axes[i, 3].set_title("FP=red, FN=blue")
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

def compute_probs_and_targets(model, dataloader):
    device = next(model.parameters()).device
    model.eval()
    all_probs = []
    all_targets = []
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            preds = model(images).sigmoid().cpu().numpy()  # shape: (B, 1, H, W)
            trues = masks.cpu().numpy()                    # shape: (B, 1, H, W)
            all_probs.extend(preds.flatten())
            all_targets.extend(trues.flatten())
    return np.array(all_probs), np.array(all_targets)

def compute_pixel_accuracy(model, dataloader, threshold=0.5):
    model.eval()
    total_correct = 0
    total_pixels = 0
    with torch.no_grad():
        for images, masks in dataloader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            preds = model(images).sigmoid()
            pred_masks = (preds >= threshold).int()
            correct = (pred_masks == masks.int()).sum().item()
            total = masks.numel()
            total_correct += correct
            total_pixels += total
    return total_correct / total_pixels if total_pixels > 0 else 0

# ------------------------------------------------------------------------------------
# Function to compute separate IoU for empty vs. non-empty images
# ------------------------------------------------------------------------------------
def compute_separate_iou(model, dataloader, threshold=0.5):
    """
    Computes separate IoU metrics for images with fire (non-empty ground truth)
    and images without fire (empty ground truth). For empty images, if the prediction
    is also empty, the IoU is defined as 1.0, else 0.0.
    
    Returns:
        (list, list): non_empty_ious, empty_ious
    """
    model.eval()
    non_empty_ious = []
    empty_ious = []
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            preds = model(images).sigmoid()
            for i in range(images.size(0)):
                gt = masks[i]
                pred = preds[i].unsqueeze(0)  # shape: (1, 1, H, W)
                
                if gt.sum() > 0:
                    iou = compute_iou_manual(pred, gt, threshold)
                    if iou is not None:
                        non_empty_ious.append(iou)
                else:
                    # If ground truth is empty
                    pred_mask = (pred >= threshold).int()
                    if pred_mask.sum() == 0:
                        empty_ious.append(1.0)
                    else:
                        empty_ious.append(0.0)
    return non_empty_ious, empty_ious

# ------------------------------------------------------------------------------------
# Training Function
# ------------------------------------------------------------------------------------
def train_model(model, train_loader, val_loader):
    model.to(DEVICE)
    criterion = smp.losses.FocalLoss(mode='binary')
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Updated usage of GradScaler and autocast to avoid future warnings
    scaler = torch.amp.GradScaler(enabled=(DEVICE.type == "cuda"))
    
    best_iou = 0.0
    train_loss_history = []
    val_loss_history = []
    train_iou_history = []
    val_iou_history = []
    
    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0.0
        total_train_iou = 0.0
        valid_train_samples = 0
        
        for images, masks in train_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            
            with torch.amp.autocast(device_type="cuda", enabled=(DEVICE.type == "cuda")):
                outputs = model(images)
                loss = criterion(outputs, masks)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_train_loss += loss.item()
            
            # Compute IoU for non-empty ground truth samples
            for i in range(images.size(0)):
                if masks[i].sum() > 0:
                    sample_iou = compute_iou_manual(outputs[i].unsqueeze(0).sigmoid(),
                                                    masks[i].unsqueeze(0),
                                                    threshold=0.5)
                    if sample_iou is not None:
                        total_train_iou += sample_iou
                        valid_train_samples += 1
        
        avg_train_loss = total_train_loss / len(train_loader)
        avg_train_iou = total_train_iou / valid_train_samples if valid_train_samples > 0 else 0.0
        
        # Validation loop
        model.eval()
        total_val_loss = 0.0
        total_val_iou = 0.0
        valid_val_samples = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, masks)
                total_val_loss += loss.item()
                
                for i in range(images.size(0)):
                    if masks[i].sum() > 0:
                        sample_iou = compute_iou_manual(outputs[i].unsqueeze(0).sigmoid(),
                                                        masks[i].unsqueeze(0),
                                                        threshold=0.5)
                        if sample_iou is not None:
                            total_val_iou += sample_iou
                            valid_val_samples += 1
        
        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_iou = total_val_iou / valid_val_samples if valid_val_samples > 0 else 0.0
        
        train_loss_history.append(avg_train_loss)
        val_loss_history.append(avg_val_loss)
        train_iou_history.append(avg_train_iou)
        val_iou_history.append(avg_val_iou)
        
        print(f"[Epoch {epoch+1}/{EPOCHS}] "
              f"Train Loss: {avg_train_loss:.4f}, Train IoU(non-empty): {avg_train_iou:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, Val IoU(non-empty): {avg_val_iou:.4f}")
        
        if avg_val_iou > best_iou:
            best_iou = avg_val_iou
            torch.save(model.state_dict(), "best_model_deeplabv3plus_false_rgb.pth")
            print(f"  New Best IoU: {best_iou:.4f} - model saved.")
    
    print(f"Training Complete. Best Val IoU (non-empty): {best_iou:.4f}")
    
    # Plot histories
    plot_iou_over_epochs(train_iou_history, val_iou_history)
    plot_loss_over_epochs(train_loss_history, val_loss_history)
    
    # Compute & plot confusion matrix
    confmat_metric = ConfusionMatrix(task="binary", threshold=0.5).to(DEVICE)
    model.eval()
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            outputs = model(images).sigmoid()
            confmat_metric.update(outputs, masks.int())
    cm = confmat_metric.compute().cpu().numpy()
    plot_confusion_matrix(cm, save_path="confusion_matrix.png")
    
    return train_loss_history, val_loss_history, train_iou_history, val_iou_history

# ------------------------------------------------------------------------------------
# Main Function
# ------------------------------------------------------------------------------------
def main(base_path):
    # List all .tif files in the training/image directory
    image_dir = os.path.join(base_path, "training", "image")
    all_files = sorted([
        f for f in os.listdir(image_dir) 
        if f.endswith(".tif")
    ])
    
    # Split into train (~70%), validation (~15%), and test (~15%)
    train_val_files, test_files = train_test_split(all_files, test_size=0.15, random_state=42)
    train_files, val_files = train_test_split(train_val_files, test_size=0.1765, random_state=42)
    
    # Create datasets
    train_dataset = FireDataset(base_path, file_list=train_files, transform=get_transforms())
    val_dataset = FireDataset(base_path, file_list=val_files, transform=get_transforms())
    test_dataset = FireDataset(base_path, file_list=test_files, transform=get_transforms())
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Model
    model = smp.DeepLabV3Plus(
        encoder_name=ENCODER_NAME,
        encoder_weights=ENCODER_WEIGHTS,
        in_channels=3,
        classes=1
    )
    replace_batchnorm_with_groupnorm(model, num_groups=8)
    
    # Train
    train_loss_history, val_loss_history, train_iou_history, val_iou_history = train_model(model, train_loader, val_loader)
    
    # Post-training evaluations & plots
    log_worst_samples(model, val_dataset, k=5, threshold=0.5, device=DEVICE)
    plot_sample_predictions(model, val_dataset, num_samples=10)
    
    all_probs, all_targets = compute_probs_and_targets(model, val_loader)
    plot_precision_recall_roc(all_probs, all_targets)
    plot_threshold_sweep_iou(all_probs, all_targets)
    plot_difference_maps(model, val_dataset, num_samples=10)
    plot_predicted_area_histogram(model, val_dataset, num_samples=10)
    plot_prediction_heatmap(model, val_dataset, num_samples=10, device=DEVICE, save_path="prediction_heatmap.png")
    plot_aggregate_error_heatmap(model, val_loader, device=DEVICE, save_path="aggregate_error_heatmap.png")
    plot_overlay_predictions(model, val_dataset, num_samples=10, device=DEVICE, alpha=0.4, save_path="overlay_predictions.png")
    plot_overlay_predictions_with_contours(model, val_dataset, num_samples=10, device=DEVICE, save_path="overlay_with_contours.png")
    plot_fp_fn(model, val_dataset, num_samples=10, device=DEVICE, save_path="fp_fn_overlay.png")
    plot_predicted_vs_ground_truth_area_scatter(model, val_dataset, device=DEVICE, save_path="pred_vs_gt_scatter.png")
    plot_prediction_probability_histogram(all_probs, bins=50, save_path="prediction_probability_histogram.png")
    plot_iou_bar_with_error(model, val_dataset, device=DEVICE, save_path="iou_bar_with_error.png")
    
    pa = compute_pixel_accuracy(model, val_loader, threshold=0.5)
    print(f"Pixel Accuracy (PA): {pa:.4f}")
    
    plot_worst_samples(model, val_dataset, k=5, threshold=0.5, device=DEVICE, save_path="worst_samples.png")
    plot_best_samples(model, val_dataset, k=5, threshold=0.5, device=DEVICE, save_path="best_samples.png")
    
    # Separate IoU for empty vs. non-empty images
    non_empty_ious, empty_ious = compute_separate_iou(model, val_loader, threshold=0.5)
    if non_empty_ious:
        avg_non_empty_iou = sum(non_empty_ious) / len(non_empty_ious)
        print(f"Average IoU on images with fire (non-empty): {avg_non_empty_iou:.4f}")
    else:
        print("No images with fire in the validation set.")
    
    if empty_ious:
        avg_empty_iou = sum(empty_ious) / len(empty_ious)
        print(f"Average IoU on empty images: {avg_empty_iou:.4f}")
    else:
        print("No empty images in the validation set.")

# ------------------------------------------------------------------------------------
# Entry Point
# ------------------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Train and evaluate DeepLabV3+ on FireDataset_20m."
    )
    parser.add_argument("--base_path", type=str, required=True,
                        help="Path to the FireDataset_20m directory")
    args = parser.parse_args()
    main(args.base_path)
