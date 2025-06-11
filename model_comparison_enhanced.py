import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from torchmetrics.classification import JaccardIndex, ConfusionMatrix
import rasterio
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score, precision_recall_curve, roc_curve, auc
from matplotlib.colors import ListedColormap
import timm
import seaborn as sns
from pathlib import Path

# ------------------------------------------------------------------------------------
# Global Constants
# ------------------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 5
IMAGE_SIZE = 448
ENCODER_NAME = "se_resnext101_32x4d"
ENCODER_WEIGHTS = "imagenet"

# ------------------------------------------------------------------------------------
# Model Definitions (from original script)
# ------------------------------------------------------------------------------------
class HRNet(nn.Module):
    def __init__(self, in_channels=4, num_classes=1):
        super(HRNet, self).__init__()
        self.backbone = timm.create_model('hrnet_w48', pretrained=True, features_only=True, in_chans=in_channels)
        
        with torch.no_grad():
            dummy_input = torch.randn(1, in_channels, IMAGE_SIZE, IMAGE_SIZE)
            features = self.backbone(dummy_input)
            self.feature_dims = [f.shape[1] for f in features]
        
        self.fusion = nn.ModuleList([
            nn.Conv2d(dim, 256, 3, padding=1) for dim in self.feature_dims
        ])
        
        self.classifier = nn.Sequential(
            nn.Conv2d(256 * len(self.feature_dims), 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, num_classes, 1)
        )
        
    def forward(self, x):
        input_size = x.shape[-2:]
        features = self.backbone(x)
        target_size = features[0].shape[-2:]
        fused_features = []
        
        for i, feat in enumerate(features):
            if feat.shape[-2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            fused_features.append(self.fusion[i](feat))
        
        x = torch.cat(fused_features, dim=1)
        x = self.classifier(x)
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)
        return x

class JPU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(JPU, self).__init__()
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels[-1], out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels[-2], out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels[-3], out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels[-4], out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.dilation1 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, dilation=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.dilation2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=2, dilation=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.dilation3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=4, dilation=4),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.dilation4 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=8, dilation=8),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, features):
        c2, c3, c4, c5 = features
        size = c2.shape[-2:]
        
        c5 = self.conv5(c5)
        c5 = F.interpolate(c5, size=size, mode='bilinear', align_corners=False)
        
        c4 = self.conv4(c4)
        c4 = F.interpolate(c4, size=size, mode='bilinear', align_corners=False)
        
        c3 = self.conv3(c3)
        c3 = F.interpolate(c3, size=size, mode='bilinear', align_corners=False)
        
        c2 = self.conv2(c2)
        
        fused = c2 + c3 + c4 + c5
        
        out1 = self.dilation1(fused)
        out2 = self.dilation2(fused)
        out3 = self.dilation3(fused)
        out4 = self.dilation4(fused)
        
        return out1 + out2 + out3 + out4

class FastFCN(nn.Module):
    def __init__(self, in_channels=4, num_classes=1, backbone='resnet50'):
        super(FastFCN, self).__init__()
        
        self.backbone = timm.create_model(backbone, pretrained=True, features_only=True, in_chans=in_channels)
        
        with torch.no_grad():
            dummy_input = torch.randn(1, in_channels, IMAGE_SIZE, IMAGE_SIZE)
            features = self.backbone(dummy_input)
            self.feature_dims = [f.shape[1] for f in features]
        
        self.jpu = JPU(self.feature_dims[-4:], 512)
        
        self.classifier = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_classes, 1)
        )
        
    def forward(self, x):
        input_size = x.shape[-2:]
        features = self.backbone(x)
        x = self.jpu(features[-4:])
        x = self.classifier(x)
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)
        return x

# ------------------------------------------------------------------------------------
# Dataset and Transforms
# ------------------------------------------------------------------------------------
class DeforestationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        with rasterio.open(image_path) as img_file:
            image = img_file.read().transpose(1, 2, 0)

        with rasterio.open(mask_path) as mask_file:
            mask = mask_file.read(1)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]

        return image, mask.unsqueeze(0).float()

def get_transforms():
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Lambda(image=lambda x, **kwargs: (x / 10000.0).astype(np.float32)),
        A.Normalize(mean=0, std=1, max_pixel_value=1.0),
        ToTensorV2(),
    ])

def brighten_image(image_np, factor=1.5):
    return np.clip(image_np * factor, 0, 1)

def replace_batchnorm_with_groupnorm(module, num_groups=8):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            num_channels = child.num_features
            setattr(module, name, nn.GroupNorm(num_groups=num_groups, num_channels=num_channels))
        else:
            replace_batchnorm_with_groupnorm(child, num_groups=num_groups)

# ------------------------------------------------------------------------------------
# Enhanced Comparison Visualization Functions
# ------------------------------------------------------------------------------------
def plot_side_by_side_comparison(models, model_names, dataset, num_samples=5, device=DEVICE, save_path="side_by_side_comparison.png"):
    """Enhanced side-by-side comparison showing original, GT, and all model predictions"""
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    # Set up the plot grid: Original, GT, Model1, Model2, Model3, Difference Map
    fig, axes = plt.subplots(num_samples, 6, figsize=(24, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    # Difference colormap
    diff_cmap = ListedColormap(['black', 'red', 'blue', 'yellow'])  # BG, FP, FN, Overlap
    
    for model in models:
        model.eval()
    
    for i, idx in enumerate(indices):
        image_tensor, mask_tensor = dataset[idx]
        image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
        image_np = brighten_image(image_np, factor=1.5)
        if image_np.shape[2] > 3:
            image_np = image_np[:, :, :3]
        
        gt_mask = mask_tensor.squeeze().cpu().numpy().astype(np.uint8)
        
        # Get predictions from all models
        predictions = []
        with torch.no_grad():
            for model in models:
                pred = model(image_tensor.unsqueeze(0).to(device)).sigmoid().cpu().numpy()[0, 0]
                pred_mask = (pred > 0.5).astype(np.uint8)
                predictions.append(pred_mask)
        
        # Original image
        axes[i, 0].imshow(image_np)
        axes[i, 0].set_title(f"Original Image (Sample {idx})")
        axes[i, 0].axis('off')
        
        # Ground truth
        axes[i, 1].imshow(gt_mask, cmap='gray', vmin=0, vmax=1)
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis('off')
        
        # Model predictions
        for j, (pred_mask, model_name) in enumerate(zip(predictions, model_names)):
            axes[i, j+2].imshow(pred_mask, cmap='gray', vmin=0, vmax=1)
            
            # Calculate IoU for this sample
            intersection = np.logical_and(pred_mask, gt_mask).sum()
            union = np.logical_or(pred_mask, gt_mask).sum()
            iou = intersection / union if union > 0 else 0
            
            axes[i, j+2].set_title(f"{model_name}\nIoU: {iou:.3f}")
            axes[i, j+2].axis('off')
        
        # Create comprehensive difference map showing all models
        diff_map = np.zeros_like(gt_mask)
        for j, pred_mask in enumerate(predictions):
            fp = np.logical_and(pred_mask == 1, gt_mask == 0)
            fn = np.logical_and(pred_mask == 0, gt_mask == 1)
            diff_map[fp] = 1  # False Positive (Red)
            diff_map[fn] = 2  # False Negative (Blue)
        
        axes[i, 5].imshow(diff_map, cmap=diff_cmap, vmin=0, vmax=3)
        axes[i, 5].set_title("Aggregate Errors\n(FP=Red, FN=Blue)")
        axes[i, 5].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Side-by-side comparison saved to {save_path}")

def plot_model_agreement_analysis(models, model_names, dataset, num_samples=5, device=DEVICE, save_path="model_agreement_analysis.png"):
    """Analyze where models agree and disagree"""
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for model in models:
        model.eval()
    
    for i, idx in enumerate(indices):
        image_tensor, mask_tensor = dataset[idx]
        image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
        image_np = brighten_image(image_np, factor=1.5)
        if image_np.shape[2] > 3:
            image_np = image_np[:, :, :3]
        
        gt_mask = mask_tensor.squeeze().cpu().numpy().astype(np.uint8)
        
        # Get predictions from all models
        predictions = []
        with torch.no_grad():
            for model in models:
                pred = model(image_tensor.unsqueeze(0).to(device)).sigmoid().cpu().numpy()[0, 0]
                pred_mask = (pred > 0.5).astype(np.uint8)
                predictions.append(pred_mask)
        
        # Original image
        axes[i, 0].imshow(image_np)
        axes[i, 0].set_title(f"Original (Sample {idx})")
        axes[i, 0].axis('off')
        
        # Ground truth
        axes[i, 1].imshow(gt_mask, cmap='gray')
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis('off')
        
        # Agreement map (how many models agree on deforestation)
        agreement_map = np.sum(predictions, axis=0)
        axes[i, 2].imshow(agreement_map, cmap='viridis', vmin=0, vmax=3)
        axes[i, 2].set_title("Model Agreement\n(0=None, 3=All)")
        axes[i, 2].axis('off')
        
        # Uncertainty map (where models disagree most)
        uncertainty_map = np.std(predictions, axis=0)
        axes[i, 3].imshow(uncertainty_map, cmap='hot')
        axes[i, 3].set_title("Model Uncertainty\n(High=Disagreement)")
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Model agreement analysis saved to {save_path}")

def plot_error_pattern_analysis(models, model_names, dataset, device=DEVICE, save_path="error_pattern_analysis.png"):
    """Analyze error patterns across different models"""
    
    # Collect error statistics
    error_stats = {name: {'fp': [], 'fn': [], 'tp': [], 'tn': []} for name in model_names}
    
    for model in models:
        model.eval()
    
    with torch.no_grad():
        for idx in range(min(len(dataset), 50)):  # Limit to 50 samples for efficiency
            image_tensor, mask_tensor = dataset[idx]
            gt_mask = mask_tensor.squeeze().cpu().numpy().astype(np.uint8)
            
            for model, model_name in zip(models, model_names):
                pred = model(image_tensor.unsqueeze(0).to(device)).sigmoid().cpu().numpy()[0, 0]
                pred_mask = (pred > 0.5).astype(np.uint8)
                
                # Calculate confusion matrix components
                tp = np.logical_and(pred_mask == 1, gt_mask == 1).sum()
                tn = np.logical_and(pred_mask == 0, gt_mask == 0).sum()
                fp = np.logical_and(pred_mask == 1, gt_mask == 0).sum()
                fn = np.logical_and(pred_mask == 0, gt_mask == 1).sum()
                
                error_stats[model_name]['tp'].append(tp)
                error_stats[model_name]['tn'].append(tn)
                error_stats[model_name]['fp'].append(fp)
                error_stats[model_name]['fn'].append(fn)
    
    # Create error pattern visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # False Positive distribution
    fp_data = [error_stats[name]['fp'] for name in model_names]
    axes[0, 0].boxplot(fp_data, labels=model_names)
    axes[0, 0].set_title('False Positive Distribution')
    axes[0, 0].set_ylabel('FP Count')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # False Negative distribution
    fn_data = [error_stats[name]['fn'] for name in model_names]
    axes[0, 1].boxplot(fn_data, labels=model_names)
    axes[0, 1].set_title('False Negative Distribution')
    axes[0, 1].set_ylabel('FN Count')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Precision comparison
    precisions = []
    for name in model_names:
        tp_total = sum(error_stats[name]['tp'])
        fp_total = sum(error_stats[name]['fp'])
        precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0
        precisions.append(precision)
    
    axes[1, 0].bar(model_names, precisions, color=['skyblue', 'lightcoral', 'lightgreen'])
    axes[1, 0].set_title('Model Precision Comparison')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Recall comparison
    recalls = []
    for name in model_names:
        tp_total = sum(error_stats[name]['tp'])
        fn_total = sum(error_stats[name]['fn'])
        recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0
        recalls.append(recall)
    
    axes[1, 1].bar(model_names, recalls, color=['skyblue', 'lightcoral', 'lightgreen'])
    axes[1, 1].set_title('Model Recall Comparison')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Error pattern analysis saved to {save_path}")

def plot_prediction_confidence_heatmaps(models, model_names, dataset, num_samples=3, device=DEVICE, save_path="confidence_heatmaps.png"):
    """Show prediction confidence heatmaps for each model"""
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    fig, axes = plt.subplots(num_samples, len(models) + 1, figsize=(5 * (len(models) + 1), 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for model in models:
        model.eval()
    
    for i, idx in enumerate(indices):
        image_tensor, mask_tensor = dataset[idx]
        image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
        image_np = brighten_image(image_np, factor=1.5)
        if image_np.shape[2] > 3:
            image_np = image_np[:, :, :3]
        
        # Original image
        axes[i, 0].imshow(image_np)
        axes[i, 0].set_title(f"Original (Sample {idx})")
        axes[i, 0].axis('off')
        
        # Confidence heatmaps for each model
        with torch.no_grad():
            for j, (model, model_name) in enumerate(zip(models, model_names)):
                pred = model(image_tensor.unsqueeze(0).to(device)).sigmoid().cpu().numpy()[0, 0]
                
                im = axes[i, j+1].imshow(pred, cmap='hot', vmin=0, vmax=1)
                axes[i, j+1].set_title(f"{model_name}\nConfidence Map")
                axes[i, j+1].axis('off')
                
                # Add colorbar
                plt.colorbar(im, ax=axes[i, j+1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Confidence heatmaps saved to {save_path}")

def plot_comprehensive_metrics_comparison(models, model_names, val_loader, device=DEVICE, save_path="comprehensive_metrics.png"):
    """Comprehensive metrics comparison with multiple evaluation criteria"""
    
    metrics = {name: {'iou': [], 'precision': [], 'recall': [], 'f1': []} for name in model_names}
    
    for model in models:
        model.eval()
    
    # Evaluate on validation set
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            gt_masks_np = masks.cpu().numpy().astype(int)
            
            for model, model_name in zip(models, model_names):
                preds = model(images).sigmoid()
                pred_masks_np = (preds.cpu().numpy() > 0.5).astype(int)
                
                # Calculate metrics for each sample in batch
                for b in range(images.shape[0]):
                    gt_flat = gt_masks_np[b].flatten()
                    pred_flat = pred_masks_np[b].flatten()
                    
                    # IoU
                    intersection = np.logical_and(pred_flat, gt_flat).sum()
                    union = np.logical_or(pred_flat, gt_flat).sum()
                    iou = intersection / union if union > 0 else 0
                    
                    # Precision, Recall, F1
                    precision = precision_score(gt_flat, pred_flat, zero_division=0)
                    recall = recall_score(gt_flat, pred_flat, zero_division=0)
                    f1 = f1_score(gt_flat, pred_flat, zero_division=0)
                    
                    metrics[model_name]['iou'].append(iou)
                    metrics[model_name]['precision'].append(precision)
                    metrics[model_name]['recall'].append(recall)
                    metrics[model_name]['f1'].append(f1)
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Box plots for each metric
    metric_names = ['iou', 'precision', 'recall', 'f1']
    metric_titles = ['IoU', 'Precision', 'Recall', 'F1-Score']
    
    for i, (metric, title) in enumerate(zip(metric_names, metric_titles)):
        row, col = i // 2, i % 2
        if i < 4:
            data = [metrics[name][metric] for name in model_names]
            bp = axes[row, col].boxplot(data, labels=model_names, patch_artist=True)
            
            # Color the boxes
            colors = ['lightblue', 'lightcoral', 'lightgreen']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
            
            axes[row, col].set_title(f'{title} Distribution')
            axes[row, col].set_ylabel(title)
            axes[row, col].tick_params(axis='x', rotation=45)
            axes[row, col].grid(True, alpha=0.3)
    
    # Mean metrics comparison (bar chart)
    mean_metrics = {name: {metric: np.mean(metrics[name][metric]) for metric in metric_names} 
                   for name in model_names}
    
    x = np.arange(len(metric_titles))
    width = 0.25
    
    for i, (model_name, color) in enumerate(zip(model_names, ['skyblue', 'lightcoral', 'lightgreen'])):
        values = [mean_metrics[model_name][metric] for metric in metric_names]
        axes[1, 2].bar(x + i * width, values, width, label=model_name, color=color, alpha=0.8)
    
    axes[1, 2].set_xlabel('Metrics')
    axes[1, 2].set_ylabel('Mean Score')
    axes[1, 2].set_title('Mean Metrics Comparison')
    axes[1, 2].set_xticks(x + width)
    axes[1, 2].set_xticklabels(metric_titles)
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    # Performance summary table
    axes[0, 2].axis('off')
    table_data = []
    for model_name in model_names:
        row = [model_name]
        for metric in metric_names:
            mean_val = np.mean(metrics[model_name][metric])
            std_val = np.std(metrics[model_name][metric])
            row.append(f"{mean_val:.3f}±{std_val:.3f}")
        table_data.append(row)
    
    table = axes[0, 2].table(cellText=table_data, 
                            colLabels=['Model'] + [title for title in metric_titles],
                            cellLoc='center',
                            loc='center',
                            bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    axes[0, 2].set_title('Performance Summary\n(Mean ± Std)', pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Comprehensive metrics comparison saved to {save_path}")

def plot_qualitative_failure_analysis(models, model_names, dataset, device=DEVICE, save_path="failure_analysis.png"):
    """Analyze failure cases and challenging scenarios"""
    
    # Find challenging samples (where all models perform poorly)
    challenging_samples = []
    
    for model in models:
        model.eval()
    
    for idx in range(min(len(dataset), 100)):  # Check first 100 samples
        image_tensor, mask_tensor = dataset[idx]
        gt_mask = mask_tensor.squeeze().cpu().numpy().astype(np.uint8)
        
        model_ious = []
        with torch.no_grad():
            for model in models:
                pred = model(image_tensor.unsqueeze(0).to(device)).sigmoid().cpu().numpy()[0, 0]
                pred_mask = (pred > 0.5).astype(np.uint8)
                
                intersection = np.logical_and(pred_mask, gt_mask).sum()
                union = np.logical_or(pred_mask, gt_mask).sum()
                iou = intersection / union if union > 0 else 0
                model_ious.append(iou)
        
        avg_iou = np.mean(model_ious)
        if avg_iou < 0.3:  # Consider as challenging if all models perform poorly
            challenging_samples.append((idx, avg_iou, model_ious))
    
    # Sort by difficulty (lowest average IoU first)
    challenging_samples.sort(key=lambda x: x[1])
    
    # Plot the most challenging cases
    num_samples = min(5, len(challenging_samples))
    if num_samples > 0:
        fig, axes = plt.subplots(num_samples, len(models) + 2, figsize=(6 * (len(models) + 2), 4 * num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(num_samples):
            idx, avg_iou, model_ious = challenging_samples[i]
            image_tensor, mask_tensor = dataset[idx]
            
            image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
            image_np = brighten_image(image_np, factor=1.5)
            if image_np.shape[2] > 3:
                image_np = image_np[:, :, :3]
            
            gt_mask = mask_tensor.squeeze().cpu().numpy().astype(np.uint8)
            
            # Original image
            axes[i, 0].imshow(image_np)
            axes[i, 0].set_title(f"Challenging Sample {idx}\nAvg IoU: {avg_iou:.3f}")
            axes[i, 0].axis('off')
            
            # Ground truth
            axes[i, 1].imshow(gt_mask, cmap='gray')
            axes[i, 1].set_title("Ground Truth")
            axes[i, 1].axis('off')
            
            # Model predictions
            with torch.no_grad():
                for j, (model, model_name, iou) in enumerate(zip(models, model_names, model_ious)):
                    pred = model(image_tensor.unsqueeze(0).to(device)).sigmoid().cpu().numpy()[0, 0]
                    pred_mask = (pred > 0.5).astype(np.uint8)
                    
                    axes[i, j+2].imshow(pred_mask, cmap='gray')
                    axes[i, j+2].set_title(f"{model_name}\nIoU: {iou:.3f}")
                    axes[i, j+2].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Failure analysis saved to {save_path}")
    else:
        print("No challenging samples found for failure analysis")

# ------------------------------------------------------------------------------------
# Main Comparison Function
# ------------------------------------------------------------------------------------
def load_models_and_compare(model_paths, base_path):
    """Load trained models and perform comprehensive comparison"""
    
    print("Loading models and preparing data...")
    
    # Prepare dataset
    val_image_dir = os.path.join(base_path, "testing", "image")
    val_mask_dir = os.path.join(base_path, "testing", "mask")
    
    if not os.path.exists(val_image_dir):
        # Fallback to training data if testing doesn't exist
        val_image_dir = os.path.join(base_path, "training", "image")
        val_mask_dir = os.path.join(base_path, "training", "mask")
    
    val_image_paths = sorted([os.path.join(val_image_dir, f) for f in os.listdir(val_image_dir) if f.endswith(".tif")])
    val_mask_paths = sorted([os.path.join(val_mask_dir, f) for f in os.listdir(val_mask_dir) if f.endswith(".tif")])
    
    # Take subset for comparison (first 50 samples)
    val_image_paths = val_image_paths[:50]
    val_mask_paths = val_mask_paths[:50]
    
    val_dataset = DeforestationDataset(val_image_paths, val_mask_paths, transform=get_transforms())
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize models
    models = []
    model_names = ['HRNet', 'PSPNet', 'FastFCN']
    
    # HRNet
    hrnet_model = HRNet(in_channels=4, num_classes=1)
    replace_batchnorm_with_groupnorm(hrnet_model, num_groups=8)
    if os.path.exists(model_paths['hrnet']):
        hrnet_model.load_state_dict(torch.load(model_paths['hrnet'], map_location=DEVICE))
        hrnet_model.to(DEVICE)
        models.append(hrnet_model)
        print("✓ HRNet model loaded")
    else:
        print(f"✗ HRNet model not found at {model_paths['hrnet']}")
        return
    
    # PSPNet
    pspnet_model = smp.PSPNet(
        encoder_name=ENCODER_NAME,
        encoder_weights=ENCODER_WEIGHTS,
        in_channels=4,
        classes=1
    )
    replace_batchnorm_with_groupnorm(pspnet_model, num_groups=8)
    if os.path.exists(model_paths['pspnet']):
        pspnet_model.load_state_dict(torch.load(model_paths['pspnet'], map_location=DEVICE))
        pspnet_model.to(DEVICE)
        models.append(pspnet_model)
        print("✓ PSPNet model loaded")
    else:
        print(f"✗ PSPNet model not found at {model_paths['pspnet']}")
        return
    
    # FastFCN
    fastfcn_model = FastFCN(in_channels=4, num_classes=1, backbone='resnet50')
    replace_batchnorm_with_groupnorm(fastfcn_model, num_groups=8)
    if os.path.exists(model_paths['fastfcn']):
        fastfcn_model.load_state_dict(torch.load(model_paths['fastfcn'], map_location=DEVICE))
        fastfcn_model.to(DEVICE)
        models.append(fastfcn_model)
        print("✓ FastFCN model loaded")
    else:
        print(f"✗ FastFCN model not found at {model_paths['fastfcn']}")
        return
    
    print("\nGenerating comprehensive comparison visualizations...")
    
    # Generate all comparison visualizations
    plot_side_by_side_comparison(models, model_names, val_dataset, num_samples=5, 
                                save_path="enhanced_side_by_side_comparison.png")
    
    plot_model_agreement_analysis(models, model_names, val_dataset, num_samples=5, 
                                save_path="model_agreement_analysis.png")
    
    plot_error_pattern_analysis(models, model_names, val_dataset, 
                               save_path="error_pattern_analysis.png")
    
    plot_prediction_confidence_heatmaps(models, model_names, val_dataset, num_samples=3, 
                                      save_path="confidence_heatmaps.png")
    
    plot_comprehensive_metrics_comparison(models, model_names, val_loader, 
                                        save_path="comprehensive_metrics_comparison.png")
    
    plot_qualitative_failure_analysis(models, model_names, val_dataset, 
                                     save_path="qualitative_failure_analysis.png")
    
    print("\n" + "="*60)
    print("ENHANCED MODEL COMPARISON COMPLETE")
    print("="*60)
    print("Generated visualizations:")
    print("• enhanced_side_by_side_comparison.png - Detailed sample-by-sample comparison")
    print("• model_agreement_analysis.png - Where models agree/disagree")
    print("• error_pattern_analysis.png - Statistical error analysis")
    print("• confidence_heatmaps.png - Prediction confidence visualization")
    print("• comprehensive_metrics_comparison.png - Quantitative metrics comparison")
    print("• qualitative_failure_analysis.png - Analysis of challenging cases")
    print("\nThese visualizations provide:")
    print("1. Qualitative assessment of segmentation quality")
    print("2. Model agreement and uncertainty analysis")
    print("3. Error pattern identification")
    print("4. Prediction confidence evaluation")
    print("5. Statistical performance comparison")
    print("6. Failure case analysis for model improvement")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced comparison of trained HRNet, PSPNet, and FastFCN models")
    parser.add_argument("--base_path", type=str, required=True, 
                       help="Path to dataset root directory (containing training/testing folders)")
    parser.add_argument("--hrnet_model", type=str, default="best_model_hrnet.pth",
                       help="Path to trained HRNet model")
    parser.add_argument("--pspnet_model", type=str, default="best_model_pspnet.pth",
                       help="Path to trained PSPNet model")
    parser.add_argument("--fastfcn_model", type=str, default="best_model_fastfcn.pth",
                       help="Path to trained FastFCN model")
    
    args = parser.parse_args()
    
    model_paths = {
        'hrnet': args.hrnet_model,
        'pspnet': args.pspnet_model,
        'fastfcn': args.fastfcn_model
    }
    
    load_models_and_compare(model_paths, args.base_path)