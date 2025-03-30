import os
import numpy as np
import matplotlib
matplotlib.use('agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import cv2
import rasterio
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    jaccard_score, precision_score, recall_score, f1_score, 
    confusion_matrix, precision_recall_curve, roc_curve, auc
)
from matplotlib.colors import ListedColormap
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# ------------------------------------------------------------------------------------
# Global Constants
# ------------------------------------------------------------------------------------
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 10

# ------------------------------------------------------------------------------------
# Utility Function
# ------------------------------------------------------------------------------------
def brighten_image(image_np, factor=1.5):
    """Multiply the image by a brightness factor and clamp to [0,1]."""
    return np.clip(image_np * factor, 0, 1)

# ------------------------------------------------------------------------------------
# Custom Dataset
# ------------------------------------------------------------------------------------
class DeforestationDataset(Dataset):
    """
    A PyTorch Dataset for reading 4-channel imagery (GeoTIFF).
    """
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
            # shape (C, H, W) -> transpose to (H, W, C)
            image = img_file.read().transpose(1, 2, 0)

        with rasterio.open(mask_path) as mask_file:
            # shape (1, H, W) -> squeeze to (H, W)
            mask = mask_file.read(1)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]

        # Also return the filename for reference in visualizations
        filename = os.path.basename(image_path)
        
        # Return mask with an extra channel dimension: (1, H, W)
        return image, mask.unsqueeze(0).float(), filename

# ------------------------------------------------------------------------------------
# Data Transforms
# ------------------------------------------------------------------------------------
def get_transforms(img_size=512):
    """
    Returns transforms for preprocessing images and masks.
    """
    return A.Compose([
        A.Resize(img_size, img_size),
        # Convert raw values (assumed 0-10000) to [0, 1]
        A.Lambda(image=lambda x, **kwargs: (x / 10000.0).astype(np.float32)),
        ToTensorV2(),
    ])

# ------------------------------------------------------------------------------------
# SAM2 Prediction Functions
# ------------------------------------------------------------------------------------
def predict_mask(predictor, image_tensor):
    """
    Generate a segmentation mask using SAM2.
    
    Args:
        predictor: SAM2ImagePredictor instance
        image_tensor: PyTorch tensor of shape (C, H, W)
        
    Returns:
        mask: Predicted binary mask as numpy array
        score: Confidence score
    """
    # Convert PyTorch tensor to numpy array for SAM2
    image_np = image_tensor.cpu().numpy().transpose(1, 2, 0)
    
    # Extract RGB channels for SAM2
    if image_np.shape[2] > 3:
        image_np = image_np[:, :, :3]
    
    # Ensure values are in [0, 1]
    image_np = np.clip(image_np, 0, 1)
    
    # Center point prompt
    h, w = image_np.shape[:2]
    point_x, point_y = w // 2, h // 2  # Center point
    
    # Run inference
    predictor.set_image(image_np)
    
    # Use the correct API
    masks, scores, logits = predictor.predict(
        point_coords=np.array([[point_x, point_y]]),
        point_labels=np.array([1]),  # 1 = foreground
        multimask_output=False
    )
    
    # Get the binary mask - already a numpy array
    mask = masks[0]  # First mask
    score = scores[0]  # First score
    
    return mask, score

def predict_batch(predictor, image_tensors):
    """
    Generate segmentation masks for a batch of images.
    
    Args:
        predictor: SAM2ImagePredictor instance
        image_tensors: List of PyTorch tensors, each with shape (C, H, W)
        
    Returns:
        masks: List of binary masks as numpy arrays
        scores: List of confidence scores
    """
    masks = []
    scores = []
    
    for image_tensor in image_tensors:
        mask, score = predict_mask(predictor, image_tensor)
        masks.append(mask)
        scores.append(score)
    
    return masks, scores

# ------------------------------------------------------------------------------------
# Metrics Calculation
# ------------------------------------------------------------------------------------
def calculate_metrics(pred_mask, true_mask):
    """Calculate IoU, precision, recall, and F1 score."""
    # Convert PyTorch tensor to numpy if needed
    if torch.is_tensor(true_mask):
        true_mask = true_mask.squeeze().cpu().numpy()
    
    pred_flat = pred_mask.flatten()
    true_flat = true_mask.flatten().astype(np.uint8)
    
    # Calculate basic metrics
    iou = jaccard_score(true_flat, pred_flat, average='binary', zero_division=0)
    precision = precision_score(true_flat, pred_flat, zero_division=0)
    recall = recall_score(true_flat, pred_flat, zero_division=0)
    f1 = f1_score(true_flat, pred_flat, zero_division=0)
    
    # Calculate confusion matrix
    cm = confusion_matrix(true_flat, pred_flat)
    
    # Calculate accuracy
    accuracy = (cm[0, 0] + cm[1, 1]) / cm.sum()
    
    return {
        'iou': iou,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'confusion_matrix': cm
    }

def compute_metrics_per_sample(predictor, dataset):
    """
    Compute metrics for each sample in the dataset.
    
    Args:
        predictor: SAM2ImagePredictor instance
        dataset: A DeforestationDataset
        
    Returns:
        metrics_list: List of dictionaries with metrics for each sample
    """
    metrics_list = []
    
    for idx in range(len(dataset)):
        image_tensor, mask_tensor, filename = dataset[idx]
        
        # Generate prediction
        pred_mask, score = predict_mask(predictor, image_tensor)
        
        # Calculate metrics
        metrics = calculate_metrics(pred_mask, mask_tensor)
        metrics['index'] = idx
        metrics['filename'] = filename
        metrics['score'] = score
        
        metrics_list.append(metrics)
        
    return metrics_list

def compute_aggregated_metrics(predictor, dataloader):
    """
    Compute aggregated metrics across all samples.
    
    Args:
        predictor: SAM2ImagePredictor instance
        dataloader: A PyTorch DataLoader
        
    Returns:
        dict: Aggregated metrics
    """
    all_preds = []
    all_gts = []
    
    for images, masks, _ in dataloader:
        batch_preds = []
        batch_gts = []
        
        for i in range(images.size(0)):
            pred_mask, _ = predict_mask(predictor, images[i])
            gt_mask = masks[i].squeeze().cpu().numpy()
            
            batch_preds.extend(pred_mask.flatten())
            batch_gts.extend(gt_mask.flatten())
        
        all_preds.extend(batch_preds)
        all_gts.extend(batch_gts)
    
    # Calculate metrics on all data
    iou = jaccard_score(all_gts, all_preds, average='binary', zero_division=0)
    precision = precision_score(all_gts, all_preds, zero_division=0)
    recall = recall_score(all_gts, all_preds, zero_division=0)
    f1 = f1_score(all_gts, all_preds, zero_division=0)
    cm = confusion_matrix(all_gts, all_preds)
    accuracy = (cm[0, 0] + cm[1, 1]) / cm.sum()
    
    return {
        'iou': iou,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'confusion_matrix': cm
    }

# ------------------------------------------------------------------------------------
# Visualization Functions
# ------------------------------------------------------------------------------------
def plot_confusion_matrix(confusion_matrix, class_names=None, save_path="confusion_matrix.png"):
    """Plots and saves a confusion matrix."""
    if class_names is None:
        class_names = ['Background', 'Deforestation']
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
        xlabel="Predicted label",
    )
    thresh = confusion_matrix.max() / 2.0
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            ax.text(j, i, int(confusion_matrix[i, j]),
                    ha="center", va="center",
                    color="white" if confusion_matrix[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

def plot_sample_predictions(predictor, dataset, num_samples=10, save_path="sample_predictions.png"):
    """
    Plot sample predictions from SAM2 alongside the original image and ground truth.
    """
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    
    for i, idx in enumerate(indices):
        image_tensor, mask_tensor, filename = dataset[idx]
        
        # Convert image to numpy for visualization
        image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
        
        # Extract RGB channels if needed
        if image_np.shape[2] > 3:
            rgb_image = image_np[:, :, :3]
        else:
            rgb_image = image_np
        
        # Brighten for display
        rgb_image = brighten_image(rgb_image, factor=1.5)
        
        # Get ground truth
        gt_mask = mask_tensor.squeeze().cpu().numpy()
        
        # Get prediction
        pred_mask, score = predict_mask(predictor, image_tensor)
        
        # Display
        axes[i, 0].imshow(rgb_image)
        axes[i, 0].set_title(f"Image: {filename}")
        axes[i, 0].axis("off")
        
        axes[i, 1].imshow(gt_mask, cmap='gray')
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis("off")
        
        axes[i, 2].imshow(pred_mask, cmap='gray')
        axes[i, 2].set_title(f"SAM2 Prediction\nScore: {score:.3f}")
        axes[i, 2].axis("off")
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

def plot_difference_maps(predictor, dataset, num_samples=10, save_path="difference_maps.png"):
    """
    Plot difference maps showing false positives and false negatives.
    """
    diff_cmap = ListedColormap(["black", "red", "blue"])  # 0=BG, 1=FP, 2=FN
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    
    for i, idx in enumerate(indices):
        image_tensor, mask_tensor, filename = dataset[idx]
        
        # Convert image to numpy for visualization
        image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
        
        # Extract RGB channels if needed
        if image_np.shape[2] > 3:
            rgb_image = image_np[:, :, :3]
        else:
            rgb_image = image_np
        
        # Brighten for display
        rgb_image = brighten_image(rgb_image, factor=1.5)
        
        # Get ground truth
        gt_mask = mask_tensor.squeeze().cpu().numpy()
        
        # Get prediction
        pred_mask, _ = predict_mask(predictor, image_tensor)
        
        # Calculate difference map
        fp = (pred_mask == 1) & (gt_mask == 0)  # False positives
        fn = (pred_mask == 0) & (gt_mask == 1)  # False negatives
        diff_map = np.zeros_like(gt_mask, dtype=np.uint8)
        diff_map[fp] = 1  # False positives in red
        diff_map[fn] = 2  # False negatives in blue
        
        # Display
        axes[i, 0].imshow(rgb_image)
        axes[i, 0].set_title(f"Image: {filename}")
        axes[i, 0].axis("off")
        
        axes[i, 1].imshow(gt_mask, cmap='gray')
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis("off")
        
        axes[i, 2].imshow(diff_map, cmap=diff_cmap, vmin=0, vmax=2)
        axes[i, 2].set_title("Difference Map\nRed=FP, Blue=FN")
        axes[i, 2].axis("off")
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

def plot_overlay_predictions(predictor, dataset, num_samples=10, alpha=0.4, save_path="overlay_predictions.png"):
    """
    Plot the original image with the prediction overlaid.
    """
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    fig, axes = plt.subplots(num_samples, 2, figsize=(12, 5 * num_samples))
    
    for i, idx in enumerate(indices):
        image_tensor, mask_tensor, filename = dataset[idx]
        
        # Convert image to numpy for visualization
        image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
        
        # Extract RGB channels if needed
        if image_np.shape[2] > 3:
            rgb_image = image_np[:, :, :3]
        else:
            rgb_image = image_np
        
        # Brighten for display
        rgb_image = brighten_image(rgb_image, factor=1.5)
        
        # Get prediction
        pred_mask, _ = predict_mask(predictor, image_tensor)
        
        # Create overlay
        overlay = np.zeros_like(rgb_image)
        overlay[..., 0] = pred_mask  # Fill red channel with mask
        
        # Blend original image with overlay
        blended = (1 - alpha) * rgb_image + alpha * overlay
        
        # Display
        axes[i, 0].imshow(rgb_image)
        axes[i, 0].set_title(f"Original Image: {filename}")
        axes[i, 0].axis("off")
        
        axes[i, 1].imshow(blended)
        axes[i, 1].set_title("Prediction Overlay (Red)")
        axes[i, 1].axis("off")
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

def plot_overlay_predictions_with_contours(predictor, dataset, num_samples=10, save_path="overlay_contours.png"):
    """
    Plot the original image with the prediction contours.
    """
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    fig, axes = plt.subplots(num_samples, 2, figsize=(12, 5 * num_samples))
    
    for i, idx in enumerate(indices):
        image_tensor, mask_tensor, filename = dataset[idx]
        
        # Convert image to numpy for visualization
        image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
        
        # Extract RGB channels if needed
        if image_np.shape[2] > 3:
            rgb_image = image_np[:, :, :3]
        else:
            rgb_image = image_np
        
        # Brighten for display
        rgb_image = brighten_image(rgb_image, factor=1.5)
        
        # Get prediction
        pred_mask, _ = predict_mask(predictor, image_tensor)
        
        # Convert to uint8 for OpenCV
        rgb_uint8 = (rgb_image * 255).astype(np.uint8)
        mask_uint8 = pred_mask.astype(np.uint8)
        
        # Find contours
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw contours
        contour_img = rgb_uint8.copy()
        cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
        
        # Convert back to float
        contour_img = contour_img.astype(np.float32) / 255.0
        
        # Display
        axes[i, 0].imshow(rgb_image)
        axes[i, 0].set_title(f"Original Image: {filename}")
        axes[i, 0].axis("off")
        
        axes[i, 1].imshow(contour_img)
        axes[i, 1].set_title("Prediction Contours (Green)")
        axes[i, 1].axis("off")
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

def plot_worst_samples(predictor, dataset, metrics_list=None, k=5, save_path="worst_samples.png"):
    """
    Plot the k worst samples based on IoU.
    
    Args:
        predictor: SAM2ImagePredictor instance
        dataset: Dataset
        metrics_list: Optional pre-computed metrics list
        k: Number of worst samples to plot
        save_path: Path to save the plot
    """
    # Calculate metrics if not provided
    if metrics_list is None:
        metrics_list = compute_metrics_per_sample(predictor, dataset)
    
    # Sort by IoU in ascending order
    sorted_metrics = sorted(metrics_list, key=lambda x: x['iou'])
    worst_samples = sorted_metrics[:k]
    
    # Plot
    fig, axes = plt.subplots(k, 4, figsize=(20, 5 * k))
    diff_cmap = ListedColormap(["black", "red", "blue"])  # 0=BG, 1=FP, 2=FN
    
    for i, entry in enumerate(worst_samples):
        idx = entry['index']
        filename = entry['filename']
        
        image_tensor, mask_tensor, _ = dataset[idx]
        
        # Convert image to numpy for visualization
        image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
        
        # Extract RGB channels if needed
        if image_np.shape[2] > 3:
            rgb_image = image_np[:, :, :3]
        else:
            rgb_image = image_np
        
        # Brighten for display
        rgb_image = brighten_image(rgb_image, factor=1.5)
        
        # Get ground truth
        gt_mask = mask_tensor.squeeze().cpu().numpy()
        
        # Get prediction
        pred_mask, score = predict_mask(predictor, image_tensor)
        
        # Calculate difference map
        fp = (pred_mask == 1) & (gt_mask == 0)  # False positives
        fn = (pred_mask == 0) & (gt_mask == 1)  # False negatives
        diff_map = np.zeros_like(gt_mask, dtype=np.uint8)
        diff_map[fp] = 1  # False positives in red
        diff_map[fn] = 2  # False negatives in blue
        
        # Display
        axes[i, 0].imshow(rgb_image)
        axes[i, 0].set_title(f"Image: {filename}")
        axes[i, 0].axis("off")
        
        axes[i, 1].imshow(gt_mask, cmap='gray')
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis("off")
        
        axes[i, 2].imshow(pred_mask, cmap='gray')
        metrics_text = f"IoU: {entry['iou']:.3f}, F1: {entry['f1']:.3f}\nPrec: {entry['precision']:.3f}, Rec: {entry['recall']:.3f}"
        axes[i, 2].set_title(f"Prediction\n{metrics_text}")
        axes[i, 2].axis("off")
        
        axes[i, 3].imshow(diff_map, cmap=diff_cmap, vmin=0, vmax=2)
        axes[i, 3].set_title("Error Map\nRed=FP, Blue=FN")
        axes[i, 3].axis("off")
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

def plot_best_samples(predictor, dataset, metrics_list=None, k=5, save_path="best_samples.png"):
    """
    Plot the k best samples based on IoU.
    
    Args:
        predictor: SAM2ImagePredictor instance
        dataset: Dataset
        metrics_list: Optional pre-computed metrics list
        k: Number of best samples to plot
        save_path: Path to save the plot
    """
    # Calculate metrics if not provided
    if metrics_list is None:
        metrics_list = compute_metrics_per_sample(predictor, dataset)
    
    # Sort by IoU in descending order
    sorted_metrics = sorted(metrics_list, key=lambda x: x['iou'], reverse=True)
    best_samples = sorted_metrics[:k]
    
    # Plot
    fig, axes = plt.subplots(k, 4, figsize=(20, 5 * k))
    diff_cmap = ListedColormap(["black", "red", "blue"])  # 0=BG, 1=FP, 2=FN
    
    for i, entry in enumerate(best_samples):
        idx = entry['index']
        filename = entry['filename']
        
        image_tensor, mask_tensor, _ = dataset[idx]
        
        # Convert image to numpy for visualization
        image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
        
        # Extract RGB channels if needed
        if image_np.shape[2] > 3:
            rgb_image = image_np[:, :, :3]
        else:
            rgb_image = image_np
        
        # Brighten for display
        rgb_image = brighten_image(rgb_image, factor=1.5)
        
        # Get ground truth
        gt_mask = mask_tensor.squeeze().cpu().numpy()
        
        # Get prediction
        pred_mask, score = predict_mask(predictor, image_tensor)
        
        # Calculate difference map
        fp = (pred_mask == 1) & (gt_mask == 0)  # False positives
        fn = (pred_mask == 0) & (gt_mask == 1)  # False negatives
        diff_map = np.zeros_like(gt_mask, dtype=np.uint8)
        diff_map[fp] = 1  # False positives in red
        diff_map[fn] = 2  # False negatives in blue
        
        # Display
        axes[i, 0].imshow(rgb_image)
        axes[i, 0].set_title(f"Image: {filename}")
        axes[i, 0].axis("off")
        
        axes[i, 1].imshow(gt_mask, cmap='gray')
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis("off")
        
        axes[i, 2].imshow(pred_mask, cmap='gray')
        metrics_text = f"IoU: {entry['iou']:.3f}, F1: {entry['f1']:.3f}\nPrec: {entry['precision']:.3f}, Rec: {entry['recall']:.3f}"
        axes[i, 2].set_title(f"Prediction\n{metrics_text}")
        axes[i, 2].axis("off")
        
        axes[i, 3].imshow(diff_map, cmap=diff_cmap, vmin=0, vmax=2)
        axes[i, 3].set_title("Error Map\nRed=FP, Blue=FN")
        axes[i, 3].axis("off")
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

def generate_pr_curve(predictor, dataset, num_thresholds=100, save_path="pr_curve.png"):
    """
    Generate a precision-recall curve.
    
    This is trickier with SAM2 since it doesn't output probabilities directly.
    We'll use distance transform to create an approximate probability map.
    """
    all_true = []
    all_dist = []
    
    for idx in range(min(len(dataset), 50)):  # Limit to 50 samples for speed
        image_tensor, mask_tensor, _ = dataset[idx]
        
        # Get ground truth
        gt_mask = mask_tensor.squeeze().cpu().numpy()
        all_true.extend(gt_mask.flatten())
        
        # Get prediction
        pred_mask, _ = predict_mask(predictor, image_tensor)
        
        # Calculate distance transform (approximation of "confidence")
        dist = cv2.distanceTransform((pred_mask * 255).astype(np.uint8), cv2.DIST_L2, 3)
        dist = dist / dist.max()  # Normalize to [0, 1]
        all_dist.extend(dist.flatten())
    
    # Convert to numpy arrays
    all_true = np.array(all_true)
    all_dist = np.array(all_dist)
    
    # Calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(all_true, all_dist)
    pr_auc = auc(recall, precision)
    
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(all_true, all_dist)
    roc_auc = auc(fpr, tpr)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Precision-Recall curve
    axes[0].plot(recall, precision, 'b-', linewidth=2)
    axes[0].set_xlabel('Recall')
    axes[0].set_ylabel('Precision')
    axes[0].set_title(f'Precision-Recall Curve (AUC = {pr_auc:.3f})')
    axes[0].grid(True)
    
    # ROC curve
    axes[1].plot(fpr, tpr, 'r-', linewidth=2)
    axes[1].plot([0, 1], [0, 1], 'k--')  # Diagonal
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title(f'ROC Curve (AUC = {roc_auc:.3f})')
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

def plot_predicted_vs_gt_area(predictor, dataset, save_path="area_comparison.png"):
    """
    Plot predicted area vs ground truth area.
    """
    gt_areas = []
    pred_areas = []
    
    for idx in range(len(dataset)):
        image_tensor, mask_tensor, _ = dataset[idx]
        
        # Get ground truth area
        gt_mask = mask_tensor.squeeze().cpu().numpy()
        gt_area = gt_mask.sum() / gt_mask.size
        gt_areas.append(gt_area)
        
        # Get predicted area
        pred_mask, _ = predict_mask(predictor, image_tensor)
        pred_area = pred_mask.sum() / pred_mask.size
        pred_areas.append(pred_area)
    
    # Plot
    plt.figure(figsize=(8, 8))
    plt.scatter(gt_areas, pred_areas, alpha=0.7)
    plt.plot([0, 1], [0, 1], 'r--')  # Diagonal
    plt.xlabel('Ground Truth Area (fraction)')
    plt.ylabel('Predicted Area (fraction)')
    plt.title('Predicted vs Ground Truth Area')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_aggregate_error_heatmap(predictor, dataset, save_path="aggregate_error_heatmap.png"):
    """
    Generate an aggregate error heatmap.
    """
    error_sum = None
    count = 0
    
    for idx in range(len(dataset)):
        image_tensor, mask_tensor, _ = dataset[idx]
        
        # Get ground truth
        gt_mask = mask_tensor.squeeze().cpu().numpy()
        
        # Get prediction
        pred_mask, _ = predict_mask(predictor, image_tensor)
        
        # Calculate error
        error = (pred_mask != gt_mask).astype(np.float32)
        
        # Initialize or add to error sum
        if error_sum is None:
            error_sum = error
        else:
            error_sum += error
        
        count += 1
    
    # Calculate error frequency
    error_heatmap = error_sum / count
    
    # Plot
    plt.figure(figsize=(8, 8))
    im = plt.imshow(error_heatmap, cmap='hot')
    plt.colorbar(im, label='Error Frequency')
    plt.title('Aggregate Error Heatmap')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_metrics_distribution(metrics_list, save_path="metrics_distribution.png"):
    """
    Plot distribution of IoU, F1, precision, and recall.
    """
    # Extract metrics
    ious = [m['iou'] for m in metrics_list]
    f1s = [m['f1'] for m in metrics_list]
    precisions = [m['precision'] for m in metrics_list]
    recalls = [m['recall'] for m in metrics_list]
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # IoU
    axes[0, 0].hist(ious, bins=20, alpha=0.7, color='blue')
    axes[0, 0].axvline(np.mean(ious), color='r', linestyle='--', label=f'Mean: {np.mean(ious):.3f}')
    axes[0, 0].set_xlabel('IoU')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('IoU Distribution')
    axes[0, 0].grid(True)
    axes[0, 0].legend()
    
    # F1
    axes[0, 1].hist(f1s, bins=20, alpha=0.7, color='green')
    axes[0, 1].axvline(np.mean(f1s), color='r', linestyle='--', label=f'Mean: {np.mean(f1s):.3f}')
    axes[0, 1].set_xlabel('F1 Score')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('F1 Score Distribution')
    axes[0, 1].grid(True)
    axes[0, 1].legend()
    
    # Precision
    axes[1, 0].hist(precisions, bins=20, alpha=0.7, color='purple')
    axes[1, 0].axvline(np.mean(precisions), color='r', linestyle='--', label=f'Mean: {np.mean(precisions):.3f}')
    axes[1, 0].set_xlabel('Precision')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Precision Distribution')
    axes[1, 0].grid(True)
    axes[1, 0].legend()
    
    # Recall
    axes[1, 1].hist(recalls, bins=20, alpha=0.7, color='orange')
    axes[1, 1].axvline(np.mean(recalls), color='r', linestyle='--', label=f'Mean: {np.mean(recalls):.3f}')
    axes[1, 1].set_xlabel('Recall')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Recall Distribution')
    axes[1, 1].grid(True)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

def log_metrics_summary(metrics_list, global_metrics=None):
    """
    Log a summary of metrics.
    """
    # Calculate average metrics
    avg_iou = np.mean([m['iou'] for m in metrics_list])
    avg_f1 = np.mean([m['f1'] for m in metrics_list])
    avg_precision = np.mean([m['precision'] for m in metrics_list])
    avg_recall = np.mean([m['recall'] for m in metrics_list])
    avg_accuracy = np.mean([m['accuracy'] for m in metrics_list])
    
    # Standard deviations
    std_iou = np.std([m['iou'] for m in metrics_list])
    std_f1 = np.std([m['f1'] for m in metrics_list])
    std_precision = np.std([m['precision'] for m in metrics_list])
    std_recall = np.std([m['recall'] for m in metrics_list])
    std_accuracy = np.std([m['accuracy'] for m in metrics_list])
    
    # Print summary
    print("\n" + "="*50)
    print("SAM2 EVALUATION SUMMARY")
    print("="*50)
    print(f"Samples evaluated: {len(metrics_list)}")
    print("\nPer-Sample Metrics (mean ± std):")
    print(f"IoU:       {avg_iou:.4f} ± {std_iou:.4f}")
    print(f"F1 Score:  {avg_f1:.4f} ± {std_f1:.4f}")
    print(f"Precision: {avg_precision:.4f} ± {std_precision:.4f}")
    print(f"Recall:    {avg_recall:.4f} ± {std_recall:.4f}")
    print(f"Accuracy:  {avg_accuracy:.4f} ± {std_accuracy:.4f}")
    
    if global_metrics:
        print("\nGlobal Metrics (calculated across all pixels):")
        print(f"IoU:       {global_metrics['iou']:.4f}")
        print(f"F1 Score:  {global_metrics['f1']:.4f}")
        print(f"Precision: {global_metrics['precision']:.4f}")
        print(f"Recall:    {global_metrics['recall']:.4f}")
        print(f"Accuracy:  {global_metrics['accuracy']:.4f}")
        
        print("\nConfusion Matrix:")
        cm = global_metrics['confusion_matrix']
        print(f"TN: {cm[0, 0]}, FP: {cm[0, 1]}")
        print(f"FN: {cm[1, 0]}, TP: {cm[1, 1]}")
    
    # Get best and worst samples
    sorted_by_iou = sorted(metrics_list, key=lambda x: x['iou'])
    worst_samples = sorted_by_iou[:5]
    best_samples = sorted_by_iou[-5:]
    
    print("\nWorst 5 Samples (by IoU):")
    for idx, sample in enumerate(worst_samples):
        print(f"{idx+1}. File: {sample['filename']}, IoU: {sample['iou']:.4f}, F1: {sample['f1']:.4f}")
    
    print("\nBest 5 Samples (by IoU):")
    for idx, sample in enumerate(reversed(best_samples)):
        print(f"{idx+1}. File: {sample['filename']}, IoU: {sample['iou']:.4f}, F1: {sample['f1']:.4f}")
    
    print("="*50 + "\n")

# ------------------------------------------------------------------------------------
# Main Evaluation Function
# ------------------------------------------------------------------------------------
def evaluate_sam2(base_path, output_dir="sam2_results", num_samples=None, use_full_dataset=True):
    """
    Main function to evaluate SAM2 on a dataset.
    
    Args:
        base_path: Path to the dataset directory
        output_dir: Directory to save output files
        num_samples: Number of samples to evaluate (None = all)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up dataset paths
    image_dir = os.path.join(base_path, "training", "image")
    mask_dir = os.path.join(base_path, "training", "mask")
    
    # Get all image and mask files
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.tif')])
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.tif')])
    
    # Limit the number of samples if specified
    if num_samples is not None and num_samples > 0:
        image_files = image_files[:num_samples]
        mask_files = mask_files[:num_samples]
    
    # Create full paths
    image_paths = [os.path.join(image_dir, f) for f in image_files]
    mask_paths = [os.path.join(mask_dir, f) for f in mask_files]
    
    # Split into train/val sets
    train_img_paths, val_img_paths, train_mask_paths, val_mask_paths = train_test_split(
        image_paths, mask_paths, test_size=0.2, random_state=42
    )
    
    # Create dataset and dataloader
    val_dataset = DeforestationDataset(
        val_img_paths, 
        val_mask_paths, 
        transform=get_transforms(img_size=512)
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False
    )
    
    # Load SAM2 model
    print("Loading SAM2 model...")
    predictor = SAM2ImagePredictor.from_pretrained(
        "facebook/sam2-hiera-large", 
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Model loaded on {predictor.device}")
    
    # Compute metrics for each sample
    print(f"Evaluating on {len(val_dataset)} validation images...")
    metrics_list = compute_metrics_per_sample(predictor, val_dataset)
    
    # Compute global metrics
    global_metrics = None
    try:
        global_metrics = compute_aggregated_metrics(predictor, DataLoader(val_dataset, batch_size=1, shuffle=False))
    except Exception as e:
        print(f"Warning: Failed to compute global metrics: {e}")
    
    # Log metrics summary
    log_metrics_summary(metrics_list, global_metrics)
    
    # Generate visualizations
    print("Generating visualizations...")
    
    # Basic visualizations
    plot_sample_predictions(predictor, val_dataset, num_samples=10, 
                         save_path=os.path.join(output_dir, "sample_predictions.png"))
    
    plot_difference_maps(predictor, val_dataset, num_samples=10, 
                        save_path=os.path.join(output_dir, "difference_maps.png"))
    
    plot_overlay_predictions(predictor, val_dataset, num_samples=10, alpha=0.4, 
                          save_path=os.path.join(output_dir, "overlay_predictions.png"))
    
    plot_overlay_predictions_with_contours(predictor, val_dataset, num_samples=10, 
                                        save_path=os.path.join(output_dir, "overlay_contours.png"))
    
    plot_worst_samples(predictor, val_dataset, metrics_list, k=5, 
                    save_path=os.path.join(output_dir, "worst_samples.png"))
    
    plot_best_samples(predictor, val_dataset, metrics_list, k=5, 
                   save_path=os.path.join(output_dir, "best_samples.png"))
    
    # Advanced visualizations
    try:
        generate_pr_curve(predictor, val_dataset, 
                         save_path=os.path.join(output_dir, "pr_curve.png"))
    except Exception as e:
        print(f"Warning: Failed to generate PR curve: {e}")
    
    plot_predicted_vs_gt_area(predictor, val_dataset, 
                           save_path=os.path.join(output_dir, "area_comparison.png"))
    
    try:
        plot_aggregate_error_heatmap(predictor, val_dataset, 
                                  save_path=os.path.join(output_dir, "aggregate_error_heatmap.png"))
    except Exception as e:
        print(f"Warning: Failed to generate error heatmap: {e}")
    
    plot_metrics_distribution(metrics_list, 
                          save_path=os.path.join(output_dir, "metrics_distribution.png"))
    
    if global_metrics is not None:
        plot_confusion_matrix(global_metrics['confusion_matrix'], 
                           save_path=os.path.join(output_dir, "confusion_matrix.png"))
    
    print(f"Evaluation complete. Results saved to {output_dir}")
    return metrics_list, global_metrics

# ------------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate SAM2 on deforestation imagery.")
    parser.add_argument("--base_path", type=str, required=True, help="Path to dataset root directory")
    parser.add_argument("--output_dir", type=str, default="sam2_results", help="Directory to save output files")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to evaluate (0=all)")
    # Add this line to support the --full_dataset flag:
    parser.add_argument("--full_dataset", action="store_true", help="Use full dataset instead of just validation set")
    args = parser.parse_args()
    
    if args.num_samples == 0:
        args.num_samples = None
        
    # Make sure your function call includes the new parameter
    evaluate_sam2(args.base_path, args.output_dir, args.num_samples, args.full_dataset)