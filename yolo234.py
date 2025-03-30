import os
import glob
import shutil
import argparse
import cv2
import rasterio
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from ultralytics import YOLO
from sklearn.metrics import confusion_matrix

# -----------------------------
# Part 1. Dataset Preparation: Convert TIFFs to 4–channel PNGs, generate YOLO labels,
# and copy ground truth masks for pixel evaluation.
# -----------------------------
def generate_label_for_image(image_path, mask_path, label_path):
    """
    Reads the original TIFF (to get dimensions) and its corresponding mask,
    extracts bounding boxes (via contour detection), converts them to YOLO format,
    and writes them to a label file.
    """
    with rasterio.open(image_path) as img_file:
        img = img_file.read()  # shape: (channels, H, W)
        _, H, W = img.shape

    with rasterio.open(mask_path) as mask_file:
        mask = mask_file.read(1)

    # Create binary mask and extract contours
    mask_bin = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    with open(label_path, 'w') as f:
        for cnt in contours:
            x, y, w_box, h_box = cv2.boundingRect(cnt)
            # Convert bounding box to YOLO format: <class> x_center y_center width height (normalized)
            x_center = (x + x + w_box) / 2.0 / W
            y_center = (y + y + h_box) / 2.0 / H
            norm_w = w_box / W
            norm_h = h_box / H
            f.write(f"0 {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")

def convert_tiff_to_png_4ch(tiff_path, dest_path):
    """
    Reads an 11–channel TIFF image using rasterio, extracts channels 1–4,
    normalizes the values to 0–255, and saves as a PNG.
    """
    with rasterio.open(tiff_path) as src:
        img = src.read([1, 2, 3, 4]).astype(np.float32)  # shape: (4, H, W)
        img = np.transpose(img, (1, 2, 0))                # shape: (H, W, 4)
        img_min, img_max = img.min(), img.max()
        img = (img - img_min) / (img_max - img_min + 1e-6) * 255
        img = img.astype(np.uint8)
    cv2.imwrite(dest_path, img)

def prepare_yolo_dataset(base_path):
    """
    Assumes your original dataset is organized as:
       base_path/training/image  (11–channel TIFF images)
       base_path/training/mask   (segmentation masks as TIFF)
       
    This function splits the data into training and validation sets,
    converts images to 4–channel PNGs (using channels 1–4),
    generates YOLO–formatted label files, and copies the ground truth masks
    for pixel-level evaluation.
    """
    image_dir = os.path.join(base_path, "training", "image")
    mask_dir  = os.path.join(base_path, "training", "mask")
    
    image_files = sorted(glob.glob(os.path.join(image_dir, "*.tif")))
    mask_files  = sorted(glob.glob(os.path.join(mask_dir, "*.tif")))
    
    train_images, val_images, train_masks, val_masks = train_test_split(
        image_files, mask_files, test_size=0.2, random_state=42
    )
    
    # Create YOLO folder structure
    yolo_images_train = os.path.join(base_path, "yolo", "images", "train")
    yolo_images_val   = os.path.join(base_path, "yolo", "images", "val")
    yolo_labels_train = os.path.join(base_path, "yolo", "labels", "train")
    yolo_labels_val   = os.path.join(base_path, "yolo", "labels", "val")
    yolo_masks_val    = os.path.join(base_path, "yolo", "masks", "val")  # For pixel-level GT
    os.makedirs(yolo_images_train, exist_ok=True)
    os.makedirs(yolo_images_val, exist_ok=True)
    os.makedirs(yolo_labels_train, exist_ok=True)
    os.makedirs(yolo_labels_val, exist_ok=True)
    os.makedirs(yolo_masks_val, exist_ok=True)
    
    # Process training images and labels
    for img_path, mask_path in zip(train_images, train_masks):
        basename = os.path.basename(img_path)
        base_no_ext = os.path.splitext(basename)[0]
        dest_img_path = os.path.join(yolo_images_train, base_no_ext + ".png")
        convert_tiff_to_png_4ch(img_path, dest_img_path)
        
        label_filename = base_no_ext + ".txt"
        dest_label_path = os.path.join(yolo_labels_train, label_filename)
        generate_label_for_image(img_path, mask_path, dest_label_path)
    
    # Process validation images and labels; also copy GT masks for pixel eval.
    for img_path, mask_path in zip(val_images, val_masks):
        basename = os.path.basename(img_path)
        base_no_ext = os.path.splitext(basename)[0]
        dest_img_path = os.path.join(yolo_images_val, base_no_ext + ".png")
        convert_tiff_to_png_4ch(img_path, dest_img_path)
        
        label_filename = base_no_ext + ".txt"
        dest_label_path = os.path.join(yolo_labels_val, label_filename)
        generate_label_for_image(img_path, mask_path, dest_label_path)
        
        # Convert the GT mask to a binary PNG for pixel evaluation.
        with rasterio.open(mask_path) as mask_file:
            mask = mask_file.read(1)
            binary_mask = (mask > 0).astype(np.uint8) * 255
        dest_mask_path = os.path.join(yolo_masks_val, base_no_ext + ".png")
        cv2.imwrite(dest_mask_path, binary_mask)
    
    print("YOLO dataset prepared:")
    print(f"  Training images: {len(train_images)}")
    print(f"  Validation images: {len(val_images)}")

def generate_data_yaml(base_path):
    """
    Creates a data.yaml file for YOLOv8 training.
    """
    yolo_dir = os.path.join(base_path, "yolo")
    images_train = os.path.abspath(os.path.join(yolo_dir, "images", "train"))
    images_val   = os.path.abspath(os.path.join(yolo_dir, "images", "val"))
    data_yaml_path = os.path.join(yolo_dir, "data.yaml")
    content = f"""train: {images_train}
val: {images_val}
nc: 1
names: ["deforestation"]
"""
    with open(data_yaml_path, "w") as f:
        f.write(content)
    print(f"Data YAML file created at: {data_yaml_path}")
    return data_yaml_path

# -----------------------------
# Part 2. Evaluation: Compute Detection Metrics and Pixel-Level Confusion Matrix & IoU
# -----------------------------
def compute_iou(boxA, boxB):
    """
    Compute Intersection over Union (IoU) for two boxes in [x_min, y_min, x_max, y_max] format.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)

def match_boxes(pred_boxes, gt_boxes, iou_threshold=0.5):
    """
    Greedily match predicted boxes to ground truth boxes.
    Returns TP (matched pairs), FP, FN, and a list of IoU values for matched pairs.
    """
    tp = 0
    iou_list = []
    matched_gt = set()
    for pb in pred_boxes:
        best_iou = 0.0
        best_gt_idx = -1
        for j, gb in enumerate(gt_boxes):
            if j in matched_gt:
                continue
            iou_val = compute_iou(pb, gb)
            if iou_val > best_iou:
                best_iou = iou_val
                best_gt_idx = j
        if best_iou >= iou_threshold and best_gt_idx != -1:
            tp += 1
            iou_list.append(best_iou)
            matched_gt.add(best_gt_idx)
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp
    return tp, fp, fn, iou_list

def plot_confusion_matrix(cm, class_names=["Positive", "Negative"], save_path="confusion_matrix.png"):
    """
    Plot a 2x2 confusion matrix.
    In detection evaluation, we define: [[TP, FN], [FP, 0]]
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    tick_marks = np.arange(len(class_names))
    ax.set(xticks=tick_marks, yticks=tick_marks,
           xticklabels=class_names, yticklabels=class_names,
           ylabel="Ground Truth", xlabel="Prediction",
           title="Detection Confusion Matrix")
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Detection confusion matrix saved to {save_path}")

def evaluate_detection(model, val_images_dir, val_labels_dir, iou_threshold=0.5, imgsz=640):
    """
    Evaluate YOLOv8 detection performance on the validation set using predicted boxes.
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0
    all_iou = []
    
    val_image_files = sorted(glob.glob(os.path.join(val_images_dir, "*.png")))
    
    for img_path in val_image_files:
        basename = os.path.basename(img_path)
        label_path = os.path.join(val_labels_dir, os.path.splitext(basename)[0] + ".txt")
        
        with rasterio.open(img_path) as src:
            img = src.read().transpose(1, 2, 0)
            H, W, _ = img.shape
        
        gt_boxes = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        _, xc, yc, bw, bh = map(float, parts)
                        x_min = (xc - bw/2) * W
                        y_min = (yc - bh/2) * H
                        x_max = (xc + bw/2) * W
                        y_max = (yc + bh/2) * H
                        gt_boxes.append([x_min, y_min, x_max, y_max])
        else:
            continue
        
        results = model.predict(source=img_path, imgsz=imgsz, conf=0.25, device='cuda:1')
        pred_boxes = []
        if results and hasattr(results[0], "boxes") and results[0].boxes is not None:
            try:
                pred_boxes = results[0].boxes.xyxy.cpu().numpy().tolist()
            except Exception as e:
                print(f"Error extracting boxes for {img_path}: {e}")
        
        tp, fp, fn, iou_list = match_boxes(pred_boxes, gt_boxes, iou_threshold=iou_threshold)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        all_iou.extend(iou_list)
    
    avg_iou = np.mean(all_iou) if all_iou else 0.0
    print(f"Detection Evaluation Results: TP={total_tp}, FP={total_fp}, FN={total_fn}, Avg IoU={avg_iou:.4f}")
    
    cm = np.array([[total_tp, total_fn],
                   [total_fp, 0]])
    plot_confusion_matrix(cm, class_names=["Positive", "Negative"], save_path="confusion_matrix.png")
    
    return total_tp, total_fp, total_fn, avg_iou

def evaluate_pixel_level(model, val_images_dir, gt_masks_dir, imgsz=640):
    """
    Evaluate pixel-level performance.
    For each validation image, generate a predicted binary mask by filling predicted bounding boxes,
    load the ground truth binary mask from gt_masks_dir, and compute a pixel-level confusion matrix and IoU.
    """
    total_TP = total_FP = total_TN = total_FN = 0
    all_iou = []
    
    val_image_files = sorted(glob.glob(os.path.join(val_images_dir, "*.png")))
    
    for img_path in val_image_files:
        basename = os.path.basename(img_path)
        with rasterio.open(img_path) as src:
            img = src.read().transpose(1, 2, 0)
            H, W, _ = img.shape
        
        # Create an empty predicted mask (0: background, 1: deforestation)
        pred_mask = np.zeros((H, W), dtype=np.uint8)
        results = model.predict(source=img_path, imgsz=imgsz, conf=0.25, device='cuda:1')
        pred_boxes = []
        if results and hasattr(results[0], "boxes") and results[0].boxes is not None:
            try:
                pred_boxes = results[0].boxes.xyxy.cpu().numpy().tolist()
            except Exception as e:
                print(f"Error extracting boxes for {img_path}: {e}")
        
        # Fill predicted boxes into pred_mask
        for box in pred_boxes:
            x_min, y_min, x_max, y_max = map(int, box)
            cv2.rectangle(pred_mask, (x_min, y_min), (x_max, y_max), 1, thickness=-1)
        
        gt_mask_path = os.path.join(gt_masks_dir, os.path.splitext(basename)[0] + ".png")
        if os.path.exists(gt_mask_path):
            gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
            _, gt_mask = cv2.threshold(gt_mask, 127, 1, cv2.THRESH_BINARY)
        else:
            continue
        
        gt_flat = gt_mask.flatten()
        pred_flat = pred_mask.flatten()
        cm = confusion_matrix(gt_flat, pred_flat, labels=[0, 1])
        TN, FP, FN, TP = cm.ravel()
        total_TP += TP
        total_FP += FP
        total_TN += TN
        total_FN += FN
        
        if TP + FP + FN > 0:
            iou = TP / (TP + FP + FN)
            all_iou.append(iou)
    
    avg_pixel_iou = np.mean(all_iou) if all_iou else 0.0
    print("Pixel-level Evaluation Results:")
    print(f"  Total Pixels: {total_TP+total_FP+total_TN+total_FN}")
    print(f"  TP={total_TP}, FP={total_FP}, TN={total_TN}, FN={total_FN}")
    print(f"  Average Pixel-level IoU (deforestation): {avg_pixel_iou:.4f}")
    
    pixel_cm = np.array([[total_TN, total_FP],
                         [total_FN, total_TP]])
    plt.figure(figsize=(5,5))
    plt.imshow(pixel_cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Pixel-level Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Background", "Deforestation"])
    plt.yticks(tick_marks, ["Background", "Deforestation"])
    thresh = pixel_cm.max() / 2.0
    for i in range(pixel_cm.shape[0]):
        for j in range(pixel_cm.shape[1]):
            plt.text(j, i, format(pixel_cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if pixel_cm[i, j] > thresh else "black")
    plt.ylabel("Ground Truth")
    plt.xlabel("Prediction")
    plt.tight_layout()
    plt.savefig("pixel_confusion_matrix.png")
    plt.close()
    print("Pixel-level confusion matrix saved to pixel_confusion_matrix.png")
    
    return total_TP, total_FP, total_TN, total_FN, avg_pixel_iou

# -----------------------------
# Part 3. Main: Prepare dataset, train model, and evaluate
# -----------------------------
def main(base_path):
    # Use the second GPU by setting CUDA_VISIBLE_DEVICES or passing device parameters.
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    
    # Step 1: Prepare dataset (convert TIFF to 4–channel PNG, generate labels, copy GT masks)
    prepare_yolo_dataset(base_path)
    
    # Step 2: Generate data.yaml file for YOLOv8 training
    data_yaml = generate_data_yaml(base_path)
    
    # Step 3: Load pretrained YOLOv8 model (nano) and modify its first conv layer for 4 channels
    model = YOLO("yolov8n.pt")
    try:
        first_conv = model.model.model[0].conv
    except AttributeError:
        raise ValueError("Cannot access first convolution layer. Verify model structure.")
    
    if first_conv.in_channels != 4:
        new_conv = torch.nn.Conv2d(
            in_channels=4,
            out_channels=first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=(first_conv.bias is not None)
        )
        torch.nn.init.kaiming_normal_(new_conv.weight, mode='fan_out', nonlinearity='relu')
        model.model.model[0].conv = new_conv
        print("Modified first convolution layer to accept 4 channels.")
    else:
        print("Model already accepts 4 channels.")
    
    # Step 4: Train the model using YOLOv8's built-in trainer on device 'cuda:1'
    model.train(data=data_yaml, imgsz=640, epochs=250, device='cuda:1')
    
    # Step 5: Evaluate detection performance (object-level)
    val_images_dir = os.path.join(base_path, "yolo", "images", "val")
    val_labels_dir = os.path.join(base_path, "yolo", "labels", "val")
    print("\n--- Detection Evaluation ---")
    evaluate_detection(model, val_images_dir, val_labels_dir, iou_threshold=0.5, imgsz=640)
    
    # Step 6: Evaluate pixel-level performance using the GT masks copied to yolo/masks/val
    val_masks_dir = os.path.join(base_path, "yolo", "masks", "val")
    print("\n--- Pixel-level Evaluation ---")
    evaluate_pixel_level(model, val_images_dir, val_masks_dir, imgsz=640)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 for deforestation detection with 4-channel data on second GPU, and evaluate using detection and pixel-level metrics."
    )
    parser.add_argument("--base_path", type=str, required=True, help="Path to the dataset root directory")
    args = parser.parse_args()
    main(args.base_path)
