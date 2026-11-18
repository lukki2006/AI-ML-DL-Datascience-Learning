
# ============================================================
# PCB DEFECT DETECTION - COMPLETE AUTOMATED PIPELINE
# YOLOv8 + PyTorch
#
# Dataset: DsPCBSD+
#
# Main functions:
# 1. Dataset inspection
# 2. Image and label validation
# 3. Train / Validation / Test split
# 4. Data preparation
# 5. YOLOv8 model training
# 6. Model evaluation
# 7. Precision, Recall, F1, mAP
# 8. Confusion matrices
# 9. Save trained model
# 10. Predict new PCB images
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import os
import random
import shutil
import multiprocessing
import warnings

from pathlib import Path
from collections import Counter

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import torch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from ultralytics import YOLO

warnings.filterwarnings("ignore")


# ============================================================
# 2. CONFIGURATION
# ============================================================

DATASET_PATH = Path(
    r"C:\Users\hp\Desktop\ML Model\DsPCBSD+"
)

OUTPUT_PATH = Path(
    r"C:\Users\hp\Desktop\ML Model\PCB_Results"
)

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

RUNS_PATH = OUTPUT_PATH / "training_runs"
RUN_NAME = "PCB_YOLOv8"

IMAGE_SIZE = 640

EPOCHS = 5             # Initial test: 5
BATCH_SIZE = 8
PATIENCE = 5

LEARNING_RATE = 0.001

TEST_RATIO = 0.50

CONFIDENCE_THRESHOLD = 0.25

SEED = 42

MODEL_NAME = "yolov8n.pt"

# Set a path here if you want to predict a specific image.
# Leave None to open a file selection window.

PREDICT_IMAGE = None

# Example:
# PREDICT_IMAGE = r"C:\Users\hp\Desktop\test_pcb.jpg"


# ============================================================
# 3. REPRODUCIBILITY AND DEVICE
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():

    DEVICE = 0

    torch.cuda.manual_seed_all(SEED)

    print("\nGPU AVAILABLE")

    print(torch.cuda.get_device_name(0))

else:

    DEVICE = "cpu"

    print("\nGPU NOT AVAILABLE - USING CPU")


print("\nPyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())


# ============================================================
# 4. DATASET DIRECTORY DETECTION
# ============================================================

DATA_YOLO_PATH = DATASET_PATH / "Data_YOLO"

if not DATA_YOLO_PATH.exists():

    DATA_YOLO_PATH = DATASET_PATH

TRAIN_IMAGES_DIR = DATA_YOLO_PATH / "images" / "train"

VAL_IMAGES_DIR = DATA_YOLO_PATH / "images" / "val"

TRAIN_LABELS_DIR = DATA_YOLO_PATH / "labels" / "train"

VAL_LABELS_DIR = DATA_YOLO_PATH / "labels" / "val"


required_directories = [
    TRAIN_IMAGES_DIR,
    VAL_IMAGES_DIR,
    TRAIN_LABELS_DIR,
    VAL_LABELS_DIR
]

for directory in required_directories:

    if not directory.exists():

        raise FileNotFoundError(
            f"Dataset directory not found: {directory}"
        )


print("\nDataset directory detected:")
print(DATA_YOLO_PATH)


# ============================================================
# 5. READ ORIGINAL DATASET YAML
# ============================================================

def find_dataset_yaml():

    candidates = (
        list(DATA_YOLO_PATH.glob("*.yaml"))
        + list(DATA_YOLO_PATH.glob("*.yml"))
        + list(DATASET_PATH.glob("*.yaml"))
        + list(DATASET_PATH.glob("*.yml"))
    )

    # Remove duplicates while preserving order
    candidates = list(dict.fromkeys(candidates))

    for yaml_file in candidates:

        try:

            with open(
                yaml_file,
                "r",
                encoding="utf-8"
            ) as file:

                data = yaml.safe_load(file)

            if isinstance(data, dict) and "names" in data:

                return yaml_file, data

        except Exception:

            continue

    raise FileNotFoundError(
        "Original dataset YAML containing class names was not found. "
        "Please place the original data.yaml file inside Data_YOLO."
    )


ORIGINAL_YAML_PATH, ORIGINAL_YAML = find_dataset_yaml()

raw_names = ORIGINAL_YAML["names"]

if isinstance(raw_names, dict):

    CLASS_NAMES = [
        str(raw_names[key])
        for key in sorted(raw_names, key=lambda x: int(x))
    ]

elif isinstance(raw_names, list):

    CLASS_NAMES = [str(name) for name in raw_names]

else:

    raise ValueError(
        "Unsupported class names format in dataset YAML."
    )


NUM_CLASSES = len(CLASS_NAMES)

print("\nOriginal dataset YAML:", ORIGINAL_YAML_PATH)

print("\nNumber of classes:", NUM_CLASSES)

print("\nClass names:")

for i, name in enumerate(CLASS_NAMES):

    print(i, ":", name)

if ORIGINAL_YAML.get("nc") is not None:

    if int(ORIGINAL_YAML["nc"]) != NUM_CLASSES:

        raise ValueError(
            "The nc value in YAML does not match the number of class names."
        )


# ============================================================
# 6. COLLECT IMAGE FILES
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp"
}


def collect_images(folder):

    images = [

        path

        for path in folder.rglob("*")

        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS

    ]

    return sorted(images)


train_images = collect_images(TRAIN_IMAGES_DIR)

val_images = collect_images(VAL_IMAGES_DIR)


print("\nDATASET SUMMARY")
print("=" * 50)

print("Training images:", len(train_images))
print("Validation images:", len(val_images))

if len(train_images) == 0:

    raise ValueError("No training images found.")

if len(val_images) < 2:

    raise ValueError(
        "At least two original validation images are required "
        "to create separate validation and test sets."
    )


# ============================================================
# 7. IMAGE AND LABEL PATH FUNCTIONS
# ============================================================

def get_label_path(image_path):

    image_path = Path(image_path)

    parts = list(image_path.parts)

    if "images" not in parts:

        raise ValueError(
            f"Cannot determine label path for: {image_path}"
        )

    index = parts.index("images")

    parts[index] = "labels"

    return Path(*parts).with_suffix(".txt")


# ============================================================
# 8. IMAGE AND LABEL VALIDATION
# ============================================================

def validate_dataset(images, split_name):

    print(f"\nChecking {split_name} dataset...")

    corrupt_images = []

    missing_labels = []

    invalid_labels = []

    empty_labels = 0

    class_counter = Counter()

    total_objects = 0

    for image_path in images:

        # Check image readability

        image = cv2.imread(str(image_path))

        if image is None:

            corrupt_images.append(str(image_path))

            continue

        height, width = image.shape[:2]

        if height <= 0 or width <= 0:

            corrupt_images.append(str(image_path))

            continue

        label_path = get_label_path(image_path)

        # Missing labels are treated as background images.
        # This is valid for YOLO datasets containing negative images.

        if not label_path.exists():

            missing_labels.append(str(label_path))

            continue

        try:

            with open(
                label_path,
                "r",
                encoding="utf-8"
            ) as file:

                lines = file.readlines()

        except Exception:

            invalid_labels.append(
                f"{label_path}: cannot read file"
            )

            continue

        if len(lines) == 0:

            empty_labels += 1

            continue

        for line_number, line in enumerate(lines, start=1):

            values = line.strip().split()

            if len(values) != 5:

                invalid_labels.append(
                    f"{label_path}, line {line_number}: "
                    "Expected 5 values."
                )

                continue

            try:

                class_float = float(values[0])

                class_id = int(class_float)

                x_center = float(values[1])
                y_center = float(values[2])
                box_width = float(values[3])
                box_height = float(values[4])

            except ValueError:

                invalid_labels.append(
                    f"{label_path}, line {line_number}: "
                    "Non-numeric values."
                )

                continue

            if class_float != class_id:

                invalid_labels.append(
                    f"{label_path}, line {line_number}: "
                    "Class ID must be an integer."
                )

                continue

            if not 0 <= class_id < NUM_CLASSES:

                invalid_labels.append(
                    f"{label_path}, line {line_number}: "
                    f"Invalid class ID {class_id}."
                )

                continue

            coordinates = [
                x_center,
                y_center,
                box_width,
                box_height
            ]

            if not np.all(np.isfinite(coordinates)):

                invalid_labels.append(
                    f"{label_path}, line {line_number}: "
                    "Non-finite coordinates."
                )

                continue

            if not (
                0 <= x_center <= 1
                and 0 <= y_center <= 1
                and 0 < box_width <= 1
                and 0 < box_height <= 1
            ):

                invalid_labels.append(
                    f"{label_path}, line {line_number}: "
                    "Coordinates outside valid YOLO range."
                )

                continue

            class_counter[class_id] += 1

            total_objects += 1

    print("Images checked:", len(images))

    print("Corrupt images:", len(corrupt_images))

    print("Missing label files:", len(missing_labels))

    print("Empty label files:", empty_labels)

    print("Total valid objects:", total_objects)

    print("Invalid label records:", len(invalid_labels))

    if corrupt_images:

        print("\nFirst corrupt images:")

        for item in corrupt_images[:10]:

            print(item)

    if invalid_labels:

        print("\nFirst invalid labels:")

        for item in invalid_labels[:10]:

            print(item)

    if corrupt_images or invalid_labels:

        raise ValueError(
            f"Dataset validation failed for {split_name}. "
            "Correct the listed image or label problems before training."
        )

    return class_counter


train_class_counts = validate_dataset(
    train_images,
    "TRAIN"
)

val_class_counts = validate_dataset(
    val_images,
    "VALIDATION"
)


# ============================================================
# 9. CLASS DISTRIBUTION REPORT
# ============================================================

distribution = []

for class_id, class_name in enumerate(CLASS_NAMES):

    train_count = train_class_counts[class_id]

    val_count = val_class_counts[class_id]

    distribution.append({

        "Class_ID": class_id,

        "Class": class_name,

        "Train_Objects": train_count,

        "Validation_Objects": val_count,

        "Total_Objects": train_count + val_count

    })


class_distribution = pd.DataFrame(distribution)

print("\nCLASS DISTRIBUTION")

print(class_distribution.to_string(index=False))

class_distribution.to_csv(
    OUTPUT_PATH / "class_distribution.csv",
    index=False
)


plt.figure(figsize=(12, 6))

plt.bar(
    class_distribution["Class"],
    class_distribution["Total_Objects"]
)

plt.xlabel("Defect Class")

plt.ylabel("Number of Annotated Objects")

plt.title("PCB Defect Class Distribution")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    OUTPUT_PATH / "class_distribution.png",
    dpi=300
)

plt.close()


# ============================================================
# 10. TRAIN / VALIDATION / TEST SPLIT
# ============================================================

print("\nCreating train / validation / test split...")

random.shuffle(val_images)

test_count = max(
    1,
    int(len(val_images) * TEST_RATIO)
)

test_images = val_images[:test_count]

validation_images = val_images[test_count:]

if len(validation_images) == 0:

    raise ValueError(
        "Validation set is empty after splitting."
    )


print("\nFINAL DATA SPLIT")

print("Training:", len(train_images))

print("Validation:", len(validation_images))

print("Test:", len(test_images))


# ============================================================
# 11. CREATE IMAGE LIST FILES
# ============================================================

SPLIT_PATH = OUTPUT_PATH / "dataset_split"

SPLIT_PATH.mkdir(parents=True, exist_ok=True)


def write_image_list(images, filename):

    file_path = SPLIT_PATH / filename

    with open(
        file_path,
        "w",
        encoding="utf-8"
    ) as file:

        for image in images:

            file.write(
                str(Path(image).resolve()) + "\n"
            )

    return file_path


train_list = write_image_list(
    train_images,
    "train.txt"
)

val_list = write_image_list(
    validation_images,
    "val.txt"
)

test_list = write_image_list(
    test_images,
    "test.txt"
)


# ============================================================
# 12. CREATE YOLO DATASET YAML
# ============================================================

YOLO_YAML_PATH = OUTPUT_PATH / "pcb_dataset.yaml"

yolo_data = {

    "path": str(DATASET_PATH.resolve()),

    "train": str(train_list.resolve()),

    "val": str(val_list.resolve()),

    "test": str(test_list.resolve()),

    "nc": NUM_CLASSES,

    "names": CLASS_NAMES

}


with open(
    YOLO_YAML_PATH,
    "w",
    encoding="utf-8"
) as file:

    yaml.safe_dump(
        yolo_data,
        file,
        sort_keys=False,
        allow_unicode=True
    )


print("\nYOLO dataset configuration created:")

print(YOLO_YAML_PATH)


# ============================================================
# 13. CREATE AND TRAIN YOLO MODEL
# ============================================================

print("\nLoading YOLOv8 model...")

model = YOLO(MODEL_NAME)


print("\nSTARTING MODEL TRAINING")

print("=" * 60)


train_results = model.train(

    data=str(YOLO_YAML_PATH),

    epochs=EPOCHS,

    imgsz=IMAGE_SIZE,

    batch=BATCH_SIZE,

    patience=PATIENCE,

    device=DEVICE,

    workers=2,

    seed=SEED,

    pretrained=True,

    optimizer="AdamW",

    lr0=LEARNING_RATE,

    project=str(RUNS_PATH),

    name=RUN_NAME,

    exist_ok=True,

    # Data augmentation

    hsv_h=0.015,

    hsv_s=0.7,

    hsv_v=0.4,

    degrees=2.0,

    translate=0.05,

    scale=0.2,

    fliplr=0.5,

    flipud=0.0,

    mosaic=1.0,

    close_mosaic=10,

    # Output

    plots=True,

    save=True,

    verbose=True

)


# ============================================================
# 14. LOCATE BEST MODEL
# ============================================================

run_directory = Path(model.trainer.save_dir)

best_model_path = (
    run_directory / "weights" / "best.pt"
)

if not best_model_path.exists():

    raise FileNotFoundError(
        f"Best model was not found: {best_model_path}"
    )


FINAL_MODEL_PATH = (
    OUTPUT_PATH / "PCB_Defect_Final.pt"
)

shutil.copy2(
    best_model_path,
    FINAL_MODEL_PATH
)


print("\nBEST MODEL SAVED")

print(FINAL_MODEL_PATH)


# ============================================================
# 15. LOAD TRAINED MODEL
# ============================================================

best_model = YOLO(str(FINAL_MODEL_PATH))


# ============================================================
# 16. VALIDATION EVALUATION
# ============================================================

print("\nVALIDATION EVALUATION")

val_metrics = best_model.val(

    data=str(YOLO_YAML_PATH),

    split="val",

    imgsz=IMAGE_SIZE,

    batch=BATCH_SIZE,

    device=DEVICE,

    plots=True,

    project=str(OUTPUT_PATH / "evaluation"),

    name="validation",

    exist_ok=True

)


# ============================================================
# 17. TEST EVALUATION
# ============================================================

print("\nTEST EVALUATION")

test_metrics = best_model.val(

    data=str(YOLO_YAML_PATH),

    split="test",

    imgsz=IMAGE_SIZE,

    batch=BATCH_SIZE,

    device=DEVICE,

    plots=True,

    project=str(OUTPUT_PATH / "evaluation"),

    name="test",

    exist_ok=True

)


# ============================================================
# 18. CLASS-BASED DETECTION METRICS
# ============================================================

print("\nCLASS-BASED DETECTION PERFORMANCE")

box_metrics = test_metrics.box

# Initialize metrics for every class.
# This prevents incorrect indexing when some classes
# are absent from the test evaluation.

p_values = np.zeros(NUM_CLASSES)

r_values = np.zeros(NUM_CLASSES)

map50_values = np.zeros(NUM_CLASSES)

map50_95_values = np.asarray(
    box_metrics.maps
)

# Get actual class IDs evaluated by Ultralytics.

class_ids = np.asarray(
    box_metrics.ap_class_index,
    dtype=int
)

# Map metric values to the original class IDs.

for j, class_id in enumerate(class_ids):

    p_values[class_id] = box_metrics.p[j]

    r_values[class_id] = box_metrics.r[j]

    map50_values[class_id] = box_metrics.ap50[j]


# Build class performance report.

class_results = []

for i, class_name in enumerate(CLASS_NAMES):

    precision = p_values[i]

    recall = r_values[i]

    f1 = (

        2 * precision * recall / (precision + recall)

        if precision + recall > 0

        else 0

    )

    class_results.append({

        "Class_ID": i,

        "Class": class_name,

        "Precision": precision,

        "Recall": recall,

        "F1_Score": f1,

        "mAP50": map50_values[i],

        "mAP50_95": map50_95_values[i]

    })


class_performance = pd.DataFrame(
    class_results
)


print("\nCLASS PERFORMANCE REPORT")

print("=" * 80)

print(

    class_performance.to_string(

        index=False,

        float_format=lambda x: f"{x:.4f}"

    )

)


class_performance.to_csv(

    OUTPUT_PATH / "class_performance.csv",

    index=False

)


# ============================================================
# 19. OVERALL DETECTION METRICS
# ============================================================

overall_results = {

    "Precision": float(box_metrics.mp),

    "Recall": float(box_metrics.mr),

    "mAP50": float(box_metrics.map50),

    "mAP50_95": float(box_metrics.map)

}


overall_df = pd.DataFrame(
    [overall_results]
)

print("\nOVERALL DETECTION METRICS")

print(overall_df.to_string(index=False))

overall_df.to_csv(

    OUTPUT_PATH / "test_metrics.csv",

    index=False

)


# ============================================================
# 20. IMAGE-LEVEL CLASS PRESENCE EVALUATION
# ============================================================

print("\nIMAGE-LEVEL CLASS PRESENCE EVALUATION")

print(
    "This is an additional classification-style evaluation, "
    "not object detection accuracy."
)


y_true = []

y_pred = []


for index, image_path in enumerate(test_images):

    if index % 50 == 0:

        print(
            f"Processing image {index + 1}/{len(test_images)}"
        )

    true_vector = np.zeros(NUM_CLASSES)

    pred_vector = np.zeros(NUM_CLASSES)

    # Read ground truth labels

    label_path = get_label_path(image_path)

    if label_path.exists():

        with open(
            label_path,
            "r",
            encoding="utf-8"
        ) as file:

            for line in file:

                values = line.strip().split()

                if len(values) == 5:

                    class_id = int(float(values[0]))

                    true_vector[class_id] = 1

    # Predict image

    predictions = best_model.predict(

        source=str(image_path),

        imgsz=IMAGE_SIZE,

        conf=CONFIDENCE_THRESHOLD,

        device=DEVICE,

        verbose=False

    )

    result = predictions[0]

    if result.boxes is not None:

        detected_classes = (
            result.boxes.cls.cpu().numpy().astype(int)
        )

        for class_id in detected_classes:

            pred_vector[class_id] = 1

    y_true.append(true_vector)

    y_pred.append(pred_vector)


y_true = np.asarray(y_true)

y_pred = np.asarray(y_pred)


# Exact-match accuracy:
# All class-presence decisions must match for an image.

image_accuracy = accuracy_score(
    y_true,
    y_pred
)

micro_precision = precision_score(
    y_true,
    y_pred,
    average="micro",
    zero_division=0
)

micro_recall = recall_score(
    y_true,
    y_pred,
    average="micro",
    zero_division=0
)

micro_f1 = f1_score(
    y_true,
    y_pred,
    average="micro",
    zero_division=0
)

macro_f1 = f1_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)


image_metrics = {

    "Exact_Match_Accuracy": image_accuracy,

    "Micro_Precision": micro_precision,

    "Micro_Recall": micro_recall,

    "Micro_F1": micro_f1,

    "Macro_F1": macro_f1

}


image_metrics_df = pd.DataFrame(
    [image_metrics]
)


print("\nIMAGE-LEVEL METRICS")

print(image_metrics_df.to_string(index=False))


image_metrics_df.to_csv(

    OUTPUT_PATH / "image_level_metrics.csv",

    index=False

)


# ============================================================
# 21. IMAGE-LEVEL CONFUSION MATRICES
# ============================================================

print("\nCreating image-level confusion matrices...")


fig, axes = plt.subplots(

    3,

    3,

    figsize=(15, 13)

)

axes = axes.flatten()


for i, class_name in enumerate(CLASS_NAMES):

    cm = confusion_matrix(

        y_true[:, i],

        y_pred[:, i],

        labels=[0, 1]

    )

    sns.heatmap(

        cm,

        annot=True,

        fmt="d",

        cmap="Blues",

        xticklabels=["No Defect", "Defect"],

        yticklabels=["No Defect", "Defect"],

        ax=axes[i]

    )

    axes[i].set_title(class_name)

    axes[i].set_xlabel("Predicted")

    axes[i].set_ylabel("Actual")


# Hide unused axes if fewer than nine classes.

for i in range(NUM_CLASSES, len(axes)):

    axes[i].axis("off")


plt.tight_layout()

plt.savefig(

    OUTPUT_PATH / "image_level_confusion_matrices.png",

    dpi=300

)

plt.close()


# ============================================================
# 22. TRAINING CURVES
# ============================================================

print("\nSaving training curves...")


results_csv = run_directory / "results.csv"


if results_csv.exists():

    history = pd.read_csv(results_csv)

    history.columns = [
        column.strip()
        for column in history.columns
    ]

    # Plot training and validation losses.

    loss_columns = [

        column

        for column in history.columns

        if "loss" in column.lower()

    ]

    if loss_columns:

        plt.figure(figsize=(12, 6))

        for column in loss_columns:

            plt.plot(
                history[column],
                label=column
            )

        plt.title("YOLO Training and Validation Loss")

        plt.xlabel("Epoch")

        plt.ylabel("Loss")

        plt.legend()

        plt.grid(True)

        plt.tight_layout()

        plt.savefig(

            OUTPUT_PATH / "training_loss.png",

            dpi=300

        )

        plt.close()


    # Plot detection metrics.

    metric_columns = [

        column

        for column in history.columns

        if "metrics/" in column.lower()

    ]

    if metric_columns:

        plt.figure(figsize=(12, 6))

        for column in metric_columns:

            plt.plot(
                history[column],
                label=column
            )

        plt.title("YOLO Validation Metrics")

        plt.xlabel("Epoch")

        plt.ylabel("Metric Value")

        plt.legend()

        plt.grid(True)

        plt.tight_layout()

        plt.savefig(

            OUTPUT_PATH / "training_metrics.png",

            dpi=300

        )

        plt.close()


# ============================================================
# 23. SAVE TRAINING SUMMARY
# ============================================================

summary = {

    "Model": MODEL_NAME,

    "Number_of_Classes": NUM_CLASSES,

    "Training_Images": len(train_images),

    "Validation_Images": len(validation_images),

    "Test_Images": len(test_images),

    "Epochs_Configured": EPOCHS,

    "Image_Size": IMAGE_SIZE,

    "Batch_Size": BATCH_SIZE,

    "Device": str(DEVICE),

    "Model_Path": str(FINAL_MODEL_PATH),

    "Test_Precision": float(box_metrics.mp),

    "Test_Recall": float(box_metrics.mr),

    "Test_mAP50": float(box_metrics.map50),

    "Test_mAP50_95": float(box_metrics.map)

}


summary_df = pd.DataFrame(
    [summary]
)

summary_df.to_csv(

    OUTPUT_PATH / "project_summary.csv",

    index=False

)


# ============================================================
# 24. PREDICT A NEW PCB IMAGE
# ============================================================

def select_prediction_image():

    if PREDICT_IMAGE is not None:

        image_path = Path(PREDICT_IMAGE)

        if not image_path.exists():

            raise FileNotFoundError(
                f"Prediction image not found: {image_path}"
            )

        return image_path

    try:

        import tkinter as tk

        from tkinter import filedialog

        root = tk.Tk()

        root.withdraw()

        root.attributes("-topmost", True)

        selected_file = filedialog.askopenfilename(

            title="Select a PCB image",

            filetypes=[

                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tif"),

                ("All files", "*.*")

            ]

        )

        root.destroy()

        if selected_file:

            return Path(selected_file)

    except Exception as error:

        print("File selection window unavailable:", error)

    return None


print("\nSELECT A NEW PCB IMAGE")

prediction_image = select_prediction_image()


if prediction_image is not None:

    print("\nPredicting:", prediction_image)

    prediction_results = best_model.predict(

        source=str(prediction_image),

        imgsz=IMAGE_SIZE,

        conf=CONFIDENCE_THRESHOLD,

        device=DEVICE,

        save=False,

        verbose=False

    )

    result = prediction_results[0]

    # Save annotated image.

    annotated_image = result.plot()

    prediction_output = OUTPUT_PATH / "prediction_annotated.jpg"

    cv2.imwrite(

        str(prediction_output),

        annotated_image

    )

    # Extract detection details.

    detections = []

    if result.boxes is not None:

        for box in result.boxes:

            class_id = int(box.cls.item())

            confidence = float(box.conf.item())

            coordinates = (
                box.xyxy[0].cpu().numpy()
            )

            detections.append({

                "Class_ID": class_id,

                "Defect_Class": CLASS_NAMES[class_id],

                "Confidence": confidence,

                "X1": coordinates[0],

                "Y1": coordinates[1],

                "X2": coordinates[2],

                "Y2": coordinates[3]

            })


    detection_df = pd.DataFrame(

        detections,

        columns=[

            "Class_ID",

            "Defect_Class",

            "Confidence",

            "X1",

            "Y1",

            "X2",

            "Y2"

        ]

    )


    detection_df.to_csv(

        OUTPUT_PATH / "prediction_report.csv",

        index=False

    )


    print("\nPREDICTION RESULTS")

    if len(detection_df) > 0:

        print(detection_df.to_string(index=False))

        print(
            "\nDetected defects:",
            len(detection_df)
        )

    else:

        print("No defects detected above the confidence threshold.")


    print("\nAnnotated image saved:")

    print(prediction_output)


else:

    print("\nNo prediction image selected.")

    print("You can run the prediction section again later.")


# ============================================================
# 25. FINAL REPORT
# ============================================================

print("\n" + "=" * 60)

print("PCB DEFECT DETECTION PIPELINE COMPLETED")

print("=" * 60)

print("\nFinal model:")

print(FINAL_MODEL_PATH)

print("\nResults folder:")

print(OUTPUT_PATH)

print("\nGenerated reports:")

for file in sorted(OUTPUT_PATH.rglob("*")):

    if file.is_file():

        print(file)

print("\nPROCESS FINISHED SUCCESSFULLY.")


# ============================================================
# END OF PROJECT
# ============================================================


if __name__ == "__main__":

    multiprocessing.freeze_support()
