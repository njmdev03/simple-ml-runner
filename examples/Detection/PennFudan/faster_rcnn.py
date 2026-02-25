"""
Faster R-CNN (MobileNetV3-Large FPN) on Penn-Fudan Pedestrian Dataset.

Usage:
    # Run training + evaluation:
    python main.py job --config examples/Detection/PennFudan/faster_rcnn.py

    # Visualize results:
    python main.py vis --config examples/Detection/PennFudan/faster_rcnn.py
"""

import torch
import torch.optim as optim
import torchvision.models.detection as detection_models
import torchvision.transforms.v2 as T
from torchvision import tv_tensors

from examples.Detection.PennFudan.dataset import PennFudanDataset
from examples.Detection.detection_steps import detection_train_step, detection_eval_step

# ─── Dataset ─────────────────────────────────────────────────────────────────

_transform = T.Compose([
    T.Resize((512, 512)),
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True),
])

TRAIN_DATASET = PennFudanDataset("./data/PennFudanPed", train=True, transform=_transform)
TEST_DATASET  = PennFudanDataset("./data/PennFudanPed", train=False, transform=_transform)

# ─── Model ───────────────────────────────────────────────────────────────────

MODEL = detection_models.fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT")

# ─── Collation ───────────────────────────────────────────────────────────────
# Object detection batches are lists of (image, target) pairs with
# variable numbers of bounding boxes — they cannot be stacked into
# a single tensor. This collate_fn keeps them as a tuple of lists.

COLLATE_FN = lambda batch: tuple(zip(*batch))

# ─── Custom Step Functions ────────────────────────────────────────────────────
# These replace the standard forward-pass logic in the engine so that
# the detection model's dict-based I/O is handled correctly.

TRAIN_STEP_FN = detection_train_step
EVAL_STEP_FN  = detection_eval_step

# ─── Training ────────────────────────────────────────────────────────────────

DEVICES        = ["cuda", "cpu"]
BATCH_SIZE     = 2
EPOCHS         = 12
LEARNING_RATE  = 0.005
OPTIMIZER      = optim.SGD

# Faster R-CNN computes its own loss dict internally — no external criterion needed.
TRAIN_CRITERION = None

TRAIN = True
TEST  = True

TEST_WHILE_TRAINING = False
TEST_ON_TRAINING_DATA = True
TEST_CHECKPOINTS = True
TESTING_BATCH_SIZE  = 2

# ─── Metrics ─────────────────────────────────────────────────────────────────
# DetectionMAP wraps torchmetrics and expects list-of-dict format from EVAL_STEP_FN.
# Results are flattened into columns: mAP@0.5, mAP, mAR in the CSV.

TASK_TYPE    = "detection"
EVAL_METRICS = ["DetectionMAP"]

# ─── Early Stopping ──────────────────────────────────────────────────────────

# EARLY_HALT_CONDITION = "plateau"
# EARLY_HALT_THRESHOLD = 0.001

# ─── Checkpointing ───────────────────────────────────────────────────────────

CHECK_RATE      = 2
CHECK_MODEL_DIR = "checkpoints/"

# ─── Reporting ───────────────────────────────────────────────────────────────

SAVE_TESTS     = "results/faster_rcnn_results.csv"
PROFILE        = True
PROFILE_OUTPUT = "profiles/faster_rcnn_profile.csv"

# ─── Visualization ───────────────────────────────────────────────────────────

VIS_TYPE       = ["all"]
VIS_METRICS    = ["all"]
VIS_DATASETS   = ["all"]
VIS_OUTPUT_DIR = "vis/"
SHOW           = False
