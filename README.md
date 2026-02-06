# Surgery Phase Detection

Automatic detection of surgical phases from operating room video data using deep learning. This project analyzes multi-view OR camera feeds to identify what phase of a procedure is happening, enabling hospitals to analyze surgery efficiency, track workflow deviations, and improve training.

## Dataset: MVOR

This project uses the [MVOR (Multi-View Operating Room)](https://github.com/CAMMA-public/MVOR) dataset from CAMMA, University of Strasbourg. MVOR contains **732 synchronized multi-view frames** from **3 RGB-D cameras** recorded over **4 days** in a hybrid operating room during procedures such as vertebroplasty and lung biopsy.

Key dataset features:
- 2,196 images (732 per camera) across 4 recording days
- 4,699 person bounding box annotations
- 2,926 2D keypoint annotations (10 upper-body joints)
- 1,061 3D keypoint annotations
- Person role labels: **clinician** and **patient**

## Phases Detected

Since MVOR does not include explicit surgical phase labels, we derive **OR activity phases** from scene context — the number and roles of people present, their poses, and temporal position:

| Phase ID | Phase Name       | Description |
|----------|------------------|-------------|
| 0 | Idle/Empty        | No people or minimal background activity |
| 1 | Preparation       | Patient positioned, clinicians arriving |
| 2 | Procedure Active  | Multiple clinicians actively engaged |
| 3 | Closure/Cleanup   | Activity winding down, clinicians departing |

These map to the general surgical workflow: **Preparation** → **Incision/Active Procedure** → **Suturing/Closure**.

## Architecture

Three model variants are provided, including an implementation inspired by the state-of-the-art **PreViPS** framework ([arXiv:2502.13883](https://arxiv.org/abs/2502.13883)):

### 1. Single-View Model (ResNet + LSTM)
1. **ResNet-50** backbone (pretrained on ImageNet) extracts per-frame spatial features.
2. **Multi-layer LSTM** processes frame sequences to capture temporal workflow patterns.

### 2. Multi-View Fusion Model
1. **Shared ResNet-50** backbone processes each camera view.
2. **View Fusion** combines features from all 3 cameras (attention, concat, mean, or max pooling).
3. **LSTM** models temporal dependencies across fused multi-view features.

### 3. PreViPS: Video-Pose Dual-Encoder (arXiv:2502.13883)
A CLIP-style calibration-free multi-view multi-modal framework:
1. **Video Encoder** — ResNet-50 (or MViT-S) extracts global visual features per camera view.
2. **Pose Tokenizer** — VQ-VAE converts continuous 2D keypoints into discrete Pose Compositional Tokens (PCT).
3. **Pose Transformer** — Processes tokenized pose sequences with spatio-temporal positional embeddings.
4. **Pretraining** — Aligns video and pose embeddings across camera views using:
   - Cross-modality contrastive learning (video ↔ pose, CLIP-style InfoNCE)
   - In-modality alignment (video ↔ video, pose ↔ pose across views)
   - Cross-modal and in-modal geometric consistency regularizers
   - Masked pose token prediction (MAE-style)
5. **Finetuning** — Average-pools global tokens from all views/modalities → MLP classifier.

## Project Structure

```
surgery-phase-detection/
├── configs/
│   ├── default.yaml              # Single-view MVOR config
│   ├── mvor_multiview.yaml       # Multi-view fusion config
│   └── previps.yaml              # PreViPS pretraining config
├── surgery_phase_detection/
│   ├── __init__.py
│   ├── data/
│   │   ├── dataset.py            # Cholec80 dataset loader (alternative)
│   │   ├── mvor_dataset.py       # MVOR dataset loaders
│   │   ├── phase_labeler.py      # Derive phase labels from MVOR annotations
│   │   └── transforms.py         # Data augmentation and preprocessing
│   ├── models/
│   │   ├── resnet_lstm.py        # Single-view ResNet + LSTM
│   │   ├── multiview_net.py      # Multi-view fusion model
│   │   ├── previps.py            # PreViPS dual-encoder framework
│   │   ├── pose_encoder.py       # Pose tokenizer + transformer
│   │   └── feature_extractor.py  # Standalone feature extractor
│   └── utils/
│       ├── metrics.py            # Evaluation metrics (Jaccard, F1, etc.)
│       └── visualization.py      # Confusion matrices, timelines, curves
├── scripts/
│   ├── download_mvor.py          # Download and setup the MVOR dataset
│   ├── pretrain.py               # PreViPS video-pose pretraining
│   ├── train.py                  # Supervised training / finetuning
│   ├── evaluate.py               # Evaluation script
│   ├── predict.py                # Inference on new videos/images
│   └── extract_frames.py         # Extract frames from video files
├── notebooks/
│   └── exploration.ipynb         # Data exploration notebook
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the MVOR Dataset

```bash
# Full dataset (images + annotations, ~2 GB)
python scripts/download_mvor.py --output_dir data/mvor

# Annotations only (for development without images)
python scripts/download_mvor.py --output_dir data/mvor --annotations_only
```

This downloads the dataset from [CAMMA's S3 server](https://s3.unistra.fr/camma_public/datasets/mvor/camma_mvor_dataset.zip) and sets up the expected directory structure:

```
data/mvor/
├── annotations/
│   └── camma_mvor_2018.json
├── camma_mvor_dataset/
│   ├── day1/
│   │   ├── cam1/color/
│   │   ├── cam2/color/
│   │   └── cam3/color/
│   ├── day2/
│   ├── day3/
│   └── day4/
└── MVOR/                       # Cloned repo with eval scripts
```

### 3. Train the Model

```bash
# Single-view model (default)
python scripts/train.py --config configs/default.yaml

# Multi-view fusion model
python scripts/train.py --config configs/mvor_multiview.yaml

# PreViPS: pretrain video-pose alignment, then finetune
python scripts/pretrain.py --config configs/previps.yaml
python scripts/train.py --config configs/previps.yaml --resume checkpoints/pretrain_last.pth
```

### 4. Evaluate

```bash
python scripts/evaluate.py \
    --config configs/default.yaml \
    --checkpoint checkpoints/best_model.pth \
    --output_dir results/
```

### 5. Run Inference

```bash
# On a directory of OR images
python scripts/predict.py \
    --frames_dir path/to/or_images/ \
    --checkpoint checkpoints/best_model.pth \
    --visualize

# On a video file
python scripts/predict.py \
    --video_path path/to/or_video.mp4 \
    --checkpoint checkpoints/best_model.pth \
    --output_path results/prediction.json
```

## Configuration

Key settings in `configs/default.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dataset.name` | mvor | Dataset to use (mvor or cholec80) |
| `dataset.train_days` | [2, 3] | MVOR recording days for training |
| `dataset.val_days` | [4] | MVOR recording days for validation |
| `dataset.test_days` | [1] | MVOR recording days for testing |
| `model.type` | single_view | Model type (single_view or multi_view) |
| `model.backbone` | resnet50 | CNN backbone architecture |
| `model.lstm_hidden` | 512 | LSTM hidden dimension |
| `model.sequence_length` | 5 | Frames per temporal sequence |
| `model.fusion` | attention | Multi-view fusion strategy |
| `training.batch_size` | 4 | Training batch size |
| `training.learning_rate` | 1e-4 | Initial learning rate |
| `training.epochs` | 30 | Maximum training epochs |

## Phase Labeling Strategy

The `MVORPhaseLabeler` derives phase labels from MVOR scene annotations using:

1. **Scene occupancy**: Number of people detected in the multi-view frame
2. **Person roles**: Whether clinicians and/or patients are present
3. **Temporal position**: Where the frame falls within the day's timeline
4. **Spatial features**: Bounding box areas and keypoint patterns
5. **Temporal smoothing**: Majority-vote sliding window to reduce noise

The labeling rules follow the observation that surgical workflow phases correlate strongly with OR occupancy patterns (who is in the room and when).

## Metrics

The model is evaluated using:

- **Accuracy** — Overall frame-level accuracy
- **Precision / Recall / F1** — Per-phase and macro-averaged
- **Jaccard Index** — Intersection over union per phase
- **Confusion Matrix** — Phase-level confusion visualization

## Requirements

- Python >= 3.9
- PyTorch >= 2.0
- CUDA-capable GPU recommended

## References

- Hamoud, I., et al. "Multi-view Video-Pose Pretraining for Operating Room Surgical Activity Recognition." arXiv:2502.13883, 2025. [Paper](https://arxiv.org/abs/2502.13883) | [Code](https://github.com/CAMMA-public/PreViPS)
- Srivastav, V., et al. "MVOR: A Multi-view RGB-D Operating Room Dataset for 2D and 3D Human Pose Estimation." MICCAI-LABELS, 2018. [arXiv:1808.08180](https://arxiv.org/abs/1808.08180)
- Twinanda, A.P., et al. "EndoNet: A Deep Architecture for Recognition Tasks on Laparoscopic Videos." IEEE TMI, 2017.
- Jiang, T., et al. "PCT: Pose as Compositional Tokens." arXiv:2303.17428, 2023.

## License

This project is for research and educational purposes. The MVOR dataset is released under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Please refer to the [MVOR repository](https://github.com/CAMMA-public/MVOR) for full dataset terms.
