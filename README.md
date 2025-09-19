# Stanford-CV-Driving-Perception

**YOLO (Car Detection) & U‑Net (Semantic Segmentation)**

> A polished, portfolio-ready collection of two Stanford workshop notebooks demonstrating core computer-vision tasks for self-driving perception: object detection with YOLO and dense semantic segmentation with U‑Net. Includes runnable notebooks, dataset structure, setup instructions, and suggestions to make the repo presentation-ready for GitHub.

---

## Table of Contents

1. [Overview](#overview)
2. [What's included](#whats-included)
3. [Quickstart (run locally)](#quickstart)
4. [Data & model files](#data--model-files)
5. [How to run each notebook](#how-to-run-each-notebook)
6. [Repository structure (recommended)](#repository-structure)
7. [Requirements / environment](#requirements--environment)
8. [Tips to make this repo portfolio-ready](#tips-to-make-this-repo-portfolio-ready)
9. [Results & expected outputs](#results--expected-outputs)
10. [License & citation](#license--citation)

---

## Overview

This repository packages two Jupyter notebooks from a Stanford computer-vision workshop (YOLO object detection and U‑Net semantic segmentation) into a single, well-documented portfolio project. The goal is to showcase practical skills in deep learning for perception tasks that appear in self-driving stacks: detecting cars with a pre-trained YOLO model, and training a U‑Net model for pixel-wise semantic segmentation on a driving dataset.

The notebooks are left mostly intact (exercises and didactic text preserved), but this README explains the code, how to run it, and how to present it on GitHub.

---

## What's included

- `notebooks/Autonomous_driving_application_Car_detection.ipynb` — YOLO-based car detection (Stanford YOLO assignment): implementations of `yolo_filter_boxes`, `iou`, `yolo_non_max_suppression`, `yolo_eval`, plus a `predict()` routine that runs YOLO on sample images and draws bounding boxes using COCO class names and anchors.

- `notebooks/Image_segmentation_Unet_v2.ipynb` — U‑Net semantic segmentation: building blocks (`conv_block`, `upsampling_block`), full `unet_model` implementation with `n_classes=23`, data loading & preprocessing (`CameraRGB` / `CameraMask`), training example (small number of epochs for demo), and utilities such as `create_mask()` and `show_predictions()`.

- Guidance files that you should add to the repo: `requirements.txt`, `figures/` (demo images, GIFs).

---

## Quickstart

1. Clone the repo:

```bash
git clone <your-repo-url>
cd Stanford-CV-Driving-Perception
```

2. Create an environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
# .venv\Scripts\activate   # Windows (PowerShell)
pip install -r requirements.txt
```

3. Place datasets and model files (see next section). Start Jupyter and open the notebooks:

```bash
jupyter lab   # or jupyter notebook
```

4. Follow the README `How to run` steps for each notebook.

**Note:** GPU is strongly recommended for training U‑Net; YOLO inference can run on CPU for small demos but is faster on GPU.

---

## Data & model files

Both notebooks expect local data directories. Create the structure below and add the required files.

```
/data (ignored in git)
  /CameraRGB/        # input RGB images for segmentation
  /CameraMask/       # corresponding ground-truth masks for segmentation
/images/             # sample images used by the YOLO notebook
/model_data/
  coco_classes.txt   # COCO class names used by YOLO notebook
  yolo_anchors.txt   # anchor box definitions
  yolo.h5            # (converted) pre-trained YOLO Keras weights
```

**Notes:**

- The segmentation notebook expects paired images & masks with identical filenames in `CameraRGB/` and `CameraMask/` (the notebook builds `mask_list` from `image_list`).
- The YOLO notebook expects `model_data/coco_classes.txt`, `model_data/yolo_anchors.txt` and `model_data/yolo.h5`. If these are not present the notebook includes instructions (and helper functions) to load them.

> For publishing the repo publicly, do **not** commit large datasets or pre-trained weights; instead provide small demo images in `assets/` and add download instructions in the README.

---

## How to run each notebook

### YOLO — `Autonomous_driving_application_Car_detection.ipynb`

1. Ensure `model_data/` contains `coco_classes.txt`, `yolo_anchors.txt`, and `yolo.h5` (converted Keras weights). There are multiple community converters and the `yad2k` utilities used by the original Stanford assignment; if you prefer, you can also adapt the notebook to use a modern TensorFlow Hub or OpenCV DNN model for YOLO.

2. Install `yad2k` helper code if needed:

```bash
pip install git+https://github.com/allanzelener/yad2k.git
```

3. Run the notebook cells sequentially. Test detection with the `predict('example.jpg')` cell (replace with a demo image in `/images`).

4. Output images are saved in `out/` and displayed inline in the notebook.

### U‑Net — `Image_segmentation_Unet_v2.ipynb`

1. Populate `data/CameraRGB/` and `data/CameraMask/` with paired images and masks.
2. Adjust `input_size` and `n_classes` if your dataset has a different resolution or number of classes. The notebook uses `input_size=(96,128,3)` and `n_classes=23` by default.
3. Run preprocessing cells to create `processed_image_ds` and then train:

```python
EPOCHS = 5
BATCH_SIZE = 32
model_history = unet.fit(train_dataset, epochs=EPOCHS)
```

4. Visualize predicted segmentation masks using `show_predictions()` and `create_mask()` helper functions.

---

## Repository structure (recommended)

```
Stanford-CV-Driving-Perception/
├─ notebooks/
│  ├─ Autonomous_driving_application_Car_detection.ipynb
│  └─ Image_segmentation_Unet_v2.ipynb
├─ requirements.txt
└─ README.md
```

---

## Requirements / environment

A minimal `requirements.txt` to get started (adjust versions as needed):

```
python>=3.8
numpy
pandas
matplotlib
pillow
scipy
jupyterlab
tensorflow
imageio
git+https://github.com/allanzelener/yad2k.git
```

**Tip:** Pin exact versions when you finalize the repo (e.g., `tensorflow==2.10.0`) and include an `environment.yml` or `Dockerfile` to guarantee reproducibility.

---

## Results & expected outputs

- YOLO notebook: bounding-boxed images saved under `out/` and displayed inline. The notebook also prints the number of detected boxes and their scores.
- U‑Net notebook: training loss/accuracy printed during `fit()`, `show_predictions()` displays input image, ground-truth mask and predicted mask side-by-side.

Include small snapshots (PNG/GIF) of both in `assets/` and embed them in the repo README to maximize impact.

---

## License & citation

This repository packages material adapted from a Stanford workshop and public implementations of YOLO / U‑Net. Use the code for learning and portfolio purposes. If you reuse the notebooks or results, please cite the original authors and the Stanford workshop.

---

*Prepared by Muhammad Adam Umar — cleaned and packaged from Stanford CV workshop notebooks.*
