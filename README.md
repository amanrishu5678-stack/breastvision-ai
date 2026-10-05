# Breast Ultrasound Image Analysis: AI-Assisted Classification (Research Prototype)

> **Research/Educational Use Only.** This project is not a medical device, does not diagnose cancer, and must
> not be used for diagnosis or treatment decisions.

## Overview

A full-stack academic project (B.Tech CSE) that:

1. trains a deep-learning image classifier on the public **BUSI** breast ultrasound dataset,
2. evaluates it honestly on a held-out test set,
3. serves it through a **Flask** REST API, and
4. presents predictions, Grad-CAM explanations and model metrics in a **React** web app.

It replaces the earlier version that used 30 manually typed numeric features (Wisconsin dataset with
Logistic Regression, Random Forest and SVM). That workflow has been removed.

**Important honesty note:** the repository contains code, not trained weights or results. Every prediction and every
number on the Dashboard comes from a model you train and evaluate yourself using the commands below. Nothing is
hardcoded or simulated.

## Problem Statement

Given a breast ultrasound image, classify it as **benign**, **malignant** or **normal** using a convolutional
neural network, and report how well the model performs on unseen images.

## Objective

- Build a reproducible image-classification pipeline (data preparation, training, evaluation, inference).
- Report honest metrics beyond accuracy (precision, recall, specificity, F1, ROC-AUC, confusion matrix).
- Expose the model through an API and a clean web interface with explainability (Grad-CAM).
- Communicate clearly that the system is educational and not clinical.

## Dataset

**Breast Ultrasound Images Dataset (BUSI)**, Al-Dhabyani W., Gomaa M., Khaled H., Fahmy A., *Data in Brief*, 2020.

| Property | Value |
| --- | --- |
| Task | 3-class image classification |
| Original release | 780 PNG images, about 500 x 500 px, from women aged 25 to 75 |
| Original class counts | normal 133, benign 437, malignant 210 |
| Extras | Segmentation masks (`*_mask.png`) are **ignored** because this project does classification |
| Cleaning | Corrupt files skipped, exact duplicate files removed (counts are recorded in `dataset_stats.json`) |
| Split | Stratified 70% train / 15% validation / 15% test, fixed seed 42 |

The exact counts **after cleaning** are computed by `prepare_dataset.py` and shown on the Dashboard.

### Dataset Source

https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset (a mirror of the dataset released with
the paper https://doi.org/10.1016/j.dib.2019.104863). Read the Kaggle page and the paper for current license and
citation terms before reusing or redistributing the data. The dataset is not included in this repository.

### Dataset Classes

`Benign`, `Malignant`, `Normal` (index order 0, 1, 2).

## Methodology

`prepare_dataset.py` -> `train.py` -> `evaluate.py` -> Flask API loads the saved model -> React UI.

Framework: **PyTorch**. Reason: it installs cleanly with pip on Windows and Python 3.12, and forward hooks and
autograd make Grad-CAM straightforward to implement correctly.

### Preprocessing

The same function (`build_transforms` in `ml/common.py`) is used in training, evaluation and the live API:
convert to RGB, resize to 224 x 224, scale to [0, 1], normalize with ImageNet mean and standard deviation.
Resizing does not preserve aspect ratio, which is a known simplification.

### Data Augmentation

Training images only: horizontal flip, rotation up to 10 degrees, brightness and contrast jitter of 0.2. No
augmentation is applied to validation, test or uploaded images.

### Model Architecture

**EfficientNet-B0** pretrained on ImageNet (torchvision), with the final layer replaced by a 3-class linear layer.

## Training

- Phase 1 (head): backbone frozen, only the new classifier is trained (5 epochs, lr 1e-3).
- Phase 2 (fine-tune): the last blocks are unfrozen (up to 15 epochs, lr 1e-4, early stopping with patience 5).
- Optimizer AdamW, cross-entropy loss with inverse-frequency class weights to handle imbalance.
- The checkpoint with the lowest **validation loss** is saved. The test set is never used during training.
- Seeds are fixed (Python, NumPy, PyTorch). GPU runs can still differ slightly between machines.

Saved to `ml/artifacts/`: `model.pt`, `config.json`, `class_names.json`, `history.json`, `splits.json`,
`dataset_stats.json`.

## Evaluation

`evaluate.py` runs the saved model on the **held-out test set** and writes `metrics.json` plus
`confusion_matrix.png`, `roc_curve.png` and `training_history.png`. Metrics: accuracy, precision, recall
(sensitivity), specificity, F1 (per class and macro-averaged), one-vs-rest ROC-AUC and the confusion matrix. The
training curves come from the training and validation sets and are labeled that way in the app.
Do not tune the model using test results, or the numbers become optimistic.

## Results

No results are stored in this README on purpose. After you run the pipeline, open the **Dashboard** page or read
`ml/artifacts/metrics.json`. Suggested table for your report (copy your own numbers in):

| Metric (test set, macro average) | Your value |
| --- | --- |
| Accuracy | |
| Precision | |
| Recall (sensitivity) | |
| Specificity | |
| F1-score | |
| ROC-AUC | |

These describe benchmark performance on one public dataset. They are not clinical accuracy.

## Backend Architecture

```
backend/
├── app.py                    # Flask app factory + all routes
├── config.py                 # env-based settings (upload limit, CORS, paths)
├── services/model_service.py # loads model once at startup, reads saved artifacts
├── utils/validation.py       # secure upload validation (in memory, nothing saved)
├── utils/errors.py           # consistent JSON errors
└── tests/test_api.py         # unit tests (python -m unittest)
```

The server never trains. It loads `ml/artifacts/model.pt` once at startup via `ml/inference.py`. If artifacts are
missing, the API stays up, reports `model_loaded: false`, and `/api/analyze` returns HTTP 503 with instructions.

Upload handling: extension, MIME type, size (default 10 MB), real image format check (PNG/JPEG only), corruption
check (`Image.verify`), minimum size (64 px) and decompression-bomb limit. Images are read into memory and are never
written to disk. Filesystem paths are never returned to the client.

## Frontend Architecture

```
frontend/src/
├── pages/       Home, Analyze, Results, Dashboard, About
├── components/  Navbar, Footer, UploadZone, ImagePreview, AnalysisLoader, Stepper,
│                PredictionResult, ProbabilityChart, MetricCard, ConfusionMatrix,
│                ModelInfo, Charts (ROC, training history, distribution),
│                Disclaimer, ErrorMessage
├── services/    api.js (single Axios client, timeout, error mapping, response validation)
├── context/     AnalysisContext (selected image + latest result)
├── hooks/       useFetch
└── utils/       format.js, validateFile.js
```

## API Documentation

Base URL: `http://localhost:5000/api`. Errors always look like
`{ "success": false, "error": { "code": "...", "message": "..." } }`.

| Method | Path | Description |
| --- | --- | --- |
| GET | `/health` | `{status, model_loaded, model_message, timestamp}` |
| GET | `/model-info` | Model config, training settings and dataset stats. 404 `MODEL_NOT_TRAINED` if no artifacts |
| GET | `/model-metrics` | `{metrics, history, dataset}` from `metrics.json`. 404 `METRICS_NOT_AVAILABLE` until you run `evaluate.py` |
| POST | `/analyze` | multipart form: `image` (file), optional `explain` (`true`/`false`). Alias: `/predict` |

`POST /analyze` success response (values shown are placeholders for the shape only):

```json
{
  "success": true,
  "prediction": {
    "class": "<class name>",
    "class_index": 0,
    "probability": 0.0,
    "probabilities": { "Benign": 0.0, "Malignant": 0.0, "Normal": 0.0 },
    "explainability": { "method": "Grad-CAM", "target_class": "<class name>", "overlay": "data:image/png;base64,..." }
  },
  "model": { "name": "EfficientNet-B0", "version": "1.0" },
  "analyzed_at": "2026-01-01T00:00:00+00:00"
}
```

Error codes: `NO_FILE`, `INVALID_TYPE` (415), `EMPTY_FILE`, `FILE_TOO_LARGE` (413), `UNREADABLE_IMAGE` (422),
`IMAGE_TOO_SMALL` (422), `MODEL_UNAVAILABLE` (503), `INFERENCE_FAILED` (500).

Test from Windows (replace the path):

```bash
curl.exe -F "image=@C:\path\to\scan.png" http://localhost:5000/api/analyze
```

## Current implementation status

The repository contains the complete image-analysis application code, but it intentionally does not include the BUSI dataset or trained model weights. Those are generated locally after the dataset is downloaded and the ML pipeline is run. This prevents the repository from shipping large third-party data or pretending that evaluation metrics exist before a real training/evaluation run.

The web application is ready for this flow:

`Upload image → validate → Flask API → EfficientNet-B0 → probabilities → optional Grad-CAM → results/dashboard`

## Installation

Requirements: Windows, Python 3.12, Node.js 18+ (LTS), Git.

```bash
cd cancer-detection-project
python -m venv venv
venv\Scripts\activate
pip install -r ml\requirements.txt
pip install -r backend\requirements.txt

cd frontend
npm install
cd ..
```

PowerShell tip: if `venv\Scripts\activate` is blocked, run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or use Command Prompt (cmd) instead.

## Running the Project

**A. Get the data** (one time): download BUSI from Kaggle and unzip into `ml\dataset\` (see `ml\dataset\README.md`).

**B. Train and evaluate** (one time, with the venv active):

```bash
cd ml
python prepare_dataset.py
python train.py
python evaluate.py
cd ..
```

The first `train.py` run downloads about 20 MB of pretrained ImageNet weights, so it needs internet. Training on a
laptop CPU is slow (expect tens of minutes); a GPU is faster. For a quick trial use
`python train.py --epochs-head 2 --epochs-finetune 3`.

**C. Start the backend** (terminal 1, venv active):

```bash
cd backend
python app.py
```

Check http://localhost:5000/api/health, where `model_loaded` should be `true`.

**D. Start the frontend** (terminal 2):

```bash
cd frontend
npm run dev
```

Open http://localhost:5173. Optional production build: `npm run build`.

**Backend tests:** `cd backend` then `python -m unittest discover -s tests -v`.

## Folder Structure

```
cancer-detection-project/
├── backend/    Flask API (see above)
├── frontend/   React + Vite app (see above)
├── ml/
│   ├── dataset/            put BUSI here (git-ignored)
│   ├── artifacts/          generated: model.pt, config.json, metrics.json, plots ...
│   ├── common.py           constants, seeds, shared preprocessing
│   ├── datasets.py         PyTorch dataset
│   ├── model_builder.py    EfficientNet-B0 definition
│   ├── prepare_dataset.py  clean + stratified split
│   ├── train.py            transfer learning
│   ├── evaluate.py         test-set metrics and plots
│   ├── metrics_utils.py    metric calculations
│   └── inference.py        prediction + Grad-CAM (used by the backend)
├── README.md
└── .gitignore
```

## Limitations

- Small dataset (hundreds of images) from limited patients and one imaging setup; results may not generalize.
- BUSI has no patient IDs, so images from one patient may land in different splits (possible leakage that inflates
  test scores). Near-duplicate images are not detected, only byte-identical ones.
- The model always outputs one of three classes. It cannot recognize non-ultrasound or out-of-scope images.
- Softmax probabilities are not calibrated medical risk.
- Grad-CAM is coarse (7 x 7) and shows model attention, not lesion location.
- No external validation and no clinical study.

## Ethical Considerations

Medical AI can harm people if trusted blindly. This project therefore: labels all output as model output, never says
it diagnoses, gives no treatment advice, shows a disclaimer on every result, does not store uploaded images, and
reports metrics on a held-out test set rather than inflated numbers. Do not upload identifiable patient data to a
demo system.

## Disclaimer

This software is for education and research only. It is not a medical device and must not be used for diagnosis,
screening or treatment decisions. Consult a qualified clinician for any health concern.

## Future Improvements

- Patient-level splits and external validation on another dataset.
- Cross-validation and confidence calibration (temperature scaling).
- Out-of-distribution detection to reject non-ultrasound images.
- Compare more architectures (ResNet, MobileNet, ConvNeXt).
- Preserve aspect ratio with padding; use lesion segmentation masks for a segmentation model.
- Higher-resolution explainability (e.g. Grad-CAM++ on earlier layers).

## Viva / Project Explanation

**Why this dataset?** BUSI is public, focused on one task (breast ultrasound), small enough to train on a laptop,
and has three clear classes. It continues the breast-cancer theme of the earlier project.

**Why this model?** EfficientNet-B0 is small, accurate for its size, well documented, and runs on CPU.

**Why transfer learning?** We only have hundreds of images. A network pretrained on ImageNet already knows edges,
textures and shapes, so we only adapt it instead of training from scratch, which would overfit.

**Why preprocessing?** The network expects a fixed input size and the same value range it was pretrained on. Using
the identical preprocessing in training and in the API keeps predictions consistent.

**What is augmentation?** Random small changes (flip, rotation, brightness) applied to training images so the model
sees more variety and overfits less. It is not applied to test images.

**What is precision?** Of all images the model called class X, the fraction that truly were X (TP / (TP + FP)).

**What is recall (sensitivity)?** Of all images that truly were X, the fraction the model found (TP / (TP + FN)).
Missing a malignant case matters, so recall for the malignant class is especially important.

**What is specificity?** Of all images that were not X, the fraction correctly called not X (TN / (TN + FP)).

**What is F1-score?** The harmonic mean of precision and recall, a single number that balances both.

**What is a confusion matrix?** A table of actual class versus predicted class. The diagonal shows correct
predictions; off-diagonal cells show which classes get confused.

**How does Flask communicate with React?** React sends HTTP requests (Axios) to Flask endpoints; Flask replies with
JSON. Browsers only allow this across ports if the server permits it, which is why the backend sets CORS headers.

**How does image upload work?** The browser validates the file, wraps it in `FormData` and POSTs it as multipart
form data. Flask validates it again (never trust the client), opens it in memory with Pillow, and does not save it.

**How does inference work?** Flask converts the image to RGB, resizes and normalizes it, runs it through the
network, applies softmax to turn scores into probabilities, and returns the top class and all probabilities.

**Why is this not a clinical diagnosis system?** It was trained on a small public dataset, was never clinically
validated, cannot say when an image is out of scope, and its probabilities are not calibrated risk. It is a learning
demonstration.

**What are the limitations?** See the Limitations section: small data, possible patient leakage, no external
validation, coarse explanations, forced three-way output.
