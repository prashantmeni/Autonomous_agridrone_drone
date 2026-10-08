# Training and export pipeline for the crop-disease classifier

Produces `disease_model.onnx` for the production inference module in
`ai_disease_detection/`.

**No model is committed.** Training artifacts live in `output/` (gitignored);
`ai_disease_detection/models/` gains `disease_model.onnx` only when it is
deployed (step 6). Do not claim inference works before then.

> **Status after the 2026-10-08 smoke-test run (Colab, 2 epochs):**
> **Model: smoke test only.** `output/model2.h5`; train accuracy 0.9340,
> val accuracy 0.7370, val loss 0.79981 (best weights restored from epoch 2).
> **Evaluation (validation split — NOT held-out):** Top-1 0.7370, Top-5 0.9702,
> macro F1 0.7425. The Kaggle dataset has no usable `test/` split, so these
> are validation figures and are optimistic.
> **ONNX export: validated.** `disease_model.onnx` (15.5 MB, opset 13, NCHW);
> Keras vs ONNX agreement 10/10, worst probability diff 1.788e-06 (tol 0.01).
> **Raspberry Pi performance (Pi 4 Model B):** inference mean 96.6 ms, min
> 94.8, max 106.2 ≈ 10.4 FPS (measured 2026-10-08).

---

## What this reproduces

From the reference repository
([isupercoder401/Agricultural-Drone…](https://github.com/isupercoder401/Agricultural-Drone-For-Crop-Monitoring-and-Early-Disease-Detection)),
`project.py`:

```python
base_model = tf.keras.applications.EfficientNetB7(include_top=False)  # L60
base_model.trainable = False                                          # L61
x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)      # L64
outputs = tf.keras.layers.Dense(38, activation="softmax")(x)         # L66
```

| Item | Value |
|---|---|
| Dataset | New Plant Diseases Dataset (PlantVillage), via Kaggle `vipoooool/new-plant-diseases-dataset` |
| Architecture | `EfficientNet*(include_top=False)` → GAP → Dense(38, softmax) |
| Backbone | **`EfficientNetB0`** (ours; reference used B7) — see "Backbone migration" |
| Backbone trainable | Frozen (only the head trains) |
| Input | 224×224×3 |
| Optimiser | Adam |
| Loss | CategoricalCrossentropy |
| Callbacks | EarlyStopping(patience 3), ReduceLROnPlateau |

### Backbone migration (B7 → B0)

The reference used **EfficientNetB7** (~66 M parameters). For Raspberry Pi
deployment that is impractical at float32. We switched to **EfficientNetB0**
(~4 M parameters, same 224×224 stem, same 38-class softmax head).

Nothing else changed: same input size, same class count, same
`include_top=False`, same frozen backbone, same GAP + Dense(38, softmax).

Set `model.backbone` to any of `EfficientNetB0`…`B7` in `config.yaml`; an
unknown name fails loudly rather than silently using the wrong network.

## Preprocessing contract

This is subtle and was a real bug, so it is documented precisely.

Keras EfficientNet graphs contain their **own** `Rescaling(1/255)` stem plus an
ImageNet `Normalization`, and expect raw **`[0, 255]`** pixels. Keras states
this: *"input preprocessing is included as part of the model (as a `Rescaling`
layer)… EfficientNet models expect their inputs to be float tensors of pixels
with values in the `[0-255]` range."*

The production module `ai_disease_detection/src/preprocessing.py` divides by
255 and sends **`[0, 1]`**. Those two facts have to be reconciled, and the
arrangement is:

| Stage | Transformation | Range seen by backbone |
|---|---|---|
| Training (`train.py`) | raw pixels, no external rescale | `[0, 255]` |
| Evaluation (`evaluate.py`) | raw pixels, no external rescale | `[0, 255]` |
| ONNX input (production) | `x / 255` | `[0, 1]` |
| ONNX graph prefix (`export_onnx.py`) | `Rescaling(255)` | `[0, 255]` |
| Keras EfficientNet stem | `Rescaling(1/255)` + `Normalization` | normalised |

Both paths compute `x / 255` exactly once before the backbone, so training and
inference see identical tensors.

An earlier revision applied an external `Rescaling(1/255)` **and** let Keras do
its own, scaling every pixel twice — roughly 255× too small for ImageNet
weights. `verify_graph.py` demonstrates this empirically; regression tests in
`tests/test_preprocessing_contract.py` now fail if any second scaling returns.

Run `python verify_graph.py` on the GPU machine to confirm the graph behaviour
directly rather than trusting this document.

### Reference facts vs our results

The reference repository contains **no** weights, **no** accuracy figures, and
**no** performance data. Its model path is literally
`'path_to_your_model.tflite'`. Anything this pipeline reports about
accuracy belongs **only** to a model we train. The reference author's results
are unknown and are not claimed.

---

## Requirements

**A GPU machine. Google Colab is the intended target.**

```python
# Colab: Runtime > Change runtime type > TPU/GPU
!nvidia-smi                       # confirm a GPU is present
```

```bash
pip install -r requirements.txt
```

Do **not** run this on the Raspberry Pi: training needs a GPU. Even with our
EfficientNetB0 (4 M parameters) a CPU-only Pi would need many hours per epoch
on a 70k-image dataset.

---

## Dataset

### Kaggle (matches the reference exactly)

```bash
pip install -y kaggle
mkdir -p ~/.kaggle
# Upload your kaggle.json to ~/.kaggle/
kaggle datasets download -d vipoooool/new-plant-diseases-dataset
unzip -q new-plant-diseases-dataset.zip -d data/
```

### Expected layout

```
data/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/
├── train/
│   ├── Apple___Apple_scab/       *.jpg
│   ├── Apple___Black_rot/
│   └── ...                       38 folders
├── valid/
│   └── ...                       38 folders
└── test/                         38 folders (optional)
```

Then point `dataset.root` in `config.yaml` at that directory.

### Class ordering is not assumed

`ai_disease_detection/models/labels.txt` is the **single source of truth** for
class indices. `validate_dataset.py` compares the dataset's folder names
against it using a normalisation that tolerates spacing and separator
differences (`Pepper,_bell___Bacterial_spot` ≡ `Pepper bell Bacterial spot`),
then writes an explicit `output/label_mapping.json`.

If the orderings differ, the mapping is applied — or the run aborts when
`require_exact_label_order` is true. Class index *i* always means the same
disease in training and in production.

---

## ONNX layout decision

The production pipeline feeds **NCHW** (`ai_disease_detection/src/preprocessing.py`
returns `(1, 3, 224, 224)`), while Keras models are NHWC internally.

With `export.export_nchw: true` (default) the exporter wraps the graph:

```
input [-1,3,224,224] --transpose--> original Keras model --> output [-1,38]
```

so the transpose lives in the graph and the runtime stays a straight feed. The
alternative (`export_nchw: false`) exports raw NHWC; `onnx_engine.py` detects
this from the input shape and transposes in Python. Either works — the exporter
verifies the emitted signature and refuses to claim success on a mismatch.

**Preprocessing stays outside the graph.** The model expects pixels already in
`[0, 1]`, matching `preprocessing.normalize(..., "divide_255")`. During training
the `/255` is a `Rescaling(1/255)` layer so the two cannot drift apart.

---

## Pipeline

### 1. Validate the dataset

```bash
python validate_dataset.py --save-mapping
```

Reports class count, per-split image counts, class-to-index mapping, missing and
unexpected classes. Fails on count mismatch or an unmappable class.

### 2. Train

```bash
python train.py --batch-size 32 --epochs 50
```

Tune `batch_size` for Colab VRAM (8–16 on a T4). Extras:

```bash
python train.py --unfreeze-layers 100   # partial fine-tune (slower)
python train.py --unfreeze              # full fine-tune (much slower)
```

### 3. Evaluate

```bash
python evaluate.py --split test
```

The Kaggle archive ships no class-structured `test/` split; with that dataset
pass `--split valid` instead (that is what the recorded run did). The report
then states the figures are validation-set and `metrics.json` records
`is_true_test_set: false`.

Writes `output/metrics.json` and `output/evaluation.txt` with accuracy,
macro precision/recall/F1, weighted F1, top-5 accuracy, per-class figures, and
the full confusion matrix.

### 4. Export to ONNX

```bash
python export_onnx.py --opset 13
```

Verifies the input/output signature and runs a smoke test.

### 5. Validate Keras vs ONNX

```bash
python validate_onnx.py --num-images 25 --max-top1-diff 0.01
# Kaggle dataset (no usable test/ split):
python validate_onnx.py --split valid
```

Compares top-1 class, top-1 probability, and top-5 overlap. Exits non-zero on
any class mismatch or tolerance breach.

### 6. Deploy to the Pi

```bash
scp output/disease_model.onnx \
    agridrone123@192.168.29.72:/home/agridrone123/agridrone/ai_disease_detection/models/
```

Then on the Pi:

```bash
cd ~/agridrone/ai_disease_detection
../.venv/bin/python scripts/image_inference.py --image leaf.jpg
../.venv/bin/python scripts/live_inference.py --camera 0
```

### 7. Measure Pi performance

```bash
cd ~/agridrone/ai_disease_detection
../.venv/bin/python - <<'PY'
import time
from ai_disease.config import load_config
from ai_disease.pipeline import build_engine, run_inference
from ai_disease.preprocessing import load_image
cfg = load_config()
t0 = time.perf_counter(); engine = build_engine(cfg)
print(f"model load: {time.perf_counter()-t0:.2f} s")
frame = load_image("leaf.jpg")                    # a real capture, not a flat frame
run_inference(frame, engine, cfg)                       # warm up
lat = []
for _ in range(30):
    lat.append(run_inference(frame, engine, cfg).inference_ms)
avg = sum(lat)/len(lat)
print(f"mean {avg:.1f} ms  min {min(lat):.1f}  max {max(lat):.1f}")
print(f"=> {1000/avg:.1f} FPS")
PY
```

Measured 2026-10-08 on a **Raspberry Pi 4 Model B** (4 cores, 3.7 GB RAM,
kernel `6.18.50+rpt-rpi-v8`), onnxruntime 1.30.0, `disease_model.onnx`
(EfficientNetB0, 2-epoch smoke-test weights):

| Metric | Value |
|---|---|
| Model load | 0.28 s |
| Inference mean | 96.6 ms |
| Inference min / max | 94.8 / 106.2 ms |
| Throughput | ≈ 10.4 FPS |

Flat/gray frames are rejected by the image-quality gate before the model runs,
so benchmark against a real capture (or pass `--log`-style real frames).

---

## Files

```
ai_disease_training/
├── README.md
├── requirements.txt
├── config.yaml
├── common.py            config, dataset discovery, label mapping
├── validate_dataset.py  fail-fast dataset check
├── train.py             EfficientNetB0 training (backbone from config)
├── evaluate.py          metrics + confusion matrix
├── export_onnx.py       Keras -> ONNX with layout handling
├── validate_onnx.py     Keras vs ONNX agreement
└── output/              created on first run
    ├── model2.h5
    ├── disease_model.onnx
    ├── class_names.txt
    ├── label_mapping.json
    ├── metrics.json
    ├── evaluation.txt
    └── training_config.yaml
```

Do not commit `model2.h5` or `disease_model.onnx` to Git without being asked.

---

## Documented deviations from the reference

0. **Backbone is EfficientNetB0, not B7.** Chosen for Raspberry Pi inference
   speed. Architecture shape, input size, class count and head are unchanged.
1. **Preprocessing is split, not duplicated.** The reference relied on Keras's
   implicit dataset scaling. We feed raw `[0, 255]` to the backbone and prefix
   the exported graph with `Rescaling(255)` so the ONNX input remains `[0, 1]`
   for production. See "Preprocessing contract" above.
2. **Explicit class-index mapping** from `labels.txt`. The reference relied on
   Keras's directory-alphabetical ordering, which silently breaks if folder
   names change.
3. **`class_names` passed explicitly** to `image_dataset_from_directory` so the
   ordering is pinned by the mapping, not by directory listing.
4. **Confusion matrix and per-class metrics** added — the reference computed
   F1 in a notebook but never saved it.
5. **TF seed** set for reproducibility; GPU kernels remain non-deterministic.
6. **Frozen backbone is the default**, as in the reference. The reference's
   commented-out `Dense(256)` and BatchNorm layers were not restored, matching
   the architecture that actually produced its model.

## Not carried over from the reference

YOLOv5 weed detection, ArduPilot SITL/parameters/MAVProxy, arm/disarm logic,
the pre-arm check script, Colab drive/Kaggle plumbing, and its erroneous raw
0–255 live-inference preprocessing (which contradicts its own training
pipeline).

---

## Limitations

- **Classification only.** No object localisation or segmentation: the model
  assumes one leaf fills the frame. It cannot find a diseased leaf in a canopy.
- **No GPS.** Results carry `latitude`/`longitude`/`altitude` as `null`; a
  future telemetry bridge would fill them. This module has no MAVLink.
- **No flight control.** The AI cannot arm, land, change mode, or write PX4
  parameters. It produces a class name and a score, nothing more.
- **Performance is hardware-dependent and not yet measured on the Pi.** The
  model is EfficientNetB0 float32 at 224×224; run the step-7 script on the
  Pi before claiming real-time. If too slow, the
  inference interface accepts a lighter model — an int8 MobileNetV3 or
  EfficientNet-lite trained on the same 38 classes — with no code change.
- **Confidence is not a field guarantee.** A softmax score measures agreement
  with training data, not real-world diagnostic certainty. Get an
  agronomist's review before acting on any result.
- **Accuracy figures come from a 2-epoch smoke test on the validation
  split.** They verify the pipeline runs end-to-end, not model quality. A full
  training run and a held-out test set are needed before quoting accuracy.