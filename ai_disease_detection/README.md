# Crop Disease Detection AI (Standalone)

Offline, Raspberry-Pi-compatible plant disease classification. Runs on the Pi as
a companion to Pixhawk/PX4. **This module has no flight-control authority.**

> **Software framework: implemented and tested.**
> **Real disease inference: NOT VERIFIED** — the reference repository contains
> no trained model weights, so no real classification has been performed.
> Any output from the `mock` backend is synthetic and proves nothing about
> recognition accuracy.

---

## Model status

**Put your trained model here:**

```
ai_disease_detection/models/disease_model.onnx
```

Until then, running real inference fails with an explicit message:

```
Model file not found: models/disease_model.onnx

The reference repository does not contain trained model weights.
Please provide a converted ONNX model before running real inference.
```

The module will **not** fabricate weights, and `mock` is never used in the real
inference path.

---

## Quick start

```bash
cd ai_disease_detection
pip install -r requirements.txt

# Single image
python scripts/image_inference.py --image leaf.jpg

# Live camera with FPS + latency
python scripts/live_inference.py --camera 0
```

---

## What was verified from the reference repository

Read directly from
[isupercoder401/Agricultural-Drone-For-Crop-Monitoring-and-Early-Disease-Detection](https://github.com/isupercoder401/Agricultural-Drone-For-Crop-Monitoring-and-Early-Disease-Detection):

| Item | Source | Value |
|---|---|---|
| Architecture | `project.py:60-67` | `EfficientNetB7(include_top=False)` → `GlobalAveragePooling2D` → `Dense(38, softmax)` |
| Backbone | `project.py:61` | `model.trainable = False` — frozen; only the 38-unit head trains |
| Input size | `project.py:48` | `image_size=(224, 224)`, 3 channels |
| Classes | `project.py:66` | 38 |
| Dataset | `project.py:15` | `vipoooool/new-plant-diseases-dataset` (PlantVillage, augmented) |
| Optimiser | `project.py:68` | Adam, CategoricalCrossentropy |
| Labels | `Live_Inference:6-45` | 38 PlantVillage `Crop___Disease` names |
| TFLite intent | `project.py:100-105` | `TFLiteConverter.from_keras_model(model2)` |

The 38 labels are carried over **verbatim** into `models/labels.txt` and verified
by `tests/test_labels.py`.

### A preprocessing bug we did not copy

`Live_Inference:63` feeds raw 0–255 pixels:

```python
input_data = np.expand_dims(frame, axis=0).astype(np.float32)
```

but its own training used Keras `image_size=(224,224)`, which normalises to
`[0,1]`. The line `#im = im / 255` is commented out at `project.py:194`, so the
author appears to have noticed. **We follow the training pipeline** (`divide_255`),
because matching how the model was trained matters more than reproducing that
script. Normalisation is configurable in `config.yaml`.

---

## What is NOT in the reference repository

- **Trained model weights** — no `.tflite`, `.onnx`, or `.h5` anywhere in the tree
- **An ONNX model** — the repo predates ONNX export
- **Trustworthy accuracy results** — accuracy was computed in a Colab notebook and never committed
- **Raspberry Pi performance numbers** — no benchmark is published
- **The converted model path** — literally `model_path='path_to_your_model.tflite'`

---

## Deliberately NOT copied

| Reference component | Why excluded |
|---|---|
| `sitl_python_script` | ArduPilot SITL — forbidden by project rules |
| `Arm_Disarm_Drone` | Flight-control authority — must stay with PX4 |
| `DRONE PARAMETERS` | ArduPilot parameters — PX4/Pixhawk is our stack |
| `pre_checks` | Pixhawk pre-arm logic via pymavlink — not our concern |
| YOLOv5 crop/weed detection | Out of scope; would add torch (~200 MB) |
| `import tensorflow` at runtime | ONNX Runtime suffices, avoiding ~600 MB |
| Google Colab / Kaggle training code | Training is out of scope |

---

## Our implementation

- **Backend:** ONNX Runtime (`onnxruntime` has a cp313 manylinux aarch64 wheel)
- **Python:** 3.13, Raspberry Pi OS / Debian
- **Model-independent interface** so a lighter model can be dropped in later
- **Configurable preprocessing** (`divide_255`, `imagenet`, `none`)
- **Camera abstraction** — USB webcam, video file, or Pi CSI camera without
  touching the inference code
- **JSON results**, optional logging, no web server

### Why ONNX and not TFLite

The reference converted to TFLite, but on your Pi:

```
tflite-runtime : ERROR — no matching distribution (no cp313 aarch64 wheel)
onnxruntime     : OK — onnxruntime-1.30.0-cp313-cp313-manylinux_2_28_aarch64.whl
```

Full TensorFlow would work but adds ~600 MB. So: convert to ONNX. `src/tflite_engine.py`
exists as a documented stub so the backend slot is real, but it is **not imported**
by the default path and not covered by tests.

---

## Architecture

```
Camera ──> numpy BGR frame ──> Preprocessor ──> InferenceEngine ──> DetectionResult
                (camera.py)     (preprocessing.py)  (inference.py)     (results.py)
```

The engine never touches OpenCV or cameras, so swapping a USB webcam for a CSI
sensor cannot affect inference.

```
src/
  config.py         YAML config -> dataclasses
  preprocessing.py  BGR->RGB, resize, normalise, NCHW batch
  inference.py      InferenceEngine ABC, factory, error types
  onnx_engine.py    primary backend
  mock_engine.py    deterministic engine for tests
  tflite_engine.py  documented stub, not wired in
  camera.py         USB / file / picamera2 backends
  results.py        DetectionResult, thresholding, JSON, annotation
  pipeline.py       shared labels->engine->result path
scripts/
  image_inference.py
  live_inference.py
```

---

## Commands

**Single image**

```bash
python scripts/image_inference.py --image leaf.jpg
python scripts/image_inference.py --image leaf.jpg --confidence 0.80
python scripts/image_inference.py --image leaf.jpg --json
python scripts/image_inference.py --image leaf.jpg --log
```

Output:

```
Disease: Tomato___Late_blight
Confidence: 94.2%
Inference: 42.3 ms
Healthy: no
Model: disease_model.onnx  (onnx)
```

**Live camera**

```bash
python scripts/live_inference.py --camera 0
python scripts/live_inference.py --camera 0 --confidence 0.70
python scripts/live_inference.py --camera 0 --no-display --log
python scripts/live_inference.py --backend-cam file --path clip.mp4 --max-frames 100
```

On-screen: class name, confidence, latency, FPS. `q` quits.

---

## Result schema

```json
{
  "class_id": 30,
  "class_name": "Tomato___Late_blight",
  "confidence": 0.94,
  "healthy": false,
  "disease": "Late blight",
  "crop": "Tomato",
  "reliable": true,
  "reason": null,
  "inference_ms": 42.3,
  "preprocess_ms": 3.1,
  "total_ms": 45.4,
  "timestamp": "2026-01-01T12:00:00+00:00",
  "latitude": null,
  "longitude": null,
  "altitude": null
}
```

### Confidence handling

A weak prediction is reported honestly rather than forced:

```json
{
  "class_name": "Unknown",
  "confidence": 0.43,
  "reason": "below_confidence_threshold",
  "reliable": false
}
```

`latitude`/`longitude`/`altitude` are reserved for a future telemetry bridge and
always `null`. This module has no MAVLink code.

---

## Raspberry Pi installation

```bash
sudo apt update && sudo apt upgrade
sudo apt install -y python3-venv python3-pip

cd ~/agridrone/ai_disease_detection
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Verify:

```bash
python -c "import onnxruntime, cv2, numpy; print(onnxruntime.__version__)"
python -m pytest tests/ -q
```

CSI camera support needs one extra package:

```bash
sudo apt install -y python3-picamera2
```

---

## Benchmarking

Once a real model is present:

```bash
python scripts/live_inference.py --camera 0 --no-display --max-frames 300
```

Reports per-frame `inference_ms` and rolling FPS. For a focused measurement:

```bash
python - <<'PY'
import time, numpy as np
from src.config import load_config
from src.pipeline import build_engine, run_inference
cfg = load_config()
engine = build_engine(cfg)
frame = np.full((480, 640, 3), 120, dtype=np.uint8)
run_inference(frame, engine, cfg)                    # warm up
lat = []
for _ in range(50):
    r = run_inference(frame, engine, cfg)
    lat.append(r.inference_ms)
print(f"mean {sum(lat)/len(lat):.1f} ms   "
      f"min {min(lat):.1f}   max {max(lat):.1f}   "
      f"=> {1000/(sum(lat)/len(lat)):.1f} FPS")
PY
```

RAM and CPU:

```bash
/usr/bin/time -v python scripts/image_inference.py --image leaf.jpg 2>&1 | grep -i maximum
top -p $(pgrep -f image_inference.py)
```

**No performance claim is made here.** EfficientNetB7 at 224×224 is a heavy
graph (~66M parameters); real-time on a Pi 4 is unlikely without quantisation.
If it is too slow, the interface accepts a lighter TFLite or ONNX model with no
code change. An int8-quantised MobileNetV3 or EfficientNet-lite trained on the
same 38 classes is the practical path. **Measure before deciding.**

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `Model file not found` | Place the model at `models/disease_model.onnx` |
| `onnxruntime could not load` | Wrong conversion. Confirm the target is **PX4 FMU v2.x**-style Keras export, not a quantised or dynamic-shape variant |
| `label count mismatch` | `labels.txt` rows must equal the model's output classes (38) |
| `expected a 3D or 4D input tensor` | Export expects NHWC or 3-D input; adjust `_inspect_input` |
| `could not open camera device 0` | Check `/dev/video*`; try `--camera 1` |
| `picamera2 is not installed` | `sudo apt install python3-picamera2` |
| `Unknown` on every image | Threshold too high for the model, or preprocessing mismatch — check `config.yaml` `preprocessing.type` |
| Latency very high | Expected for EfficientNetB7 float32; quantise or use a smaller model |

Debug a model graph:

```bash
python -c "
import onnxruntime as ort
s = ort.InferenceSession('models/disease_model.onnx', providers=['CPUExecutionProvider'])
i, o = s.get_inputs()[0], s.get_outputs()[0]
print('input ', i.name, i.shape, i.type)
print('output', o.name, o.shape, o.type)
"
```

---

## Tests

```bash
python -m pytest tests/ -q
```

Covers preprocessing (RGB, resize, normalisation), labels (38 verified against
the reference), results (thresholding, healthy flag, JSON), and engines
(mock, missing model, invalid model, camera fallback).

Tests use `MockInferenceEngine` and pass with **no model file present**. They
prove plumbing only — **not** disease recognition.

A guard test asserts no MAVLink/PX4/ArduPilot/ROS imports exist anywhere in
`src/` or `scripts/`.

---

## Future: Pixhawk telemetry bridge

Deliberately not implemented. When wanted, a **separate** service will read
MAVLink telemetry from Pixhawk and stamp results with position:

```
Pixhawk ──> MAVLink ──> telemetry service ──> AI result
```

Keep the AI module a pure consumer: it produces results and never commands.

---

## Next step

Obtain a trained model, then convert to ONNX:

1. Retrain following `project.py`, or use a PlantVillage-trained
   EfficientNetB7 checkpoint (38 classes, 224×224, softmax)
2. Convert on a machine with TensorFlow:
   ```python
   import tensorflow as tf
   m = tf.keras.models.load_model("model2.h5")
   tf.lite.TFLiteConverter.from_keras_model(m)  # inspect the graph
   ```
   For ONNX, use `tf2onnx`:
   ```bash
   pip install tf2onnx
   python -c "
   import tf2onnx, tensorflow as tf
   m = tf.keras.models.load_model('model2.h5')
   tf2onnx.convert.from_keras(m, input_path='model2.onnx', opset=13,
                              output_path='disease_model.onnx')"
   ```
3. Copy to `ai_disease_detection/models/disease_model.onnx`
4. Verify input shape is `[-1, 3, 224, 224]` and output is `[-1, 38]`
5. Benchmark, then decide whether EfficientNetB7 is fast enough